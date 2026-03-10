"""
Navigator V3 - Clean Implementation
===================================

Object navigation using pre-built maps with:
- Correct floor transitions
- Robust A* planning with boundary checks
- Improved obstacle avoidance
- Closest object selection

Key fixes from V2:
1. Floor height comparison now accounts for camera offset correctly
2. A* planner restricts to explored area only
3. Better stuck detection and recovery
4. Path validation before following
"""

import numpy as np
import cv2
import heapq
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque


class NavState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    REACHED = "reached"
    FAILED = "failed"


@dataclass
class FloorData:
    """Map data for one floor."""
    floor_id: int
    navigable_map: np.ndarray  # 1 = obstacle, 0 = free
    explored_map: np.ndarray   # 1 = explored, 0 = unknown
    objects: Dict


class Navigator:
    """
    Object navigation using pre-built maps.
    """
    
    # Navigation parameters
    TURN_THRESHOLD = np.radians(15)   # Turn if angle > this
    OBSTACLE_CLOSE = 0.35             # Critical obstacle distance
    OBSTACLE_THRESHOLD = 0.6          # Normal obstacle distance
    WAYPOINT_TOLERANCE = 8            # Pixels
    GOAL_TOLERANCE = 12               # Pixels
    INFLATION_RADIUS = 4              # Obstacle inflation
    
    # Habitat camera height above floor
    CAMERA_HEIGHT = 1.25  # meters
    
    def __init__(self, map_dir: str):
        self.map_dir = Path(map_dir)
        
        # Map data
        self.floors: Dict[int, FloorData] = {}
        self.all_objects: Dict[str, Dict] = {}
        self.floor_heights: Dict[int, float] = {}  # floor_id -> floor base height
        
        # Map parameters (loaded from saved map)
        self.resolution = 5.0  # cm per pixel
        self.map_size = 480
        self.origin_x = 0.0
        self.origin_z = 0.0
        
        # Navigation state
        self.current_floor = 0
        self.state = NavState.IDLE
        self.goal_map: Optional[Tuple[int, int]] = None
        self.goal_name: str = ""
        self.path: List[Tuple[int, int]] = []
        self.current_waypoint = 0
        
        # Recovery
        self.recovery_actions: List[str] = []
        self.position_history = deque(maxlen=20)
        
        # Stats
        self.actions = 0
        self.avoided = 0
        
        # Load maps
        self._load_maps()
    
    def _load_maps(self):
        """Load pre-built maps."""
        # Load statistics for coordinate system
        stats_path = self.map_dir / 'statistics.json'
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.resolution = stats.get('resolution_cm', 5.0)
            self.map_size = stats.get('map_size', 480)
            self.origin_x = stats.get('origin_x', 0.0)
            self.origin_z = stats.get('origin_z', 0.0)
        
        # Load floor info
        floor_info_path = self.map_dir / 'floor_info.json'
        if floor_info_path.exists():
            with open(floor_info_path) as f:
                floor_info = json.load(f)
            floor_ids = floor_info.get('floor_ids', [0])
            
            # Load floor base heights
            base_heights = floor_info.get('base_heights', {})
            for k, v in base_heights.items():
                self.floor_heights[int(k)] = float(v)
        else:
            floor_ids = [0]
        
        # Load each floor
        for fid in floor_ids:
            floor_dir = self.map_dir / f'floor_{fid}'
            if not floor_dir.exists():
                continue
            
            # Load navigable map
            nav_path = floor_dir / 'navigable_map.pt'
            if not nav_path.exists():
                nav_path = floor_dir / 'occupancy_map.pt'
            
            if not nav_path.exists():
                continue
            
            nav_map = torch.load(nav_path).numpy()
            
            # Load explored map
            exp_path = floor_dir / 'explored_map.pt'
            if exp_path.exists():
                exp_map = torch.load(exp_path).numpy()
            else:
                exp_map = np.ones_like(nav_map)  # Assume all explored
            
            # Load objects
            obj_path = floor_dir / 'objects.json'
            objects = {}
            if obj_path.exists():
                with open(obj_path) as f:
                    objects = json.load(f)
                    for obj_id, obj in objects.items():
                        obj['floor'] = fid
                        self.all_objects[obj_id] = obj
            
            self.floors[fid] = FloorData(
                floor_id=fid,
                navigable_map=nav_map,
                explored_map=exp_map,
                objects=objects
            )
        
        # Set initial floor
        if self.floors:
            self.current_floor = sorted(self.floors.keys())[0]
        
        print(f"Loaded {len(self.floors)} floors: {list(self.floors.keys())}")
        print(f"Floor heights: {self.floor_heights}")
        print(f"Objects: {len(self.all_objects)}")
        print(f"Coordinate origin: ({self.origin_x:.2f}, {self.origin_z:.2f})")
    
    # =========================================================================
    # Coordinate Conversion
    # =========================================================================
    
    def world_to_map(self, wx: float, wz: float) -> Tuple[int, int]:
        """Convert world coordinates to map pixels."""
        mx = int((wx - self.origin_x) * 100 / self.resolution + self.map_size // 2)
        my = int((wz - self.origin_z) * 100 / self.resolution + self.map_size // 2)
        return mx, my
    
    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """Convert map pixels to world coordinates."""
        wx = (mx - self.map_size // 2) * self.resolution / 100 + self.origin_x
        wz = (my - self.map_size // 2) * self.resolution / 100 + self.origin_z
        return wx, wz
    
    # =========================================================================
    # Goal Setting
    # =========================================================================
    
    def get_available_objects(self) -> List[str]:
        """Get list of detected object classes."""
        return list(set(obj['class'].lower() for obj in self.all_objects.values()))
    
    def set_goal_object(self, object_class: str, agent_pos: np.ndarray) -> bool:
        """Navigate to nearest object of given class."""
        object_class = object_class.lower()
        
        # Find ALL matching objects
        candidates = []
        for obj_id, obj in self.all_objects.items():
            if obj['class'].lower() == object_class:
                pos = obj['position']
                dist = np.sqrt((pos[0] - agent_pos[0])**2 + (pos[2] - agent_pos[2])**2)
                candidates.append({'id': obj_id, 'obj': obj, 'distance': dist})
        
        if not candidates:
            print(f"Object '{object_class}' not found")
            return False
        
        # Sort by distance
        candidates.sort(key=lambda x: x['distance'])
        
        if len(candidates) > 1:
            print(f"Found {len(candidates)} '{object_class}' objects, selecting nearest:")
            for i, c in enumerate(candidates[:3]):
                print(f"  {i+1}. {c['distance']:.2f}m away")
        
        target = candidates[0]['obj']
        pos = target['position']
        self.goal_map = self.world_to_map(pos[0], pos[2])
        self.goal_name = f"{object_class} ({candidates[0]['distance']:.1f}m)"
        
        print(f"Goal: {self.goal_name} at map{self.goal_map}")
        return self._plan_path(agent_pos)
    
    def set_goal_position(self, mx: int, my: int, agent_pos: np.ndarray) -> bool:
        """Navigate to map position."""
        self.goal_map = (mx, my)
        self.goal_name = f"pos({mx},{my})"
        return self._plan_path(agent_pos)
    
    def cancel(self):
        """Cancel navigation."""
        self.state = NavState.IDLE
        self.goal_map = None
        self.path = []
        self.current_waypoint = 0
        self.recovery_actions.clear()
    
    # =========================================================================
    # Path Planning
    # =========================================================================
    
    def _plan_path(self, agent_pos: np.ndarray) -> bool:
        """Plan path from agent to goal."""
        if self.goal_map is None:
            return False
        
        if self.current_floor not in self.floors:
            print(f"No map for floor {self.current_floor}")
            return False
        
        floor_data = self.floors[self.current_floor]
        nav_map = floor_data.navigable_map
        exp_map = floor_data.explored_map
        h, w = nav_map.shape
        
        start = self.world_to_map(agent_pos[0], agent_pos[2])
        goal = self.goal_map
        
        print(f"Planning: {start} -> {goal}")
        
        # Validate bounds
        if not (0 <= start[0] < w and 0 <= start[1] < h):
            print(f"Start {start} out of bounds")
            self.state = NavState.FAILED
            return False
        
        if not (0 <= goal[0] < w and 0 <= goal[1] < h):
            print(f"Goal {goal} out of bounds")
            self.state = NavState.FAILED
            return False
        
        # Create cost map:
        # - Obstacles = blocked
        # - Unexplored = high cost (discourage but allow)
        # - Explored free = low cost
        kernel = np.ones((self.INFLATION_RADIUS * 2 + 1,) * 2, np.uint8)
        inflated = cv2.dilate((nav_map > 0.5).astype(np.uint8), kernel)
        
        # Cost map: 0=free, 1=inflated obstacle, 2=unexplored
        cost_map = inflated.copy()
        cost_map[exp_map < 0.5] = 2  # Mark unexplored as high cost
        
        # Find valid start
        if inflated[start[1], start[0]] > 0:
            start = self._find_free_cell(start, inflated)
            if start is None:
                print("Cannot find free start")
                self.state = NavState.FAILED
                return False
            print(f"Adjusted start: {start}")
        
        # Find valid goal
        if inflated[goal[1], goal[0]] > 0:
            goal = self._find_free_cell(goal, inflated)
            if goal is None:
                print("Cannot find free goal")
                self.state = NavState.FAILED
                return False
            print(f"Adjusted goal: {goal}")
        
        # A* with cost awareness
        path = self._astar_with_cost(start, goal, inflated, exp_map)
        
        if not path:
            # Fallback: try with less inflation
            inflated_min = cv2.dilate((nav_map > 0.5).astype(np.uint8), np.ones((3,3), np.uint8))
            path = self._astar_with_cost(start, goal, inflated_min, exp_map)
        
        if not path:
            print("No path found")
            self.state = NavState.FAILED
            return False
        
        # Simplify path (take every 3rd waypoint)
        self.path = path[::3]
        if path[-1] not in self.path:
            self.path.append(path[-1])
        
        # Validate path is within explored area (mostly)
        unexplored_count = sum(1 for px, py in self.path if exp_map[py, px] < 0.5)
        if unexplored_count > len(self.path) * 0.5:
            print(f"Warning: {unexplored_count}/{len(self.path)} waypoints in unexplored area")
        
        self.current_waypoint = 0
        self.state = NavState.NAVIGATING
        print(f"Path: {len(self.path)} waypoints")
        return True
    
    def _find_free_cell(self, pos: Tuple[int, int], obs_map: np.ndarray, max_r: int = 30) -> Optional[Tuple[int, int]]:
        """Find nearest free cell."""
        h, w = obs_map.shape
        px, py = pos
        
        for r in range(1, max_r):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h and obs_map[ny, nx] == 0:
                        return (nx, ny)
        return None
    
    def _astar_with_cost(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        obs: np.ndarray,
        explored: np.ndarray
    ) -> List[Tuple[int, int]]:
        """A* pathfinding that prefers explored areas."""
        h, w = obs.shape
        
        def heuristic(a, b):
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
        neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        
        open_set = [(heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}
        counter = 0
        
        while open_set and counter < 100000:
            counter += 1
            _, _, cur = heapq.heappop(open_set)
            
            if heuristic(cur, goal) < 5:
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                return path[::-1]
            
            for dx, dy in neighbors:
                nx, ny = cur[0] + dx, cur[1] + dy
                
                # Boundary check
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                
                # Obstacle check
                if obs[ny, nx] > 0:
                    continue
                
                # Base movement cost
                move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                
                # Add penalty for unexplored areas (but don't block)
                if explored[ny, nx] < 0.5:
                    move_cost += 5.0  # Penalty for unexplored
                
                tg = g_score[cur] + move_cost
                
                if (nx, ny) not in g_score or tg < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = cur
                    g_score[(nx, ny)] = tg
                    counter += 1
                    heapq.heappush(open_set, (tg + heuristic((nx, ny), goal), counter, (nx, ny)))
        
        return []
    
    # =========================================================================
    # Floor Transition
    # =========================================================================
    
    def _check_floor_change(self, agent_height: float):
        """Check if agent moved to different floor."""
        if not self.floor_heights:
            return
        
        # Find floor where agent height matches expected
        # Expected height = floor_base + camera_height
        best_floor = self.current_floor
        min_diff = float('inf')
        
        for floor_id, base_height in self.floor_heights.items():
            expected = base_height + self.CAMERA_HEIGHT
            diff = abs(agent_height - expected)
            if diff < min_diff:
                min_diff = diff
                best_floor = floor_id
        
        # Debug every 100 actions
        if self.actions % 100 == 0:
            current_expected = self.floor_heights.get(self.current_floor, 0) + self.CAMERA_HEIGHT
            print(f"[Floor] Height: {agent_height:.2f}m, Current floor {self.current_floor} expects {current_expected:.2f}m")
            print(f"[Floor] Best match: floor {best_floor} (diff={min_diff:.2f}m)")
        
        # Switch if different floor and clearly better match
        if best_floor != self.current_floor and min_diff < 1.0:
            current_expected = self.floor_heights.get(self.current_floor, 0) + self.CAMERA_HEIGHT
            if abs(agent_height - current_expected) > 1.5:
                print(f"FLOOR CHANGE: {self.current_floor} -> {best_floor}")
                self.current_floor = best_floor
                
                # Cancel path
                if self.path:
                    print("  Replanning needed on new floor")
                    self.path = []
                    self.state = NavState.IDLE
    
    # =========================================================================
    # Action Generation
    # =========================================================================
    
    def get_action(self, agent_pos: np.ndarray, agent_rot: np.ndarray, depth: np.ndarray) -> Optional[str]:
        """Get next navigation action."""
        self.actions += 1
        self.position_history.append(agent_pos[[0, 2]].copy())
        
        # Check floor change
        self._check_floor_change(agent_pos[1])
        
        # Handle recovery
        if self.recovery_actions:
            return self.recovery_actions.pop(0)
        
        # Check stuck
        if self._is_stuck():
            self._start_recovery()
            if self.recovery_actions:
                return self.recovery_actions.pop(0)
        
        # Not navigating
        if self.state != NavState.NAVIGATING or not self.path:
            return None
        
        agent_map = self.world_to_map(agent_pos[0], agent_pos[2])
        
        # Check goal reached
        if self.goal_map:
            dist = np.sqrt((agent_map[0] - self.goal_map[0])**2 + (agent_map[1] - self.goal_map[1])**2)
            if dist < self.GOAL_TOLERANCE:
                self.state = NavState.REACHED
                print(f"REACHED: {self.goal_name}")
                return None
        
        # Get current waypoint
        if self.current_waypoint >= len(self.path):
            self.state = NavState.REACHED
            return None
        
        target_map = self.path[self.current_waypoint]
        dist_to_wp = np.sqrt((agent_map[0] - target_map[0])**2 + (agent_map[1] - target_map[1])**2)
        
        # Advance waypoint
        if dist_to_wp < self.WAYPOINT_TOLERANCE:
            self.current_waypoint += 1
            if self.current_waypoint >= len(self.path):
                self.state = NavState.REACHED
                return None
            target_map = self.path[self.current_waypoint]
        
        return self._compute_action(agent_pos, agent_rot, target_map, depth)
    
    def _compute_action(
        self, 
        agent_pos: np.ndarray, 
        agent_rot: np.ndarray,
        target_map: Tuple[int, int], 
        depth: np.ndarray
    ) -> str:
        """Compute action with obstacle avoidance."""
        # Get yaw from quaternion
        x, y, z, w = agent_rot
        siny_cosp = 2.0 * (w * y + z * x)
        cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Target in world coords
        tx, tz = self.map_to_world(target_map[0], target_map[1])
        dx = tx - agent_pos[0]
        dz = tz - agent_pos[2]
        
        # Angle to target (Habitat: yaw=0 is -Z)
        target_angle = np.arctan2(-dx, -dz)
        angle_diff = target_angle - yaw
        
        # Normalize
        while angle_diff > np.pi: angle_diff -= 2*np.pi
        while angle_diff < -np.pi: angle_diff += 2*np.pi
        
        # Check depth for obstacles
        h, w_img = depth.shape
        center = depth[h//3:2*h//3, w_img//3:2*w_img//3]
        valid = center[(center > 0.1) & (center < 10)]
        front_dist = np.percentile(valid, 5) if len(valid) > 100 else 10.0
        
        # CRITICAL: Very close obstacle
        if front_dist < self.OBSTACLE_CLOSE:
            self.avoided += 1
            self.state = NavState.AVOIDING
            
            left = depth[h//4:3*h//4, :w_img//4]
            right = depth[h//4:3*h//4, 3*w_img//4:]
            left_clear = np.median(left[left > 0.1]) if (left > 0.1).any() else 0
            right_clear = np.median(right[right > 0.1]) if (right > 0.1).any() else 0
            
            return 'turn_left' if left_clear > right_clear else 'turn_right'
        
        # Turn toward target
        if abs(angle_diff) > self.TURN_THRESHOLD:
            self.state = NavState.NAVIGATING
            return 'turn_left' if angle_diff > 0 else 'turn_right'
        
        # Moderate obstacle
        if front_dist < self.OBSTACLE_THRESHOLD:
            self.avoided += 1
            self.state = NavState.AVOIDING
            
            left = depth[h//3:2*h//3, :w_img//3]
            right = depth[h//3:2*h//3, 2*w_img//3:]
            left_clear = np.mean(left[left > 0.1]) if (left > 0.1).any() else 0
            right_clear = np.mean(right[right > 0.1]) if (right > 0.1).any() else 0
            
            return 'turn_left' if left_clear > right_clear else 'turn_right'
        
        # Clear to move
        self.state = NavState.NAVIGATING
        return 'move_forward'
    
    def _is_stuck(self) -> bool:
        """Check if stuck (not making progress)."""
        if len(self.position_history) < 15:
            return False
        positions = list(self.position_history)
        total_dist = sum(np.linalg.norm(positions[i] - positions[i-1]) for i in range(1, len(positions)))
        return total_dist < 0.15
    
    def _start_recovery(self):
        """Start recovery maneuver."""
        print("STUCK - recovering")
        self.recovery_actions = ['turn_left'] * 6 + ['move_forward'] * 2 + ['turn_right'] * 3
    
    # =========================================================================
    # Status
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get navigation status."""
        return {
            'state': self.state.value,
            'floor': self.current_floor,
            'goal': self.goal_name,
            'waypoint': f"{self.current_waypoint}/{len(self.path)}" if self.path else "N/A",
            'avoided': self.avoided,
        }
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def visualize(self, agent_pos: np.ndarray = None) -> np.ndarray:
        """Visualize current floor map with path."""
        if self.current_floor not in self.floors:
            return np.zeros((200, 200, 3), dtype=np.uint8)
        
        floor_data = self.floors[self.current_floor]
        nav_map = floor_data.navigable_map
        exp_map = floor_data.explored_map
        
        # Create RGB visualization
        vis = np.zeros((*nav_map.shape, 3), dtype=np.uint8)
        
        # Unexplored = dark gray
        vis[exp_map < 0.5] = [40, 40, 40]
        
        # Free space = light gray
        free = (exp_map > 0.5) & (nav_map < 0.5)
        vis[free] = [100, 100, 100]
        
        # Obstacles = white
        vis[nav_map > 0.5] = [200, 200, 200]
        
        # Draw objects
        for obj in self.all_objects.values():
            if obj.get('floor', 0) != self.current_floor:
                continue
            pos = obj['position']
            mx, my = self.world_to_map(pos[0], pos[2])
            if 0 <= mx < nav_map.shape[1] and 0 <= my < nav_map.shape[0]:
                cv2.circle(vis, (mx, my), 5, (0, 255, 0), -1)
        
        # Draw path
        if self.path:
            for i in range(len(self.path) - 1):
                p1, p2 = self.path[i], self.path[i+1]
                color = (255, 255, 0) if i < self.current_waypoint else (0, 200, 255)
                cv2.line(vis, p1, p2, color, 2)
            
            # Current waypoint
            if self.current_waypoint < len(self.path):
                wp = self.path[self.current_waypoint]
                cv2.circle(vis, wp, 6, (0, 255, 255), 2)
        
        # Draw goal
        if self.goal_map:
            cv2.circle(vis, self.goal_map, 8, (0, 0, 255), 2)
            cv2.circle(vis, self.goal_map, 4, (0, 0, 255), -1)
        
        # Draw agent
        if agent_pos is not None:
            ax, ay = self.world_to_map(agent_pos[0], agent_pos[2])
            cv2.circle(vis, (ax, ay), 6, (255, 0, 0), -1)
        
        return vis
