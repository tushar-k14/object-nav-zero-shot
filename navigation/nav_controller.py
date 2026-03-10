"""
Navigation Controller
=====================

Unified navigation system that combines:
1. Global Planner - High-level path planning on semantic map
2. Local Planner - Collision avoidance and waypoint following
3. Scene Graph - Object/room-based goal specification

Architecture:
                    ┌─────────────────┐
                    │   User Goal     │
                    │ "go to toilet"  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Scene Graph    │
                    │ Find object loc │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Global Planner  │
                    │ A* on occ map   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Local Planner  │
                    │ Collision avoid │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Action      │
                    │ move/turn/stop  │
                    └─────────────────┘
"""

import numpy as np
import cv2
import heapq
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class GoalType(Enum):
    """Navigation goal types."""
    POSITION = "position"       # Go to (x, z) coordinates
    OBJECT = "object"           # Go to object class ("toilet", "chair")
    ROOM = "room"               # Go to room type ("kitchen", "bathroom")
    FRONTIER = "frontier"       # Explore nearest unknown area


class NavState(Enum):
    """Navigation state machine."""
    IDLE = "idle"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    RECOVERING = "recovering"
    REACHED = "reached"
    FAILED = "failed"


@dataclass
class NavGoal:
    """Navigation goal specification."""
    goal_type: GoalType
    target_name: str = None              # Object class or room type
    target_position: np.ndarray = None   # World [x, y, z]
    target_map: Tuple[int, int] = None   # Map coordinates
    tolerance: float = 0.5               # Arrival tolerance (meters)
    floor: int = 0


@dataclass
class NavPath:
    """Planned navigation path."""
    waypoints_map: List[Tuple[int, int]]      # Map coordinates
    waypoints_world: List[np.ndarray]         # World coordinates
    goal: NavGoal
    distance: float                           # Total path distance (meters)
    valid: bool = True


class NavigationController:
    """
    Main navigation controller.
    
    Usage:
        nav = NavigationController(semantic_mapper, scene_graph)
        
        # Set goal
        nav.set_goal_object("toilet")
        # or nav.set_goal_room("kitchen")
        # or nav.set_goal_position(np.array([5.0, 0, 3.0]))
        # or nav.set_goal_frontier()
        
        # In loop:
        action = nav.get_action(depth, agent_position, agent_rotation)
        if action:
            sim.step(action)
    """
    
    def __init__(
        self,
        semantic_mapper,
        scene_graph,
        # Global planning params
        path_inflation: int = 5,
        path_simplify: int = 3,
        # Local planning params
        obstacle_threshold: float = 0.45,
        waypoint_tolerance: float = 0.4,
        turn_threshold: float = 15.0,    # Degrees
        # Recovery params
        stuck_threshold: int = 15,
        stuck_movement: float = 0.15,
    ):
        self.mapper = semantic_mapper
        self.scene_graph = scene_graph
        
        # Global planning
        self.path_inflation = path_inflation
        self.path_simplify = path_simplify
        
        # Local planning
        self.obstacle_threshold = obstacle_threshold
        self.waypoint_tolerance = waypoint_tolerance
        self.turn_threshold_rad = np.radians(turn_threshold)
        
        # Recovery
        self.stuck_threshold = stuck_threshold
        self.stuck_movement = stuck_movement
        
        # State
        self.state = NavState.IDLE
        self.goal: Optional[NavGoal] = None
        self.path: Optional[NavPath] = None
        self.current_waypoint_idx = 0
        
        # History for stuck detection
        self.position_history = deque(maxlen=stuck_threshold)
        self.recovery_actions: List[str] = []
        self.stuck_count = 0
        
        # Stats
        self.total_actions = 0
        self.replans = 0
        self.collisions_avoided = 0
    
    # =========================================================================
    # Goal Setting
    # =========================================================================
    
    def set_goal_object(self, object_class: str) -> bool:
        """
        Set goal to navigate to nearest object of given class.
        
        Args:
            object_class: Object class name (e.g., "toilet", "chair", "bed")
            
        Returns:
            True if goal set successfully
        """
        object_class = object_class.lower()
        
        # Find object from scene graph
        objects = self.scene_graph.get_objects_by_class(object_class)
        
        # Also check mapper's tracked objects
        if not objects:
            confirmed = self.mapper.get_confirmed_objects()
            objects = [o for o in confirmed.values() 
                      if o.class_name.lower() == object_class]
        
        if not objects:
            print(f"No '{object_class}' found")
            self.state = NavState.FAILED
            return False
        
        # Get position of first object (TODO: find nearest)
        obj = objects[0]
        if hasattr(obj, 'mean_position'):
            pos = obj.mean_position
        elif hasattr(obj, 'position'):
            pos = obj.position
        else:
            print(f"Object has no position")
            return False
        
        # Create goal
        occ_map = self.mapper.occupancy_map
        map_pos = occ_map._world_to_map(pos[0], pos[2])
        
        self.goal = NavGoal(
            goal_type=GoalType.OBJECT,
            target_name=object_class,
            target_position=np.array(pos),
            target_map=map_pos,
            floor=getattr(obj, 'floor_level', self.mapper.current_floor)
        )
        
        print(f"Goal set: {object_class} at ({pos[0]:.1f}, {pos[2]:.1f})")
        return self._plan_path()
    
    def set_goal_room(self, room_type: str) -> bool:
        """Set goal to navigate to room center."""
        room_type = room_type.lower()
        
        # Get objects in room from scene graph
        room_objects = self.scene_graph.get_objects_in_room(room_type)
        
        if not room_objects:
            print(f"No '{room_type}' found")
            self.state = NavState.FAILED
            return False
        
        # Compute room center
        positions = []
        for obj in room_objects:
            if hasattr(obj, 'position'):
                positions.append(obj.position)
        
        if not positions:
            return False
        
        center = np.mean(positions, axis=0)
        occ_map = self.mapper.occupancy_map
        map_pos = occ_map._world_to_map(center[0], center[2])
        
        self.goal = NavGoal(
            goal_type=GoalType.ROOM,
            target_name=room_type,
            target_position=center,
            target_map=map_pos
        )
        
        print(f"Goal set: {room_type} at ({center[0]:.1f}, {center[2]:.1f})")
        return self._plan_path()
    
    def set_goal_position(self, position: np.ndarray) -> bool:
        """Set goal to navigate to specific world position."""
        occ_map = self.mapper.occupancy_map
        map_pos = occ_map._world_to_map(position[0], position[2])
        
        self.goal = NavGoal(
            goal_type=GoalType.POSITION,
            target_position=position,
            target_map=map_pos
        )
        
        print(f"Goal set: position ({position[0]:.1f}, {position[2]:.1f})")
        return self._plan_path()
    
    def set_goal_frontier(self) -> bool:
        """Set goal to explore nearest frontier."""
        frontier = self._find_nearest_frontier()
        
        if frontier is None:
            print("No unexplored areas")
            self.state = NavState.FAILED
            return False
        
        self.goal = NavGoal(
            goal_type=GoalType.FRONTIER,
            target_map=frontier
        )
        
        print(f"Goal set: frontier at map ({frontier[0]}, {frontier[1]})")
        return self._plan_path()
    
    def cancel_goal(self):
        """Cancel current navigation goal."""
        self.state = NavState.IDLE
        self.goal = None
        self.path = None
        self.current_waypoint_idx = 0
        self.recovery_actions.clear()
    
    # =========================================================================
    # Path Planning (Global)
    # =========================================================================
    
    def _plan_path(self) -> bool:
        """Plan global path to current goal."""
        if self.goal is None:
            return False
        
        self.state = NavState.PLANNING
        occ_map = self.mapper.occupancy_map
        
        # Get start position
        if not occ_map.trajectory:
            print("No agent position")
            self.state = NavState.FAILED
            return False
        
        start_world = occ_map.trajectory[-1]
        start_map = occ_map._world_to_map(start_world[0], start_world[2])
        goal_map = self.goal.target_map
        
        # Get obstacle map and inflate
        obstacle = occ_map.get_obstacle_map()
        kernel = np.ones((self.path_inflation * 2 + 1,) * 2, np.uint8)
        inflated = cv2.dilate((obstacle > 0).astype(np.uint8), kernel)
        
        # A* search
        path_map = self._astar(start_map, goal_map, inflated)
        
        if not path_map:
            # Try with less inflation
            kernel_small = np.ones((3, 3), np.uint8)
            inflated_small = cv2.dilate((obstacle > 0).astype(np.uint8), kernel_small)
            path_map = self._astar(start_map, goal_map, inflated_small)
        
        if not path_map:
            print("No path found")
            self.state = NavState.FAILED
            return False
        
        # Simplify path
        simplified = path_map[::self.path_simplify]
        if path_map[-1] not in simplified:
            simplified.append(path_map[-1])
        
        # Convert to world coordinates
        path_world = []
        for mx, my in simplified:
            wx = (mx - occ_map.map_size // 2) * occ_map.resolution / 100.0 + occ_map.origin_x
            wz = (my - occ_map.map_size // 2) * occ_map.resolution / 100.0 + occ_map.origin_z
            path_world.append(np.array([wx, start_world[1], wz]))
        
        # Calculate distance
        total_dist = 0
        for i in range(1, len(simplified)):
            dx = simplified[i][0] - simplified[i-1][0]
            dy = simplified[i][1] - simplified[i-1][1]
            total_dist += np.sqrt(dx*dx + dy*dy) * occ_map.resolution / 100.0
        
        self.path = NavPath(
            waypoints_map=simplified,
            waypoints_world=path_world,
            goal=self.goal,
            distance=total_dist
        )
        
        self.current_waypoint_idx = 0
        self.state = NavState.NAVIGATING
        self.replans += 1
        
        print(f"Path planned: {len(simplified)} waypoints, {total_dist:.1f}m")
        return True
    
    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacle_map: np.ndarray
    ) -> List[Tuple[int, int]]:
        """A* pathfinding."""
        h, w = obstacle_map.shape
        
        # Bounds check
        if not (0 <= start[0] < w and 0 <= start[1] < h):
            return []
        if not (0 <= goal[0] < w and 0 <= goal[1] < h):
            return []
        
        def heuristic(a, b):
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
        neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        
        open_set = [(heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}
        counter = 0
        
        while open_set and counter < 50000:
            counter += 1
            _, _, current = heapq.heappop(open_set)
            
            if heuristic(current, goal) < 5:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if obstacle_map[ny, nx] > 0:
                    continue
                
                cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                tentative_g = g_score[current] + cost
                
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + heuristic((nx, ny), goal), counter, (nx, ny)))
        
        return []
    
    def _find_nearest_frontier(self) -> Optional[Tuple[int, int]]:
        """Find nearest unexplored area."""
        occ_map = self.mapper.occupancy_map
        explored = occ_map.get_explored_map()
        obstacle = occ_map.get_obstacle_map()
        
        # Frontier = boundary of explored, not obstacle
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate((explored > 0).astype(np.uint8), kernel)
        frontier = ((dilated > 0) & (explored == 0) & (obstacle == 0)).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier)
        
        if num_labels <= 1:
            return None
        
        # Find nearest large frontier
        if occ_map.trajectory:
            agent = occ_map.trajectory[-1]
            agent_map = occ_map._world_to_map(agent[0], agent[2])
        else:
            agent_map = (occ_map.map_size // 2, occ_map.map_size // 2)
        
        best = None
        best_score = float('inf')
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:
                continue
            
            cx, cy = int(centroids[i, 0]), int(centroids[i, 1])
            dist = np.sqrt((cx - agent_map[0])**2 + (cy - agent_map[1])**2)
            
            # Score: prefer close and large
            score = dist / (area + 1)
            if score < best_score:
                best_score = score
                best = (cx, cy)
        
        return best
    
    # =========================================================================
    # Action Generation (Local)
    # =========================================================================
    
    def get_action(
        self,
        depth: np.ndarray,
        agent_position: np.ndarray,
        agent_rotation: np.ndarray
    ) -> Optional[str]:
        """
        Get next navigation action.
        
        Args:
            depth: Current depth image
            agent_position: World position [x, y, z]
            agent_rotation: Quaternion [x, y, z, w]
            
        Returns:
            'move_forward', 'turn_left', 'turn_right', or None if done/failed
        """
        self.total_actions += 1
        
        # Update position history
        self.position_history.append(agent_position[[0, 2]].copy())
        
        # Handle recovery
        if self.recovery_actions:
            return self.recovery_actions.pop(0)
        
        # Check stuck
        if self._is_stuck():
            self._start_recovery()
            if self.recovery_actions:
                return self.recovery_actions.pop(0)
        
        # No goal
        if self.state in [NavState.IDLE, NavState.REACHED, NavState.FAILED]:
            return None
        
        if self.path is None or not self.path.valid:
            return None
        
        # Check if reached goal
        if self.goal.target_position is not None:
            dist_to_goal = np.linalg.norm(
                agent_position[[0, 2]] - self.goal.target_position[[0, 2]]
            )
            if dist_to_goal < self.goal.tolerance:
                self.state = NavState.REACHED
                print(f"Goal reached! Distance: {dist_to_goal:.2f}m")
                return None
        
        # Check waypoint progress
        if self.current_waypoint_idx >= len(self.path.waypoints_world):
            self.state = NavState.REACHED
            return None
        
        target = self.path.waypoints_world[self.current_waypoint_idx]
        dist_to_wp = np.linalg.norm(agent_position[[0, 2]] - target[[0, 2]])
        
        if dist_to_wp < self.waypoint_tolerance:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.path.waypoints_world):
                self.state = NavState.REACHED
                return None
            target = self.path.waypoints_world[self.current_waypoint_idx]
        
        # Compute action
        return self._compute_action(agent_position, agent_rotation, target, depth)
    
    def _compute_action(
        self,
        agent_pos: np.ndarray,
        agent_rot: np.ndarray,
        target: np.ndarray,
        depth: np.ndarray
    ) -> str:
        """Compute action with collision avoidance."""
        yaw = self._get_yaw(agent_rot)
        
        # Direction to target
        dx = target[0] - agent_pos[0]
        dz = target[2] - agent_pos[2]
        target_angle = np.arctan2(dx, -dz)  # Habitat: -Z is forward at yaw=0
        
        # Angle difference
        angle_diff = target_angle - yaw
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Turn if not facing target
        if abs(angle_diff) > self.turn_threshold_rad:
            self.state = NavState.NAVIGATING
            return 'turn_left' if angle_diff > 0 else 'turn_right'
        
        # Check for obstacles ahead
        h, w = depth.shape
        center = depth[h//3:2*h//3, w//3:2*w//3]
        valid = center[(center > 0.1) & (center < 10)]
        
        if len(valid) > 0:
            min_depth = np.percentile(valid, 10)
        else:
            min_depth = 10.0
        
        if min_depth < self.obstacle_threshold:
            self.collisions_avoided += 1
            self.state = NavState.AVOIDING
            
            # Check left vs right for clearance
            left = depth[h//3:2*h//3, :w//4]
            right = depth[h//3:2*h//3, 3*w//4:]
            
            left_clear = np.mean(left[left > 0.1]) if (left > 0.1).any() else 0
            right_clear = np.mean(right[right > 0.1]) if (right > 0.1).any() else 0
            
            return 'turn_left' if left_clear > right_clear else 'turn_right'
        
        self.state = NavState.NAVIGATING
        return 'move_forward'
    
    def _get_yaw(self, quat: np.ndarray) -> float:
        """Extract yaw from quaternion."""
        x, y, z, w = quat
        return np.arctan2(2 * (w * y + z * x), 1 - 2 * (x * x + y * y))
    
    # =========================================================================
    # Stuck Detection & Recovery
    # =========================================================================
    
    def _is_stuck(self) -> bool:
        """Check if agent is stuck."""
        if len(self.position_history) < self.stuck_threshold:
            return False
        
        positions = list(self.position_history)
        total = sum(
            np.linalg.norm(positions[i] - positions[i-1])
            for i in range(1, len(positions))
        )
        
        if total < self.stuck_movement:
            self.stuck_count += 1
            return self.stuck_count > 2
        
        self.stuck_count = 0
        return False
    
    def _start_recovery(self):
        """Initiate recovery behavior."""
        print("Stuck! Recovering...")
        self.state = NavState.RECOVERING
        
        self.recovery_actions = [
            'turn_left', 'turn_left', 'turn_left',
            'turn_left', 'turn_left', 'turn_left',
            'move_forward', 'move_forward',
            'turn_right', 'turn_right', 'turn_right',
            'move_forward',
        ]
        
        self.stuck_count = 0
        
        # Replan after recovery
        if self.goal:
            # Will replan on next get_action after recovery completes
            pass
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def visualize_on_map(self, map_vis: np.ndarray) -> np.ndarray:
        """Draw navigation path and goal on map."""
        vis = map_vis.copy()
        
        if self.path is None or not self.path.valid:
            return vis
        
        # Draw path
        waypoints = self.path.waypoints_map
        for i in range(len(waypoints) - 1):
            p1, p2 = waypoints[i], waypoints[i + 1]
            color = (255, 200, 0) if i >= self.current_waypoint_idx else (100, 100, 100)
            cv2.line(vis, p1, p2, color, 2)
        
        # Draw waypoints
        for i, wp in enumerate(waypoints):
            if i < self.current_waypoint_idx:
                color = (100, 100, 100)
            elif i == self.current_waypoint_idx:
                color = (0, 255, 255)
            else:
                color = (255, 200, 0)
            cv2.circle(vis, wp, 3, color, -1)
        
        # Draw goal
        if self.goal and self.goal.target_map:
            cv2.circle(vis, self.goal.target_map, 10, (255, 0, 255), 2)
            
            label = self.goal.target_name or self.goal.goal_type.value
            cv2.putText(vis, label,
                       (self.goal.target_map[0] + 12, self.goal.target_map[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw state
        state_colors = {
            NavState.IDLE: (150, 150, 150),
            NavState.NAVIGATING: (0, 255, 0),
            NavState.AVOIDING: (0, 255, 255),
            NavState.RECOVERING: (255, 0, 0),
            NavState.REACHED: (0, 255, 0),
            NavState.FAILED: (0, 0, 255),
        }
        cv2.putText(vis, f"Nav: {self.state.value}", (10, vis.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_colors.get(self.state, (255,255,255)), 1)
        
        return vis
    
    def get_status(self) -> Dict:
        """Get navigation status."""
        return {
            'state': self.state.value,
            'goal_type': self.goal.goal_type.value if self.goal else None,
            'goal_target': self.goal.target_name if self.goal else None,
            'waypoint': f"{self.current_waypoint_idx}/{len(self.path.waypoints_map) if self.path else 0}",
            'total_actions': self.total_actions,
            'replans': self.replans,
            'collisions_avoided': self.collisions_avoided,
        }
