#!/usr/bin/env python3
"""
Pre-built Map Navigator V2 - Fixed Coordinates
===============================================

Fixes:
1. Proper coordinate system alignment with saved maps
2. Origin loaded from statistics.json
3. Floor transition support
4. Correct world<->map conversion matching mapper

Usage:
    python navigate_v2.py --scene /path/to/scene.glb --map-dir ./output_enhanced
"""

import argparse
import numpy as np
import cv2
import torch
import json
import heapq
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
from collections import deque
import time


class NavState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    REACHED = "reached"
    FAILED = "failed"


@dataclass
class MapData:
    """Loaded map data for one floor."""
    floor_id: int
    navigable_map: np.ndarray
    explored_map: np.ndarray
    objects: Dict
    map_size: int
    resolution: float
    origin_x: float
    origin_z: float


class Navigator:
    """
    Navigator using pre-built maps with correct coordinate handling.
    
    Supports both ground truth pose (Habitat) and visual odometry.
    """
    
    def __init__(self, map_dir: str, use_visual_odometry: bool = False):
        self.map_dir = Path(map_dir)
        self.use_vo = use_visual_odometry
        
        # Visual odometry (optional)
        if use_visual_odometry:
            from slam.lightweight_vo import PoseTracker
            self.pose_tracker = PoseTracker(use_vo=True)
        else:
            self.pose_tracker = None
        
        # Will be loaded from saved data
        self.floors: Dict[int, MapData] = {}
        self.all_objects: Dict = {}
        self.stairs: List[Dict] = []
        
        # Coordinate system params (loaded from map)
        self.resolution = 5.0  # cm per pixel
        self.map_size = 480
        self.origin_x = 0.0
        self.origin_z = 0.0
        
        # Load maps
        self._load_maps()
        
        # Room annotations (from map_editor_v2)
        self.room_annotations: Dict = {}
        self._load_room_annotations()
        
        # Instruction parser (regex mode — lightweight, works on Go2)
        try:
            from perception.instruction_parser import InstructionParser
            self.parser = InstructionParser(mode='regex')
        except ImportError:
            self.parser = None
        
        # Navigation state
        self.current_floor = list(self.floors.keys())[0] if self.floors else 0
        self.state = NavState.IDLE
        self.goal_map: Optional[Tuple[int, int]] = None
        self.goal_name: str = ""
        self.path: List[Tuple[int, int]] = []
        self.current_waypoint = 0
        
        # Planning params
        self.inflation_radius = 5
        self.waypoint_tolerance = 8  # pixels
        self.goal_tolerance = 12  # pixels
        
        # Local avoidance
        self.obstacle_threshold = 0.4  # meters
        self.turn_threshold = np.radians(15)
        
        # Stuck detection
        self.position_history = deque(maxlen=20)
        self.recovery_actions: List[str] = []
        
        # Stats
        self.actions = 0
        self.avoided = 0
    
    def _load_maps(self):
        """Load maps and coordinate system info."""
        print(f"Loading maps from {self.map_dir}")
        
        # Load statistics to get coordinate system
        stats_path = self.map_dir / 'statistics.json'
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.origin_x = stats.get('origin_x', 0.0)
            self.origin_z = stats.get('origin_z', 0.0)
            self.resolution = stats.get('resolution_cm', 5.0)
            self.map_size = stats.get('map_size', 480)
            print(f"  Origin: ({self.origin_x:.3f}, {self.origin_z:.3f})")
            print(f"  Resolution: {self.resolution}cm, Size: {self.map_size}")
        
        # Load floor info
        floor_info_path = self.map_dir / 'floor_info.json'
        if floor_info_path.exists():
            with open(floor_info_path) as f:
                floor_info = json.load(f)
            floor_ids = floor_info.get('floor_ids', [0])
        else:
            floor_ids = [0]
        
        # Load each floor
        for fid in floor_ids:
            floor_dir = self.map_dir / f'floor_{fid}'
            if not floor_dir.exists():
                continue
            
            # Load navigable map (prefer this over occupancy)
            nav_path = floor_dir / 'navigable_map.pt'
            if not nav_path.exists():
                nav_path = floor_dir / 'occupancy_map.pt'
            if not nav_path.exists():
                continue
            
            nav_map = torch.load(nav_path).numpy()
            
            # Load explored map
            exp_path = floor_dir / 'explored_map.pt'
            exp_map = torch.load(exp_path).numpy() if exp_path.exists() else np.ones_like(nav_map)
            
            # Load objects
            obj_path = floor_dir / 'objects.json'
            objects = {}
            if obj_path.exists():
                with open(obj_path) as f:
                    objects = json.load(f)
            
            self.floors[fid] = MapData(
                floor_id=fid,
                navigable_map=nav_map,
                explored_map=exp_map,
                objects=objects,
                map_size=nav_map.shape[0],
                resolution=self.resolution,
                origin_x=self.origin_x,
                origin_z=self.origin_z
            )
            print(f"  Floor {fid}: {nav_map.shape}, {len(objects)} objects")
        
        # Load global objects
        obj_path = self.map_dir / 'objects.json'
        if obj_path.exists():
            with open(obj_path) as f:
                self.all_objects = json.load(f)
        
        # Load stairs
        stairs_path = self.map_dir / 'stairs.json'
        if stairs_path.exists():
            with open(stairs_path) as f:
                self.stairs = json.load(f)
        
        # Floor detection
        self.floor_heights: Dict[int, float] = {}  # floor_id -> base height
        floor_info_path = self.map_dir / 'floor_info.json'
        if floor_info_path.exists():
            with open(floor_info_path) as f:
                fi = json.load(f)
                base_heights = fi.get('base_heights', {})
                for k, v in base_heights.items():
                    self.floor_heights[int(k)] = v
        
        print(f"Loaded {len(self.floors)} floors, {len(self.all_objects)} objects")
        if self.floor_heights:
            print(f"Floor heights: {self.floor_heights}")
    
    def _load_room_annotations(self):
        """Load room annotations from map_annotations.json per floor.
        Handles coordinate rescaling if annotation was made on a different-sized image."""
        self.floor_room_annotations: Dict[int, Dict] = {}
        
        for fid in self.floors:
            floor_dir = self.map_dir / f'floor_{fid}'
            ann_path = floor_dir / 'map_annotations.json'
            
            if not ann_path.exists():
                continue
            
            with open(ann_path) as f:
                data = json.load(f)
            
            rooms = data.get('rooms', {})
            
            # Check if annotation coordinates need rescaling
            ann_map_size = data.get('map_size', [0, 0])
            actual_map_size = self.floors[fid].map_size if fid in self.floors else self.map_size
            
            scale_x, scale_y = 1.0, 1.0
            if ann_map_size[0] > 0 and ann_map_size[0] != actual_map_size:
                scale_x = actual_map_size / ann_map_size[0]
                scale_y = actual_map_size / (ann_map_size[1] if len(ann_map_size) > 1 else ann_map_size[0])
                print(f"  Rescaling floor {fid} annotations: {ann_map_size} → {actual_map_size} (scale={scale_x:.2f})")
            
            def rescale_point(p):
                return [int(p[0] * scale_x), int(p[1] * scale_y)]
            
            for label, rdata in rooms.items():
                if scale_x != 1.0 or scale_y != 1.0:
                    if rdata.get('center'):
                        rdata['center'] = rescale_point(rdata['center'])
                    if rdata.get('polygon'):
                        rdata['polygon'] = [rescale_point(p) for p in rdata['polygon']]
                rdata['floor'] = fid
                self.room_annotations[label] = rdata
            
            self.floor_room_annotations[fid] = rooms
            
            # Merge annotated objects
            for i, obj in enumerate(data.get('objects', [])):
                obj_id = f"annotated_f{fid}_{i}"
                if obj_id not in self.all_objects:
                    pos = obj['position']
                    if scale_x != 1.0 or scale_y != 1.0:
                        pos = rescale_point(pos)
                    mx, my = pos[0], pos[1]
                    wx, wz = self.map_to_world(mx, my)
                    self.all_objects[obj_id] = {
                        'class': obj['label'],
                        'position': [wx, 0.0, wz],
                        'map_pos': [mx, my],
                        'floor': fid,
                        'source': 'annotation',
                    }
            
            print(f"  Floor {fid} annotations: {len(rooms)} rooms, "
                  f"{len(data.get('objects', []))} objects from {ann_path}")
        
        # Fallback: root directory
        root_ann = self.map_dir / 'map_annotations.json'
        if root_ann.exists() and not self.room_annotations:
            with open(root_ann) as f:
                data = json.load(f)
            self.room_annotations = data.get('rooms', {})
            print(f"  Root annotations: {len(self.room_annotations)} rooms from {root_ann}")
        
        if not self.room_annotations:
            print("  No room annotations found (run map_editor_v2.py to create)")
    
    def set_goal_from_instruction(self, instruction: str, agent_pos: np.ndarray) -> bool:
        """
        Parse natural language instruction and set navigation goal.
        
        Supports:
          "go to the kitchen" → navigate to room center
          "find the toilet"   → navigate to nearest detected object
          "go to room 3"      → navigate to numbered room
          "stop"              → cancel navigation
        
        Returns True if goal was set successfully.
        """
        if self.parser is None:
            print("Instruction parser not available")
            return False
        
        goal = self.parser.parse(instruction)
        print(f"  Parsed: type={goal.type}, target={goal.target}")
        
        if goal.type == 'stop':
            self.cancel()
            return False
        
        if goal.type == 'room':
            return self._set_room_goal(goal.target, agent_pos)
        
        if goal.type in ('object', 'landmark'):
            return self.set_goal_object(goal.target, agent_pos)
        
        print(f"  Unknown instruction: {instruction}")
        return False
    
    def _set_room_goal(self, room_target: str, agent_pos: np.ndarray) -> bool:
        """Navigate to a room center from annotations."""
        if not self.room_annotations:
            print(f"  No room annotations loaded")
            return False
        
        # Match by type (kitchen → kitchen_1), exact label, or prefix
        matched_label = None
        matched_data = None
        
        for label, rdata in self.room_annotations.items():
            if rdata.get('type') == room_target:
                matched_label = label
                matched_data = rdata
                break
            if label == room_target:
                matched_label = label
                matched_data = rdata
                break
            if label.startswith(room_target):
                matched_label = label
                matched_data = rdata
                break
        
        if matched_data is None:
            print(f"  Room '{room_target}' not found in annotations")
            available = [f"{l} ({r.get('type','')})" for l, r in self.room_annotations.items()]
            print(f"  Available: {', '.join(available)}")
            return False
        
        center = matched_data.get('center')
        if center is None:
            print(f"  Room {matched_label} has no center point")
            return False
        
        # Center is in map coordinates — set goal directly
        mx, my = int(center[0]), int(center[1])
        self.goal_map = (mx, my)
        self.goal_name = f"room:{matched_label}"
        
        wx, wz = self.map_to_world(mx, my)
        print(f"  Goal: {matched_label} center at map({mx},{my}) world({wx:.2f},{wz:.2f})")
        
        return self._plan_path(agent_pos)
    
    def world_to_map(self, wx: float, wz: float) -> Tuple[int, int]:
        """Convert world coordinates to map pixels - MUST match mapper!"""
        mx = int((wx - self.origin_x) * 100 / self.resolution + self.map_size // 2)
        my = int((wz - self.origin_z) * 100 / self.resolution + self.map_size // 2)
        return (mx, my)
    
    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """Convert map pixels to world coordinates."""
        wx = (mx - self.map_size // 2) * self.resolution / 100.0 + self.origin_x
        wz = (my - self.map_size // 2) * self.resolution / 100.0 + self.origin_z
        return (wx, wz)
    
    def get_available_objects(self) -> List[str]:
        """Get list of object classes we can navigate to."""
        return list(set(obj['class'] for obj in self.all_objects.values()))
    
    # =========================================================================
    # Goal Setting
    # =========================================================================
    
    def set_goal_object(self, object_class: str, agent_pos: np.ndarray) -> bool:
        """Navigate to nearest object of given class (with fuzzy matching)."""
        object_class = object_class.lower().strip()
        
        # Aliases for common names
        ALIASES = {
            'tv': ['tv', 'tv_monitor', 'television', 'monitor'],
            'tv_monitor': ['tv', 'tv_monitor', 'television', 'monitor'],
            'sofa': ['sofa', 'couch'],
            'couch': ['sofa', 'couch'],
            'table': ['table', 'dining table', 'coffee table', 'desk'],
            'dining table': ['dining table', 'table'],
            'plant': ['plant', 'potted plant'],
            'potted plant': ['plant', 'potted plant'],
            'fridge': ['refrigerator', 'fridge'],
            'refrigerator': ['refrigerator', 'fridge'],
        }
        
        match_names = ALIASES.get(object_class, [object_class])
        
        # Find ALL objects matching any alias
        candidates = []
        for obj_id, obj in self.all_objects.items():
            obj_class = obj['class'].lower()
            matched = (obj_class in match_names or 
                       object_class in obj_class or 
                       obj_class in object_class)
            if matched:
                pos = obj['position']
                dist = np.sqrt((pos[0] - agent_pos[0])**2 + (pos[2] - agent_pos[2])**2)
                candidates.append({
                    'id': obj_id,
                    'obj': obj,
                    'distance': dist
                })
        
        if not candidates:
            # Show available objects to help user
            available = sorted(set(obj['class'].lower() for obj in self.all_objects.values()))
            print(f"Object '{object_class}' not found")
            print(f"  Available: {', '.join(available)}")
            return False
        
        candidates.sort(key=lambda x: x['distance'])
        
        if len(candidates) > 1:
            print(f"Found {len(candidates)} '{object_class}' objects:")
            for i, c in enumerate(candidates[:5]):
                print(f"  {i+1}. {c['obj']['class']} dist={c['distance']:.2f}m")
        
        target = candidates[0]['obj']
        pos = target['position']
        self.goal_map = self.world_to_map(pos[0], pos[2])
        self.goal_name = f"{target['class']} (nearest of {len(candidates)})"
        
        print(f"Goal: {target['class']} at world({pos[0]:.2f}, {pos[2]:.2f}) -> map{self.goal_map}, dist={candidates[0]['distance']:.2f}m")
        
        return self._plan_path(agent_pos)
    
    def set_goal_position(self, mx: int, my: int, agent_pos: np.ndarray) -> bool:
        """Navigate to map position."""
        self.goal_map = (mx, my)
        self.goal_name = f"pos({mx},{my})"
        
        wx, wz = self.map_to_world(mx, my)
        print(f"Goal: map({mx}, {my}) -> world({wx:.2f}, {wz:.2f})")
        
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
            print("No map for current floor")
            return False
        
        map_data = self.floors[self.current_floor]
        nav_map = map_data.navigable_map
        h, w = nav_map.shape
        
        # Get start position
        start = self.world_to_map(agent_pos[0], agent_pos[2])
        goal = self.goal_map
        
        print(f"Planning: start={start}, goal={goal}, map_size={w}x{h}")
        
        # Validate positions
        if not (0 <= start[0] < w and 0 <= start[1] < h):
            print(f"ERROR: Start {start} out of bounds!")
            self.state = NavState.FAILED
            return False
        
        if not (0 <= goal[0] < w and 0 <= goal[1] < h):
            print(f"ERROR: Goal {goal} out of bounds!")
            self.state = NavState.FAILED
            return False
        
        # Inflate obstacles
        kernel = np.ones((self.inflation_radius * 2 + 1,) * 2, np.uint8)
        inflated = cv2.dilate((nav_map > 0.5).astype(np.uint8), kernel)
        
        # If start is blocked, find nearest free cell
        if inflated[start[1], start[0]] > 0:
            start = self._find_free_cell(start, inflated)
            if start is None:
                print("Cannot find free start position")
                self.state = NavState.FAILED
                return False
            print(f"Adjusted start to {start}")
        
        # If goal is blocked, find nearest free cell
        if inflated[goal[1], goal[0]] > 0:
            goal = self._find_free_cell(goal, inflated)
            if goal is None:
                print("Cannot find free goal position")
                self.state = NavState.FAILED
                return False
            print(f"Adjusted goal to {goal}")
        
        # A* search
        path = self._astar(start, goal, inflated)
        
        if not path:
            # Try with minimal inflation
            inflated_min = cv2.dilate((nav_map > 0.5).astype(np.uint8), np.ones((3,3), np.uint8))
            path = self._astar(start, goal, inflated_min)
        
        if not path:
            print("No path found")
            self.state = NavState.FAILED
            return False
        
        # Simplify path
        self.path = path[::3]
        if path[-1] not in self.path:
            self.path.append(path[-1])
        
        self.current_waypoint = 0
        self.state = NavState.NAVIGATING
        
        print(f"Path found: {len(self.path)} waypoints")
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
                    if 0 <= nx < w and 0 <= ny < h:
                        if obs_map[ny, nx] == 0:
                            return (nx, ny)
        return None
    
    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int], obs: np.ndarray) -> List[Tuple[int, int]]:
        """A* pathfinding."""
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
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if obs[ny, nx] > 0:
                    continue
                
                cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                tg = g_score[cur] + cost
                
                if (nx, ny) not in g_score or tg < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = cur
                    g_score[(nx, ny)] = tg
                    counter += 1
                    heapq.heappush(open_set, (tg + heuristic((nx, ny), goal), counter, (nx, ny)))
        
        return []
    
    # =========================================================================
    # Action Generation
    # =========================================================================
    
    def get_action(self, agent_pos: np.ndarray, agent_rot: np.ndarray, depth: np.ndarray) -> Optional[str]:
        """Get next action."""
        self.actions += 1
        self.position_history.append(agent_pos[[0, 2]].copy())
        
        # Check for floor change based on agent height
        self._check_floor_change(agent_pos[1])
        
        # Recovery actions first
        if self.recovery_actions:
            return self.recovery_actions.pop(0)
        
        # Check stuck
        if self._is_stuck():
            self._start_recovery()
            if self.recovery_actions:
                return self.recovery_actions.pop(0)
        
        # No navigation
        if self.state != NavState.NAVIGATING or not self.path:
            return None
        
        # Current map position
        agent_map = self.world_to_map(agent_pos[0], agent_pos[2])
        
        # Check if goal reached
        if self.goal_map:
            dist_to_goal = np.sqrt((agent_map[0] - self.goal_map[0])**2 + 
                                   (agent_map[1] - self.goal_map[1])**2)
            if dist_to_goal < self.goal_tolerance:
                self.state = NavState.REACHED
                print(f"GOAL REACHED: {self.goal_name}")
                return None
        
        # Check waypoint progress
        if self.current_waypoint >= len(self.path):
            self.state = NavState.REACHED
            return None
        
        target_map = self.path[self.current_waypoint]
        dist_to_wp = np.sqrt((agent_map[0] - target_map[0])**2 + 
                            (agent_map[1] - target_map[1])**2)
        
        if dist_to_wp < self.waypoint_tolerance:
            self.current_waypoint += 1
            if self.current_waypoint >= len(self.path):
                self.state = NavState.REACHED
                return None
            target_map = self.path[self.current_waypoint]
        
        # Compute action
        return self._compute_action(agent_pos, agent_rot, target_map, depth)
    
    def _compute_action(self, agent_pos: np.ndarray, agent_rot: np.ndarray, 
                       target_map: Tuple[int, int], depth: np.ndarray) -> str:
        """Compute action with collision avoidance."""
        # Get yaw from quaternion
        x, y, z, w = agent_rot
        # Standard quaternion to euler (yaw around Y axis)
        siny_cosp = 2.0 * (w * y + z * x)
        cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Target in world coordinates
        tx, tz = self.map_to_world(target_map[0], target_map[1])
        
        # Vector from agent to target
        dx = tx - agent_pos[0]
        dz = tz - agent_pos[2]
        dist_to_target = np.sqrt(dx*dx + dz*dz)
        
        # Habitat coordinate system:
        # - At yaw=0, agent faces -Z direction
        # - Positive yaw = counter-clockwise rotation (when viewed from above)
        # - +X is to the right of forward direction
        #
        # To get target_angle (angle to target in same frame as yaw):
        # - If target is in -Z direction (forward): angle = 0
        # - If target is in +X direction (right): angle = -π/2 (or +3π/2)
        # - If target is in +Z direction (behind): angle = π (or -π)
        # - If target is in -X direction (left): angle = +π/2
        #
        # Using atan2(dx, -dz):
        # - Forward (dx=0, dz<0): atan2(0, +) = 0 ✓
        # - Right (dx>0, dz=0): atan2(+, 0) = +π/2 ... but should be -π/2
        # - Behind (dx=0, dz>0): atan2(0, -) = π ✓
        # - Left (dx<0, dz=0): atan2(-, 0) = -π/2 ... but should be +π/2
        #
        # The signs are flipped! Use atan2(-dx, -dz):
        # - Forward (dx=0, dz<0): atan2(0, +) = 0 ✓
        # - Right (dx>0, dz=0): atan2(-, 0) = -π/2 ✓
        # - Behind (dx=0, dz>0): atan2(0, -) = π ✓
        # - Left (dx<0, dz=0): atan2(+, 0) = +π/2 ✓
        
        target_angle = np.arctan2(-dx, -dz)
        
        # Angle difference (how much we need to turn)
        angle_diff = target_angle - yaw
        
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Debug output every 30 frames
        if self.actions % 30 == 0:
            agent_map = self.world_to_map(agent_pos[0], agent_pos[2])
            print(f"  Agent: map{agent_map} world({agent_pos[0]:.2f},{agent_pos[2]:.2f}) yaw={np.degrees(yaw):.1f}°")
            print(f"  Target: map{target_map} world({tx:.2f},{tz:.2f})")
            print(f"  Delta: dx={dx:.2f} dz={dz:.2f} dist={dist_to_target:.2f}m")
            print(f"  Angles: target={np.degrees(target_angle):.1f}° yaw={np.degrees(yaw):.1f}° diff={np.degrees(angle_diff):.1f}°")
            action_preview = 'TURN LEFT' if angle_diff > self.turn_threshold else 'TURN RIGHT' if angle_diff < -self.turn_threshold else 'FORWARD'
            print(f"  -> {action_preview}")
        
        # FIRST: Check real-time depth for immediate obstacles
        h, w = depth.shape
        
        # Center region for forward obstacles
        center_region = depth[h//3:2*h//3, w//3:2*w//3]
        valid_center = center_region[(center_region > 0.1) & (center_region < 10)]
        min_front_depth = np.percentile(valid_center, 5) if len(valid_center) > 100 else 10.0
        
        # If obstacle very close, must avoid regardless of target direction
        if min_front_depth < 0.35:
            self.avoided += 1
            self.state = NavState.AVOIDING
            
            # Check left and right for escape
            left_region = depth[h//4:3*h//4, :w//4]
            right_region = depth[h//4:3*h//4, 3*w//4:]
            
            left_clear = np.median(left_region[left_region > 0.1]) if (left_region > 0.1).any() else 0
            right_clear = np.median(right_region[right_region > 0.1]) if (right_region > 0.1).any() else 0
            
            if self.actions % 10 == 0:
                print(f"  OBSTACLE! front={min_front_depth:.2f}m left={left_clear:.2f}m right={right_clear:.2f}m")
            
            return 'turn_left' if left_clear > right_clear else 'turn_right'
        
        # SECOND: Turn toward target if not facing it
        if abs(angle_diff) > self.turn_threshold:
            self.state = NavState.NAVIGATING
            # Positive angle_diff = target is to our left = turn left
            # Negative angle_diff = target is to our right = turn right
            return 'turn_left' if angle_diff > 0 else 'turn_right'
        
        # THIRD: Check if forward path is clear enough to move
        if min_front_depth < self.obstacle_threshold:
            self.avoided += 1
            self.state = NavState.AVOIDING
            
            # Obstacle in path but not critical - try to go around
            left_region = depth[h//3:2*h//3, :w//3]
            right_region = depth[h//3:2*h//3, 2*w//3:]
            
            left_clear = np.mean(left_region[left_region > 0.1]) if (left_region > 0.1).any() else 0
            right_clear = np.mean(right_region[right_region > 0.1]) if (right_region > 0.1).any() else 0
            
            return 'turn_left' if left_clear > right_clear else 'turn_right'
        
        # Path is clear, move forward
        self.state = NavState.NAVIGATING
        return 'move_forward'
    
    def _is_stuck(self) -> bool:
        """Check if stuck."""
        if len(self.position_history) < 15:
            return False
        
        positions = list(self.position_history)
        total = sum(np.linalg.norm(positions[i] - positions[i-1]) for i in range(1, len(positions)))
        return total < 0.15
    
    def _check_floor_change(self, agent_height: float):
        """Check if agent moved to different floor based on height.
        
        Uses simple threshold: if agent height is clearly closer to 
        another floor's expected height, switch immediately.
        """
        if not self.floor_heights or len(self.floor_heights) < 2:
            return
        
        # Find best matching floor
        best_floor = self.current_floor
        best_diff = float('inf')
        
        for floor_id, base_height in self.floor_heights.items():
            expected = base_height + 1.25  # Camera height above floor
            diff = abs(agent_height - expected)
            if diff < best_diff:
                best_diff = diff
                best_floor = floor_id
        
        if best_floor == self.current_floor:
            return
        
        # How far is agent from current floor's expected height?
        current_expected = self.floor_heights.get(self.current_floor, 0) + 1.25
        current_diff = abs(agent_height - current_expected)
        
        # Switch if: clearly closer to new floor than current floor
        # (new floor diff < 1.0m AND current floor diff > 1.5m)
        if best_diff < 1.0 and current_diff > 1.5:
            print(f"FLOOR CHANGE: {self.current_floor} -> {best_floor}")
            print(f"  Agent height: {agent_height:.2f}m")
            print(f"  Current floor {self.current_floor}: expected {current_expected:.2f}m (off by {current_diff:.2f}m)")
            print(f"  New floor {best_floor}: expected {self.floor_heights[best_floor] + 1.25:.2f}m (off by {best_diff:.2f}m)")
            
            self.current_floor = best_floor
            
            if self.path:
                print("  Path cancelled - replanning on new floor")
                self.path = []
                self.state = NavState.IDLE
    
    def _start_recovery(self):
        """Start recovery."""
        print("STUCK - recovering")
        self.recovery_actions = ['turn_left'] * 6 + ['move_forward'] * 2 + ['turn_right'] * 3
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def visualize(self, agent_pos: np.ndarray = None) -> np.ndarray:
        """Visualize map with path and room annotations."""
        if self.current_floor not in self.floors:
            return np.zeros((500, 500, 3), dtype=np.uint8)
        
        map_data = self.floors[self.current_floor]
        nav_map = map_data.navigable_map
        exp_map = map_data.explored_map
        h, w = nav_map.shape
        
        # Try to load semantic map PNG as background (more informative)
        floor_dir = self.map_dir / f'floor_{self.current_floor}'
        sem_path = floor_dir / 'semantic_map_with_trajectory.png'
        if not sem_path.exists():
            sem_path = floor_dir / 'semantic_map.png'
        
        if sem_path.exists():
            vis = cv2.imread(str(sem_path))
            if vis is not None and vis.shape[:2] == (h, w):
                pass  # Use semantic map as background
            else:
                vis = None
        else:
            vis = None
        
        if vis is None:
            # Fallback to navigable map rendering
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[:, :] = [40, 40, 40]
            vis[exp_map > 0] = [150, 150, 150]
            vis[nav_map > 0.5] = [80, 80, 80]
        
        # Draw room annotations for current floor
        ROOM_COLORS = {
            'bedroom': (107, 107, 255), 'bathroom': (196, 205, 78),
            'kitchen': (109, 230, 255), 'living_room': (211, 225, 149),
            'dining_room': (129, 129, 243), 'hallway': (218, 150, 170),
            'office': (231, 92, 108), 'garage': (234, 216, 168),
            'laundry': (211, 186, 252), 'closet': (221, 160, 221),
        }
        
        floor_rooms = self.floor_room_annotations.get(self.current_floor, {})
        for label, rdata in floor_rooms.items():
            color = ROOM_COLORS.get(rdata.get('type', ''), (136, 136, 136))
            pts = np.array(rdata['polygon'], dtype=np.int32)
            if len(pts) >= 3:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
                cv2.polylines(vis, [pts], True, color, 1)
            if rdata.get('center'):
                cx, cy = int(rdata['center'][0]), int(rdata['center'][1])
                cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
                cv2.putText(vis, label, (cx - 20, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Objects (from YOLO detection — already on semantic map, but draw anyway)
        for obj in map_data.objects.values():
            pos = obj['position']
            mx, my = self.world_to_map(pos[0], pos[2])
            if 0 <= mx < w and 0 <= my < h:
                cv2.circle(vis, (mx, my), 3, (255, 200, 0), -1)
        
        # Path
        for i in range(len(self.path) - 1):
            p1, p2 = self.path[i], self.path[i + 1]
            color = (0, 255, 0) if i >= self.current_waypoint else (0, 100, 0)
            cv2.line(vis, p1, p2, color, 2)
        
        for i, wp in enumerate(self.path):
            color = (0, 255, 255) if i == self.current_waypoint else (0, 255, 0) if i > self.current_waypoint else (0, 100, 0)
            cv2.circle(vis, wp, 4, color, -1)
        
        # Goal
        if self.goal_map:
            cv2.circle(vis, self.goal_map, 10, (255, 0, 255), 2)
            cv2.putText(vis, self.goal_name, (self.goal_map[0] + 12, self.goal_map[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
        
        # Agent
        if agent_pos is not None:
            ax, ay = self.world_to_map(agent_pos[0], agent_pos[2])
            if 0 <= ax < w and 0 <= ay < h:
                cv2.circle(vis, (ax, ay), 6, (255, 100, 0), -1)
                cv2.circle(vis, (ax, ay), 6, (0, 0, 0), 2)
        
        return vis
    
    def get_status(self) -> Dict:
        """Get status."""
        return {
            'state': self.state.value,
            'goal': self.goal_name,
            'waypoint': f"{self.current_waypoint}/{len(self.path)}",
            'floor': self.current_floor,
            'actions': self.actions,
            'avoided': self.avoided,
        }


# =============================================================================
# Main Runner
# =============================================================================

def setup_habitat(scene_path: str):
    """Setup Habitat."""
    import habitat_sim
    
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [480, 640]
    rgb_spec.position = [0.0, 1.25, 0.0]
    rgb_spec.hfov = 90
    
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [480, 640]
    depth_spec.position = [0.0, 1.25, 0.0]
    depth_spec.hfov = 90
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)),
    }
    
    return habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))


class NavRunner:
    def __init__(self, sim, nav: Navigator):
        self.sim = sim
        self.nav = nav
        self.steps = 0
        self.start_time = time.time()
        
        # Dead reckoning if enabled (instead of VO)
        self.use_dr = nav.use_vo  # Reuse the flag, but use DR instead
        if self.use_dr:
            from slam.dead_reckoning import DeadReckoningLocalizer
            self.dr = DeadReckoningLocalizer()
            # Initialize with GT start position
            gt_pos, gt_rot = self.get_state()
            self.dr.initialize(gt_pos, gt_rot)
        else:
            self.dr = None
        
        # For comparison
        self.dr_positions = []
        self.gt_positions = []
    
    def get_state(self):
        state = self.sim.get_agent(0).get_state()
        pos = np.array(state.position)
        rot = np.array([state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w])
        return pos, rot
    
    def step(self, action=None):
        # Execute action in Habitat
        if action:
            self.sim.step(action)
            self.steps += 1
        
        obs = self.sim.get_sensor_observations()
        rgb = obs['color_sensor'][:,:,:3]
        depth = obs['depth_sensor']
        
        # Get ground truth pose (AFTER action was applied)
        gt_pos, gt_rot = self.get_state()
        
        # Use dead reckoning or GT
        if self.use_dr and self.dr is not None:
            # Update DR with the action we JUST executed (not last_action!)
            result = self.dr.update(action)
            
            # Store for comparison
            self.dr_positions.append(result.position.copy())
            self.gt_positions.append(gt_pos.copy())
            
            # Use DR position for navigation
            pos = result.position
            rot = result.quaternion
        else:
            pos = gt_pos
            rot = gt_rot
        
        return rgb, depth, pos, rot
    
    def create_vis(self, rgb, depth, pos):
        ph, pw = 280, 380
        
        # RGB with obstacle overlay
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Add obstacle warning overlay on RGB
        h, w = depth.shape
        center_depth = depth[h//3:2*h//3, w//3:2*w//3]
        valid = center_depth[(center_depth > 0.1) & (center_depth < 10)]
        min_depth = np.percentile(valid, 5) if len(valid) > 100 else 10.0
        
        if min_depth < 0.5:
            # Red overlay for obstacle warning
            overlay = rgb_bgr.copy()
            cv2.rectangle(overlay, (w//3, h//3), (2*w//3, 2*h//3), (0, 0, 255), 3)
            cv2.putText(overlay, f"OBSTACLE {min_depth:.2f}m", (w//3, h//3 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            rgb_bgr = cv2.addWeighted(rgb_bgr, 0.7, overlay, 0.3, 0)
        
        rgb_r = cv2.resize(rgb_bgr, (pw, ph))
        
        # Depth colormap
        dep_r = cv2.resize(cv2.applyColorMap((np.clip(depth/5,0,1)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), (pw, ph))
        
        # Map visualization
        map_vis = self.nav.visualize(pos)
        map_scale = min(pw / map_vis.shape[1], ph / map_vis.shape[0])
        map_r = cv2.resize(map_vis, (int(map_vis.shape[1]*map_scale), int(map_vis.shape[0]*map_scale)))
        
        # Pad map to panel size
        map_panel = np.zeros((ph, pw, 3), np.uint8)
        map_panel[:, :] = [30, 30, 30]
        y_off = (ph - map_r.shape[0]) // 2
        x_off = (pw - map_r.shape[1]) // 2
        map_panel[y_off:y_off+map_r.shape[0], x_off:x_off+map_r.shape[1]] = map_r
        
        # Combine: RGB | Map on top, Depth | Info on bottom
        top = np.hstack([rgb_r, map_panel])
        
        # Info panel
        info_panel = np.zeros((ph, pw, 3), np.uint8)
        info_panel[:, :] = [30, 30, 30]
        
        status = self.nav.get_status()
        agent_map = self.nav.world_to_map(pos[0], pos[2])
        
        # State color
        state_color = {
            'navigating': (0, 255, 0),
            'avoiding': (0, 255, 255),
            'reached': (0, 255, 0),
            'failed': (0, 0, 255),
            'idle': (150, 150, 150),
        }.get(status['state'], (200, 200, 200))
        
        lines = [
            (f"State: {status['state']}", state_color),
            (f"Floor: {status['floor']}", (255, 255, 0)),
            (f"Goal: {status['goal']}", (255, 255, 255)),
            (f"Waypoint: {status['waypoint']}", (200, 200, 200)),
            ("", (0, 0, 0)),
            (f"Pos: ({pos[0]:.2f}, {pos[2]:.2f})", (200, 200, 200)),
            (f"Map: {agent_map}", (200, 200, 200)),
            (f"Depth: {min_depth:.2f}m", (0, 255, 255) if min_depth < 0.5 else (200, 200, 200)),
            ("", (0, 0, 0)),
            (f"Steps: {self.steps}", (200, 200, 200)),
            (f"Avoided: {status['avoided']}", (200, 200, 200)),
        ]
        
        # Add DR drift if using dead reckoning
        if self.use_dr and len(self.dr_positions) > 0 and len(self.gt_positions) > 0:
            drift = np.linalg.norm(np.array(self.dr_positions[-1]) - np.array(self.gt_positions[-1]))
            drift_color = (0, 255, 0) if drift < 0.3 else (0, 255, 255) if drift < 1.0 else (0, 0, 255)
            lines.append((f"DR Drift: {drift:.2f}m", drift_color))
        
        for i, (line, color) in enumerate(lines):
            cv2.putText(info_panel, line, (10, 25 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        
        bot = np.hstack([dep_r, info_panel])
        vis = np.vstack([top, bot])
        
        # Labels
        cv2.putText(vis, "RGB", (5, ph-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        cv2.putText(vis, f"Map (Floor {status['floor']})", (pw+5, ph-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        cv2.putText(vis, "Depth", (5, 2*ph-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        cv2.putText(vis, "Status", (pw+5, 2*ph-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        
        cv2.putText(vis, "1-9:Object | N:Instruction | Click:Position | C:Cancel | WASD:Manual | Q:Quit",
                   (10, vis.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,150,150), 1)
        
        return vis, ph, pw
    
    def run(self):
        print("\n" + "="*60)
        print("Navigator V2")
        print("="*60)
        
        objects = self.nav.get_available_objects()
        print(f"Objects: {objects}")
        
        # Hotkeys
        hotkeys = {}
        common = ['toilet', 'bed', 'chair', 'couch', 'tv', 'refrigerator', 'sink', 'dining table', 'potted plant']
        for i, obj in enumerate(common):
            if obj in objects and i < 9:
                hotkeys[ord('1') + i] = obj
        print(f"Hotkeys: {dict((chr(k), v) for k,v in hotkeys.items())}")
        print("="*60)
        
        window = 'Navigator V2'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 800, 600)
        
        rgb, depth, pos, rot = self.step()
        auto_nav = False
        panel_info = {'h': 280, 'w': 380}
        
        # Show initial position
        print(f"Agent start: world({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        print(f"Agent start: map{self.nav.world_to_map(pos[0], pos[2])}")
        
        def mouse_cb(event, x, y, flags, param):
            nonlocal auto_nav
            if event == cv2.EVENT_LBUTTONDOWN:
                ph, pw = panel_info['h'], panel_info['w']
                # Click on map panel (top right)
                if x >= pw and y < ph:
                    map_vis = self.nav.visualize(pos)
                    map_scale = min(pw / map_vis.shape[1], ph / map_vis.shape[0])
                    
                    # Convert click to map coords
                    x_off = (pw - int(map_vis.shape[1]*map_scale)) // 2
                    y_off = (ph - int(map_vis.shape[0]*map_scale)) // 2
                    
                    mx = int((x - pw - x_off) / map_scale)
                    my = int((y - y_off) / map_scale)
                    
                    if 0 <= mx < map_vis.shape[1] and 0 <= my < map_vis.shape[0]:
                        print(f"Click: map({mx}, {my})")
                        if self.nav.set_goal_position(mx, my, pos):
                            auto_nav = True
        
        cv2.setMouseCallback(window, mouse_cb)
        
        while True:
            vis, ph, pw = self.create_vis(rgb, depth, pos)
            panel_info['h'], panel_info['w'] = ph, pw
            cv2.imshow(window, vis)
            
            key = cv2.waitKey(30) & 0xFF
            action = None
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('w') or key == 82:
                action = 'move_forward'; auto_nav = False; self.nav.cancel()
            elif key == ord('a') or key == 81:
                action = 'turn_left'; auto_nav = False; self.nav.cancel()
            elif key == ord('d') or key == 83:
                action = 'turn_right'; auto_nav = False; self.nav.cancel()
            elif key in hotkeys:
                obj = hotkeys[key]
                print(f"\nNavigate to: {obj}")
                if self.nav.set_goal_object(obj, pos):
                    auto_nav = True
            elif key == ord('c'):
                print("Cancelled"); self.nav.cancel(); auto_nav = False
            elif key == ord('n'):
                # Natural language instruction
                print("\nType instruction: ", end='', flush=True)
                try:
                    instruction = input().strip()
                    if instruction:
                        if self.nav.set_goal_from_instruction(instruction, pos):
                            auto_nav = True
                except EOFError:
                    pass
            
            if auto_nav:
                action = self.nav.get_action(pos, rot, depth)
                if action is None:
                    print(f"Navigation ended: {self.nav.state.value}")
                    auto_nav = False
            
            if action:
                rgb, depth, pos, rot = self.step(action)
            
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
        print(f"\nSteps: {self.steps}, Avoided: {self.nav.avoided}")
        
        # Print DR stats if used
        if self.use_dr and len(self.dr_positions) > 1:
            dr_arr = np.array(self.dr_positions)
            gt_arr = np.array(self.gt_positions)
            n = min(len(dr_arr), len(gt_arr))
            errors = np.linalg.norm(dr_arr[:n] - gt_arr[:n], axis=1)
            print(f"DR Stats - Mean drift: {np.mean(errors):.3f}m, Max: {np.max(errors):.3f}m, Final: {errors[-1]:.3f}m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--map-dir', required=True)
    parser.add_argument('--use-vo', action='store_true', 
                       help='Use visual odometry instead of ground truth pose')
    args = parser.parse_args()
    
    print("Loading...")
    sim = setup_habitat(args.scene)
    
    # Set agent to navigable point
    agent = sim.get_agent(0)
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
    
    nav = Navigator(args.map_dir, use_visual_odometry=args.use_vo)
    
    if args.use_vo:
        print("Using Dead Reckoning for pose estimation")
    else:
        print("Using Ground Truth pose from Habitat")
    
    runner = NavRunner(sim, nav)
    runner.run()
    sim.close()


if __name__ == '__main__':
    main()