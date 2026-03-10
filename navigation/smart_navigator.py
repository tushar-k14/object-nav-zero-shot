"""
Smart Navigator
===============

Integrated navigation system with:
1. SLAM with collision detection
2. YOLO object detection during navigation
3. VLM (Qwen/Ollama) for natural language commands
4. Landmark-based pose correction
5. Semantic map building

For real-world deployment preparation.
"""

import numpy as np
import cv2
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import heapq
import time


# ============================================================
# Enums and Data Classes
# ============================================================

class NavState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    REACHED = "reached"
    FAILED = "failed"
    EXPLORING = "exploring"


@dataclass
class Landmark:
    """Detected landmark for localization."""
    class_name: str
    position: np.ndarray
    confidence: float
    last_seen: int
    observations: int = 1


@dataclass
class FloorData:
    """Map data for one floor."""
    floor_id: int
    navigable_map: np.ndarray
    explored_map: np.ndarray
    objects: Dict
    landmarks: Dict[str, List[Landmark]] = field(default_factory=lambda: defaultdict(list))


# ============================================================
# YOLO Detector
# ============================================================

class ObjectDetector:
    """YOLO-based object detector."""
    
    INDOOR_CLASSES = {
        'person', 'chair', 'couch', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'refrigerator', 'sink', 'potted plant', 'clock',
        'book', 'vase', 'oven', 'microwave', 'bottle', 'cup'
    }
    
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35):
        self.model = None
        self.conf = conf
        self.class_names = {}
        self.enabled = False
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.class_names = self.model.names
            self.enabled = True
            print(f"YOLO loaded: {model_name}")
        except Exception as e:
            print(f"YOLO not available: {e}")
    
    def detect(self, rgb: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Detect objects.
        
        Returns:
            List of (class_name, bbox, confidence)
        """
        if not self.enabled:
            return []
        
        results = self.model(rgb, verbose=False, conf=self.conf)
        
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                class_name = self.class_names.get(cls_id, "unknown")
                
                if class_name.lower() not in self.INDOOR_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append((class_name, (x1, y1, x2, y2), conf))
        
        return detections
    
    def detect_with_depth(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray,
        agent_pos: np.ndarray,
        agent_yaw: float,
        camera_K: np.ndarray
    ) -> List[Tuple[str, np.ndarray, float]]:
        """
        Detect objects and compute world positions.
        
        Returns:
            List of (class_name, world_position, confidence)
        """
        detections = self.detect(rgb)
        world_detections = []
        
        for class_name, (x1, y1, x2, y2), conf in detections:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Get depth
            margin = 10
            h, w = depth.shape
            region = depth[
                max(0, cy-margin):min(h, cy+margin),
                max(0, cx-margin):min(w, cx+margin)
            ]
            valid = region[(region > 0.1) & (region < 8.0)]
            
            if len(valid) < 5:
                continue
            
            obj_depth = np.median(valid)
            
            # Backproject to world
            fx = camera_K[0, 0]
            cx_cam = camera_K[0, 2]
            
            x_cam = (cx - cx_cam) * obj_depth / fx
            z_cam = obj_depth
            
            # Camera to world
            cos_yaw = np.cos(agent_yaw)
            sin_yaw = np.sin(agent_yaw)
            
            # At yaw=0, camera forward is -Z
            x_world = -x_cam * cos_yaw - z_cam * sin_yaw + agent_pos[0]
            z_world = x_cam * sin_yaw - z_cam * cos_yaw + agent_pos[2]
            
            world_pos = np.array([x_world, agent_pos[1], z_world])
            world_detections.append((class_name, world_pos, conf))
        
        return world_detections
    
    def visualize(self, rgb: np.ndarray, depth: np.ndarray = None) -> np.ndarray:
        """Draw detections on image."""
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        detections = self.detect(rgb)
        
        for class_name, (x1, y1, x2, y2), conf in detections:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{class_name} {conf:.2f}"
            if depth is not None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    d = depth[cy, cx]
                    if 0.1 < d < 10:
                        label = f"{class_name} {d:.1f}m"
            
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis


# ============================================================
# VLM Navigator (Ollama/Qwen)
# ============================================================

class VLMNavigator:
    """VLM for natural language navigation."""
    
    def __init__(self, use_ollama: bool = True):
        self.available = False
        self.use_ollama = use_ollama
        self.model = None
        
        if use_ollama:
            self._init_ollama()
        else:
            self._init_transformers()
    
    def _init_ollama(self):
        """Initialize with Ollama (local, faster)."""
        try:
            import requests
            # Test if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.available = True
                print("VLM: Using Ollama (local)")
        except:
            print("Ollama not available, trying transformers...")
            self._init_transformers()
    
    def _init_transformers(self):
        """Initialize with transformers."""
        try:
            # Use a smaller model that's more compatible
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_name = "Qwen/Qwen2-0.5B-Instruct"
            print(f"Loading VLM: {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            self.available = True
            self.use_ollama = False
            print("VLM loaded (transformers)")
            
        except Exception as e:
            print(f"VLM not available: {e}")
            self.available = False
    
    def parse_command(self, command: str, available_objects: List[str]) -> Optional[str]:
        """
        Parse natural language command to target object.
        
        Args:
            command: User command like "go to bathroom", "find a chair"
            available_objects: List of detected object classes
            
        Returns:
            Target object class or None
        """
        if not self.available:
            return self._keyword_fallback(command, available_objects)
        
        if self.use_ollama:
            return self._parse_ollama(command, available_objects)
        else:
            return self._parse_transformers(command, available_objects)
    
    def _parse_ollama(self, command: str, available_objects: List[str]) -> Optional[str]:
        """Parse using Ollama."""
        try:
            import requests
            
            prompt = f"""You are a robot navigation assistant. The user said: "{command}"
Available objects in the environment: {', '.join(available_objects) if available_objects else 'none detected yet'}

Which object should the robot navigate to? Reply with ONLY the object name, nothing else.
If asking for a room (bathroom, kitchen, etc), reply with an object typically found there.
If target not available, reply "explore"."""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",  # or qwen2
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()['response'].strip().lower()
                
                # Match to available
                for obj in available_objects:
                    if obj.lower() in result or result in obj.lower():
                        return obj
                
                if "explore" in result:
                    return "explore"
            
            return self._keyword_fallback(command, available_objects)
            
        except Exception as e:
            print(f"Ollama error: {e}")
            return self._keyword_fallback(command, available_objects)
    
    def _parse_transformers(self, command: str, available_objects: List[str]) -> Optional[str]:
        """Parse using transformers model."""
        try:
            prompt = f"""User command: "{command}"
Available objects: {', '.join(available_objects) if available_objects else 'none'}
Target object (one word):"""

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = result.split("Target object (one word):")[-1].strip().lower()
            
            for obj in available_objects:
                if obj.lower() in result or result in obj.lower():
                    return obj
            
            return self._keyword_fallback(command, available_objects)
            
        except Exception as e:
            print(f"Transformers error: {e}")
            return self._keyword_fallback(command, available_objects)
    
    def _keyword_fallback(self, command: str, available: List[str]) -> Optional[str]:
        """Simple keyword matching fallback."""
        command = command.lower()
        
        # Room -> object mapping
        room_map = {
            'bathroom': ['toilet', 'sink'],
            'restroom': ['toilet', 'sink'],
            'kitchen': ['refrigerator', 'oven', 'microwave', 'sink'],
            'bedroom': ['bed'],
            'living': ['couch', 'tv', 'sofa'],
            'office': ['chair', 'laptop', 'desk'],
            'dining': ['dining table', 'chair'],
        }
        
        # Check rooms
        for room, objects in room_map.items():
            if room in command:
                for obj in objects:
                    if obj in [o.lower() for o in available]:
                        return obj
        
        # Direct match
        for obj in available:
            if obj.lower() in command:
                return obj
        
        return None


# ============================================================
# Collision Detector
# ============================================================

class CollisionDetector:
    """Detect collisions from depth changes."""
    
    def __init__(self, expected_forward: float = 0.25):
        self.expected_forward = expected_forward
        self.prev_depth = None
        self.collision_count = 0
    
    def check_collision(
        self, 
        depth: np.ndarray, 
        action: str,
        actual_movement: float = None
    ) -> Tuple[bool, float]:
        """
        Check if collision occurred.
        
        Args:
            depth: Current depth image
            action: Action that was taken
            actual_movement: If known, actual movement distance
            
        Returns:
            (collision_detected, estimated_actual_movement)
        """
        if action != 'move_forward':
            self.prev_depth = depth.copy()
            return False, 0.0
        
        if self.prev_depth is None:
            self.prev_depth = depth.copy()
            return False, self.expected_forward
        
        # Compare center depth before/after
        h, w = depth.shape
        cy, cx = h // 2, w // 2
        margin = 50
        
        prev_center = self.prev_depth[cy-margin:cy+margin, cx-margin:cx+margin]
        curr_center = depth[cy-margin:cy+margin, cx-margin:cx+margin]
        
        prev_valid = prev_center[(prev_center > 0.1) & (prev_center < 8)]
        curr_valid = curr_center[(curr_center > 0.1) & (curr_center < 8)]
        
        if len(prev_valid) < 100 or len(curr_valid) < 100:
            self.prev_depth = depth.copy()
            return False, self.expected_forward
        
        prev_median = np.median(prev_valid)
        curr_median = np.median(curr_valid)
        
        # If we moved forward, depth to objects should decrease
        expected_decrease = self.expected_forward
        actual_decrease = prev_median - curr_median
        
        # Collision if depth didn't decrease as expected
        collision = actual_decrease < expected_decrease * 0.3
        
        if collision:
            self.collision_count += 1
            estimated_movement = max(0, actual_decrease)
        else:
            estimated_movement = self.expected_forward
        
        self.prev_depth = depth.copy()
        return collision, estimated_movement


# ============================================================
# Smart Navigator
# ============================================================

class SmartNavigator:
    """
    Integrated navigation with SLAM, YOLO, and VLM.
    """
    
    # Constants
    FORWARD_DIST = 0.25
    TURN_ANGLE = np.radians(10.0)
    CAMERA_HEIGHT = 1.25
    
    TURN_THRESHOLD = np.radians(15)
    OBSTACLE_CLOSE = 0.4
    OBSTACLE_FAR = 0.8
    WAYPOINT_TOLERANCE = 10
    GOAL_TOLERANCE = 15
    
    def __init__(
        self,
        map_dir: str = None,
        use_yolo: bool = True,
        use_vlm: bool = True,
        use_ollama: bool = True
    ):
        """
        Initialize navigator.
        
        Args:
            map_dir: Path to pre-built maps (optional)
            use_yolo: Enable YOLO detection
            use_vlm: Enable VLM for natural language
            use_ollama: Use Ollama for VLM (vs transformers)
        """
        # Components
        self.detector = ObjectDetector() if use_yolo else None
        self.vlm = VLMNavigator(use_ollama=use_ollama) if use_vlm else None
        self.collision_detector = CollisionDetector()
        
        # Maps
        self.map_dir = Path(map_dir) if map_dir else None
        self.floors: Dict[int, FloorData] = {}
        self.floor_heights: Dict[int, float] = {}
        
        # Map parameters
        self.resolution = 5.0
        self.map_size = 480
        self.origin_x = 0.0
        self.origin_z = 0.0
        
        # Pose (SLAM)
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.current_floor = 0
        self.prev_y = 0.0
        
        # Camera
        self.camera_K = self._default_camera_K()
        
        # Navigation
        self.state = NavState.IDLE
        self.goal_name = ""
        self.goal_map: Optional[Tuple[int, int]] = None
        self.path: List[Tuple[int, int]] = []
        self.current_waypoint = 0
        
        # Recovery
        self.recovery_actions: deque = deque()
        self.position_history: deque = deque(maxlen=20)
        
        # Landmarks (runtime detected)
        self.landmarks: Dict[str, List[Landmark]] = defaultdict(list)
        
        # Stats
        self.frame_count = 0
        self.total_distance = 0.0
        self.collisions = 0
        self.gt_errors: List[float] = []
        
        # Load maps if provided
        if self.map_dir:
            self._load_maps()
    
    def _default_camera_K(self) -> np.ndarray:
        """Default camera intrinsics."""
        w, h = 640, 480
        hfov = np.radians(90)
        fx = w / (2.0 * np.tan(hfov / 2))
        return np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], dtype=np.float32)
    
    def _load_maps(self):
        """Load pre-built maps."""
        if not self.map_dir or not self.map_dir.exists():
            return
        
        # Load stats
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
        floor_ids = [0]
        if floor_info_path.exists():
            with open(floor_info_path) as f:
                fi = json.load(f)
            floor_ids = fi.get('floor_ids', [0])
            for k, v in fi.get('base_heights', {}).items():
                self.floor_heights[int(k)] = float(v)
        
        # Load each floor
        for fid in floor_ids:
            floor_dir = self.map_dir / f'floor_{fid}'
            if not floor_dir.exists():
                continue
            
            nav_path = floor_dir / 'navigable_map.pt'
            if not nav_path.exists():
                nav_path = floor_dir / 'occupancy_map.pt'
            if not nav_path.exists():
                continue
            
            nav_map = torch.load(nav_path).numpy()
            
            exp_path = floor_dir / 'explored_map.pt'
            exp_map = torch.load(exp_path).numpy() if exp_path.exists() else np.ones_like(nav_map)
            
            obj_path = floor_dir / 'objects.json'
            objects = {}
            if obj_path.exists():
                with open(obj_path) as f:
                    objects = json.load(f)
            
            self.floors[fid] = FloorData(
                floor_id=fid,
                navigable_map=nav_map,
                explored_map=exp_map,
                objects=objects
            )
        
        print(f"Loaded {len(self.floors)} floors")
    
    # ========================================
    # Coordinate Conversion
    # ========================================
    
    def world_to_map(self, x: float, z: float) -> Tuple[int, int]:
        """Convert world coordinates to map pixels."""
        mx = int((x - self.origin_x) / (self.resolution / 100.0) + self.map_size / 2)
        my = int((z - self.origin_z) / (self.resolution / 100.0) + self.map_size / 2)
        return mx, my
    
    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """Convert map pixels to world coordinates."""
        x = (mx - self.map_size / 2) * (self.resolution / 100.0) + self.origin_x
        z = (my - self.map_size / 2) * (self.resolution / 100.0) + self.origin_z
        return x, z
    
    # ========================================
    # Initialization
    # ========================================
    
    def initialize(self, position: np.ndarray, yaw: float):
        """Initialize with known pose."""
        self.position = position.copy()
        self.yaw = yaw
        self.prev_y = position[1]
        self.frame_count = 0
        self.total_distance = 0.0
        self.collisions = 0
        self.gt_errors.clear()
        self.landmarks.clear()
        self.state = NavState.IDLE
        
        # Detect floor
        self._update_floor(position[1])
        
        print(f"Navigator initialized: ({position[0]:.2f}, {position[2]:.2f}), floor {self.current_floor}")
    
    def _update_floor(self, y: float):
        """Update floor based on Y coordinate."""
        if abs(y - self.prev_y) > 0.3:
            if y > self.prev_y:
                self.current_floor += 1
            else:
                self.current_floor -= 1
            self.position[1] = y
            print(f"Floor change -> {self.current_floor}")
        self.prev_y = y
    
    # ========================================
    # Navigation Commands
    # ========================================
    
    def navigate_to(self, target: str) -> bool:
        """
        Navigate to object by class name.
        
        Args:
            target: Object class like "toilet", "bed", etc.
            
        Returns:
            True if goal set successfully
        """
        target = target.lower()
        
        # Find in map objects
        candidates = []
        
        if self.current_floor in self.floors:
            for obj_id, obj in self.floors[self.current_floor].objects.items():
                if obj.get('class', '').lower() == target:
                    pos = obj.get('position', [0, 0, 0])
                    dist = np.sqrt((pos[0] - self.position[0])**2 + 
                                   (pos[2] - self.position[2])**2)
                    candidates.append((dist, pos, obj_id))
        
        # Also check runtime landmarks
        if target in self.landmarks:
            for lm in self.landmarks[target]:
                dist = np.sqrt((lm.position[0] - self.position[0])**2 +
                              (lm.position[2] - self.position[2])**2)
                candidates.append((dist, lm.position, f"lm_{target}"))
        
        if not candidates:
            print(f"Object '{target}' not found")
            return False
        
        # Select nearest
        candidates.sort(key=lambda x: x[0])
        _, pos, obj_id = candidates[0]
        
        self.goal_name = target
        self.goal_map = self.world_to_map(pos[0], pos[2])
        
        print(f"Navigating to {target} at map{self.goal_map}")
        
        return self._plan_path()
    
    def navigate_with_command(self, command: str, rgb: np.ndarray = None) -> bool:
        """
        Navigate using natural language command.
        
        Args:
            command: Natural language like "go to bathroom", "find a chair"
            rgb: Optional current view for VLM
            
        Returns:
            True if goal set successfully
        """
        available = self.get_available_objects()
        
        if self.vlm:
            target = self.vlm.parse_command(command, available)
        else:
            target = self._keyword_match(command, available)
        
        if target is None:
            print(f"Could not parse command: {command}")
            print(f"Available objects: {available}")
            return False
        
        if target == "explore":
            print("Target not found, need to explore")
            self.state = NavState.EXPLORING
            return False
        
        return self.navigate_to(target)
    
    def _keyword_match(self, command: str, available: List[str]) -> Optional[str]:
        """Fallback keyword matching."""
        command = command.lower()
        
        room_map = {
            'bathroom': ['toilet', 'sink'],
            'kitchen': ['refrigerator', 'oven', 'sink'],
            'bedroom': ['bed'],
            'living': ['couch', 'tv'],
        }
        
        for room, objs in room_map.items():
            if room in command:
                for obj in objs:
                    if obj in [o.lower() for o in available]:
                        return obj
        
        for obj in available:
            if obj.lower() in command:
                return obj
        
        return None
    
    def get_available_objects(self) -> List[str]:
        """Get list of all detected object classes."""
        objects = set()
        
        # From maps
        for floor in self.floors.values():
            for obj in floor.objects.values():
                objects.add(obj.get('class', '').lower())
        
        # From runtime landmarks
        for cls in self.landmarks.keys():
            objects.add(cls.lower())
        
        return sorted([o for o in objects if o])
    
    # ========================================
    # Path Planning
    # ========================================
    
    def _plan_path(self) -> bool:
        """Plan path to current goal."""
        if self.goal_map is None:
            return False
        
        if self.current_floor not in self.floors:
            print(f"No map for floor {self.current_floor}")
            return False
        
        floor = self.floors[self.current_floor]
        
        start = self.world_to_map(self.position[0], self.position[2])
        goal = self.goal_map
        
        # Clamp to map bounds
        start = (
            max(0, min(self.map_size - 1, start[0])),
            max(0, min(self.map_size - 1, start[1]))
        )
        goal = (
            max(0, min(self.map_size - 1, goal[0])),
            max(0, min(self.map_size - 1, goal[1]))
        )
        
        # A* planning
        path = self._astar(start, goal, floor.navigable_map, floor.explored_map)
        
        if path is None:
            print("No path found")
            self.state = NavState.FAILED
            return False
        
        # Simplify path
        self.path = path[::3]  # Every 3rd point
        self.current_waypoint = 0
        self.state = NavState.NAVIGATING
        
        print(f"Path found: {len(self.path)} waypoints")
        return True
    
    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: np.ndarray,
        explored: np.ndarray
    ) -> Optional[List[Tuple[int, int]]]:
        """A* path planning."""
        h, w = obstacles.shape
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def valid(x, y):
            if x < 0 or x >= w or y < 0 or y >= h:
                return False
            if obstacles[y, x] > 0.5:
                return False
            return True
        
        # Find valid start/goal
        if not valid(start[0], start[1]):
            start = self._find_valid_cell(start, obstacles)
        if not valid(goal[0], goal[1]):
            goal = self._find_valid_cell(goal, obstacles)
        
        if start is None or goal is None:
            return None
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        iterations = 0
        max_iter = 50000
        
        while open_set and iterations < max_iter:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                if not valid(nx, ny):
                    continue
                
                # Cost
                move_cost = 1.4 if dx != 0 and dy != 0 else 1.0
                
                # Penalty for unexplored
                if explored[ny, nx] < 0.5:
                    move_cost += 5.0
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return None
    
    def _find_valid_cell(
        self, 
        pos: Tuple[int, int], 
        obstacles: np.ndarray,
        max_dist: int = 20
    ) -> Optional[Tuple[int, int]]:
        """Find nearest valid cell."""
        h, w = obstacles.shape
        
        for d in range(1, max_dist):
            for dx in range(-d, d + 1):
                for dy in range(-d, d + 1):
                    if abs(dx) != d and abs(dy) != d:
                        continue
                    
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if obstacles[ny, nx] < 0.5:
                            return (nx, ny)
        
        return None
    
    # ========================================
    # Main Update Loop
    # ========================================
    
    def update(
        self,
        action: Optional[str],
        rgb: np.ndarray,
        depth: np.ndarray,
        gt_pose: Tuple[np.ndarray, float] = None
    ) -> Dict[str, Any]:
        """
        Update navigator after action execution.
        
        Args:
            action: Action that was just executed
            rgb: Current RGB image
            depth: Current depth image
            gt_pose: Optional (position, yaw) ground truth
            
        Returns:
            Dict with state info
        """
        self.frame_count += 1
        
        # 1. Check collision
        collision, actual_movement = self.collision_detector.check_collision(depth, action)
        if collision:
            self.collisions += 1
        
        # 2. Dead reckoning (with collision correction)
        self._apply_action(action, actual_movement if collision else None)
        
        # 3. Floor change
        if gt_pose:
            self._update_floor(gt_pose[0][1])
        
        # 4. YOLO detection
        detections = []
        if self.detector and self.detector.enabled:
            detections = self.detector.detect_with_depth(
                rgb, depth, self.position, self.yaw, self.camera_K
            )
            
            # Add to landmarks
            for cls, pos, conf in detections:
                self._add_landmark(cls, pos, conf)
        
        # 5. Landmark-based correction
        if len(detections) >= 2:
            self._landmark_correction(detections)
        
        # 6. GT comparison
        error_2d = None
        if gt_pose:
            gt_pos, gt_yaw = gt_pose
            error_2d = np.sqrt(
                (self.position[0] - gt_pos[0])**2 +
                (self.position[2] - gt_pos[2])**2
            )
            self.gt_errors.append(error_2d)
        
        # 7. Track position
        self.position_history.append(self.position[[0, 2]].copy())
        
        return {
            'position': self.position.copy(),
            'yaw': self.yaw,
            'floor': self.current_floor,
            'state': self.state.value,
            'goal': self.goal_name,
            'detections': len(detections),
            'collision': collision,
            'error_2d': error_2d,
        }
    
    def _apply_action(self, action: Optional[str], actual_dist: float = None):
        """Apply dead reckoning."""
        if action is None:
            return
        
        if action == 'move_forward':
            dist = actual_dist if actual_dist is not None else self.FORWARD_DIST
            self.position[0] -= np.sin(self.yaw) * dist
            self.position[2] -= np.cos(self.yaw) * dist
            self.total_distance += dist
            
        elif action == 'turn_left':
            self.yaw += self.TURN_ANGLE
        elif action == 'turn_right':
            self.yaw -= self.TURN_ANGLE
        
        while self.yaw > np.pi: self.yaw -= 2*np.pi
        while self.yaw < -np.pi: self.yaw += 2*np.pi
    
    def _add_landmark(self, class_name: str, position: np.ndarray, confidence: float):
        """Add or update landmark."""
        class_name = class_name.lower()
        
        # Check for nearby existing
        for lm in self.landmarks[class_name]:
            dist = np.linalg.norm(lm.position - position)
            if dist < 1.0:
                # Update
                alpha = 0.3
                lm.position = (1 - alpha) * lm.position + alpha * position
                lm.confidence = max(lm.confidence, confidence)
                lm.last_seen = self.frame_count
                lm.observations += 1
                return
        
        # New landmark
        self.landmarks[class_name].append(Landmark(
            class_name=class_name,
            position=position.copy(),
            confidence=confidence,
            last_seen=self.frame_count
        ))
    
    def _landmark_correction(self, detections: List[Tuple[str, np.ndarray, float]]):
        """Correct pose using landmarks."""
        # Need landmarks with multiple observations
        corrections = []
        
        for cls, obs_pos, conf in detections:
            cls = cls.lower()
            
            # Find corresponding landmark in map
            best_lm = None
            min_dist = float('inf')
            
            for lm in self.landmarks[cls]:
                if lm.observations >= 3:
                    dist = np.linalg.norm(lm.position[[0, 2]] - self.position[[0, 2]])
                    if dist < min_dist and dist < 5.0:
                        min_dist = dist
                        best_lm = lm
            
            if best_lm is not None:
                correction = best_lm.position - obs_pos
                corrections.append((correction[[0, 2]], conf))
        
        if len(corrections) >= 2:
            total_weight = sum(c[1] for c in corrections)
            avg_correction = np.zeros(2)
            for corr, weight in corrections:
                avg_correction += corr * weight / total_weight
            
            if np.linalg.norm(avg_correction) < 1.0:
                alpha = 0.2
                self.position[0] += alpha * avg_correction[0]
                self.position[2] += alpha * avg_correction[1]
    
    # ========================================
    # Get Action
    # ========================================
    
    def get_action(self, depth: np.ndarray) -> Optional[str]:
        """
        Get next navigation action.
        
        Args:
            depth: Current depth image
            
        Returns:
            Action string or None if done
        """
        # Recovery first
        if self.recovery_actions:
            return self.recovery_actions.popleft()
        
        # Check stuck
        if self._is_stuck():
            self._start_recovery()
            if self.recovery_actions:
                return self.recovery_actions.popleft()
        
        # Not navigating
        if self.state != NavState.NAVIGATING or not self.path:
            return None
        
        # Check goal reached
        agent_map = self.world_to_map(self.position[0], self.position[2])
        
        if self.goal_map:
            goal_dist = np.sqrt(
                (agent_map[0] - self.goal_map[0])**2 +
                (agent_map[1] - self.goal_map[1])**2
            )
            if goal_dist < self.GOAL_TOLERANCE:
                self.state = NavState.REACHED
                print(f"REACHED: {self.goal_name}")
                return None
        
        # Check waypoint
        if self.current_waypoint >= len(self.path):
            self.state = NavState.REACHED
            return None
        
        target = self.path[self.current_waypoint]
        wp_dist = np.sqrt(
            (agent_map[0] - target[0])**2 +
            (agent_map[1] - target[1])**2
        )
        
        if wp_dist < self.WAYPOINT_TOLERANCE:
            self.current_waypoint += 1
            if self.current_waypoint >= len(self.path):
                self.state = NavState.REACHED
                return None
            target = self.path[self.current_waypoint]
        
        # Compute action
        return self._compute_action(target, depth)
    
    def _compute_action(self, target: Tuple[int, int], depth: np.ndarray) -> str:
        """Compute action toward target with avoidance."""
        # Target world position
        tx, tz = self.map_to_world(target[0], target[1])
        
        # Angle to target
        dx = tx - self.position[0]
        dz = tz - self.position[2]
        target_angle = np.arctan2(-dx, -dz)
        
        angle_diff = target_angle - self.yaw
        while angle_diff > np.pi: angle_diff -= 2*np.pi
        while angle_diff < -np.pi: angle_diff += 2*np.pi
        
        # Check obstacles
        obstacle_info = self._analyze_depth(depth)
        
        # Decision
        if obstacle_info['critical']:
            self.state = NavState.AVOIDING
            # Turn away from obstacle
            if obstacle_info['left_clear'] > obstacle_info['right_clear']:
                return 'turn_left'
            else:
                return 'turn_right'
        
        if abs(angle_diff) > self.TURN_THRESHOLD:
            return 'turn_left' if angle_diff > 0 else 'turn_right'
        
        if obstacle_info['blocking']:
            self.state = NavState.AVOIDING
            if angle_diff > 0:
                return 'turn_left'
            else:
                return 'turn_right'
        
        self.state = NavState.NAVIGATING
        return 'move_forward'
    
    def _analyze_depth(self, depth: np.ndarray) -> Dict:
        """Analyze depth for obstacles."""
        h, w = depth.shape
        
        # Regions
        left = depth[h//3:2*h//3, :w//3]
        center = depth[h//3:2*h//3, w//3:2*w//3]
        right = depth[h//3:2*h//3, 2*w//3:]
        
        def min_dist(region):
            valid = region[(region > 0.1) & (region < 8)]
            if len(valid) < 50:
                return 5.0
            return np.percentile(valid, 5)
        
        left_min = min_dist(left)
        center_min = min_dist(center)
        right_min = min_dist(right)
        
        return {
            'critical': center_min < self.OBSTACLE_CLOSE,
            'blocking': center_min < self.OBSTACLE_FAR,
            'left_clear': left_min,
            'right_clear': right_min,
            'center': center_min,
        }
    
    def _is_stuck(self) -> bool:
        """Check if stuck."""
        if len(self.position_history) < 15:
            return False
        
        positions = list(self.position_history)
        total = sum(np.linalg.norm(positions[i] - positions[i-1])
                   for i in range(1, len(positions)))
        return total < 0.15
    
    def _start_recovery(self):
        """Start recovery maneuver."""
        print("STUCK - recovery")
        self.recovery_actions = deque([
            'turn_left', 'turn_left', 'turn_left',
            'turn_left', 'turn_left', 'turn_left',
            'move_forward', 'move_forward',
        ])
        self.position_history.clear()
    
    # ========================================
    # Utilities
    # ========================================
    
    def cancel(self):
        """Cancel navigation."""
        self.state = NavState.IDLE
        self.goal_name = ""
        self.goal_map = None
        self.path = []
        self.recovery_actions.clear()
    
    def correct_pose(self, position: np.ndarray, yaw: float = None):
        """Manual pose correction."""
        self.position = position.copy()
        if yaw is not None:
            self.yaw = yaw
        self.prev_y = position[1]
        print(f"Pose corrected to ({position[0]:.2f}, {position[2]:.2f})")
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        stats = {
            'frames': self.frame_count,
            'distance': self.total_distance,
            'collisions': self.collisions,
            'landmarks': sum(len(v) for v in self.landmarks.values()),
            'state': self.state.value,
        }
        
        if self.gt_errors:
            stats['error_mean'] = np.mean(self.gt_errors)
            stats['error_current'] = self.gt_errors[-1]
            stats['drift'] = self.gt_errors[-1] / max(self.total_distance, 0.01) * 100
        
        return stats
    
    def visualize(self, rgb: np.ndarray = None, depth: np.ndarray = None) -> np.ndarray:
        """Create visualization."""
        panels = []
        
        # RGB with detections
        if rgb is not None and self.detector:
            rgb_vis = self.detector.visualize(rgb, depth)
            rgb_vis = cv2.resize(rgb_vis, (320, 240))
            panels.append(rgb_vis)
        
        # Map
        if self.current_floor in self.floors:
            map_vis = self._draw_map()
            panels.append(map_vis)
        
        if panels:
            return np.hstack(panels)
        
        return np.zeros((240, 320, 3), dtype=np.uint8)
    
    def _draw_map(self) -> np.ndarray:
        """Draw navigation map."""
        floor = self.floors[self.current_floor]
        nav = floor.navigable_map
        exp = floor.explored_map
        
        vis = np.zeros((*nav.shape, 3), dtype=np.uint8)
        vis[exp < 0.5] = [30, 30, 30]
        vis[(exp > 0.5) & (nav < 0.5)] = [80, 80, 80]
        vis[nav > 0.5] = [150, 150, 150]
        
        # Draw path
        if self.path:
            for i in range(len(self.path) - 1):
                color = (255, 200, 0) if i < self.current_waypoint else (0, 200, 255)
                cv2.line(vis, self.path[i], self.path[i+1], color, 1)
        
        # Draw goal
        if self.goal_map:
            cv2.circle(vis, self.goal_map, 6, (0, 0, 255), 2)
        
        # Draw landmarks
        for lms in self.landmarks.values():
            for lm in lms:
                mx, my = self.world_to_map(lm.position[0], lm.position[2])
                if 0 <= mx < vis.shape[1] and 0 <= my < vis.shape[0]:
                    cv2.circle(vis, (mx, my), 3, (0, 255, 0), -1)
        
        # Draw agent
        ax, ay = self.world_to_map(self.position[0], self.position[2])
        if 0 <= ax < vis.shape[1] and 0 <= ay < vis.shape[0]:
            cv2.circle(vis, (ax, ay), 5, (255, 0, 0), -1)
            dx = -int(10 * np.sin(self.yaw))
            dy = -int(10 * np.cos(self.yaw))
            cv2.arrowedLine(vis, (ax, ay), (ax+dx, ay+dy), (255, 0, 0), 2)
        
        # Resize
        scale = 240 / max(vis.shape)
        vis = cv2.resize(vis, None, fx=scale, fy=scale)
        
        # Pad to 320 width
        if vis.shape[1] < 320:
            pad = np.zeros((vis.shape[0], 320 - vis.shape[1], 3), dtype=np.uint8)
            vis = np.hstack([vis, pad])
        
        return vis
