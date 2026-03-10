#!/usr/bin/env python3
"""
SLAM Test with User Control & Semantic Localization
====================================================

Features:
1. Keyboard control (WASD/arrows)
2. YOLO object detection for landmark-based pose correction
3. VLM (Qwen) for natural language navigation commands
4. Scene graph matching for relocalization
5. GT comparison for validation

Usage:
    python test_slam_interactive.py --scene /path/to/scene.glb

Controls:
    W/↑ = Forward
    A/← = Turn left  
    D/→ = Turn right
    R = Reset to GT
    C = Correct pose to GT
    L = Toggle YOLO landmarks
    V = Voice/text command (VLM)
    M = Show semantic map
    Q = Quit
"""

import argparse
import numpy as np
import cv2
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Landmark:
    """Detected landmark for localization."""
    class_name: str
    position: np.ndarray  # [x, y, z] world position
    confidence: float
    last_seen: int  # frame number
    observations: int  # how many times seen


@dataclass 
class SemanticMap:
    """Simple semantic map for relocalization."""
    landmarks: Dict[str, List[Landmark]]  # class -> landmarks
    
    def add_landmark(self, class_name: str, position: np.ndarray, 
                     confidence: float, frame: int):
        """Add or update landmark."""
        if class_name not in self.landmarks:
            self.landmarks[class_name] = []
        
        # Check for existing nearby landmark
        for lm in self.landmarks[class_name]:
            dist = np.linalg.norm(lm.position - position)
            if dist < 1.0:  # Within 1m = same object
                # Update with running average
                alpha = 0.3
                lm.position = (1 - alpha) * lm.position + alpha * position
                lm.confidence = max(lm.confidence, confidence)
                lm.last_seen = frame
                lm.observations += 1
                return
        
        # New landmark
        self.landmarks[class_name].append(Landmark(
            class_name=class_name,
            position=position.copy(),
            confidence=confidence,
            last_seen=frame,
            observations=1
        ))
    
    def get_all_landmarks(self) -> List[Landmark]:
        """Get all landmarks."""
        all_lm = []
        for lms in self.landmarks.values():
            all_lm.extend(lms)
        return all_lm
    
    def find_nearest(self, class_name: str, position: np.ndarray) -> Optional[Landmark]:
        """Find nearest landmark of given class."""
        if class_name not in self.landmarks:
            return None
        
        nearest = None
        min_dist = float('inf')
        
        for lm in self.landmarks[class_name]:
            dist = np.linalg.norm(lm.position[[0, 2]] - position[[0, 2]])
            if dist < min_dist:
                min_dist = dist
                nearest = lm
        
        return nearest


# ============================================================
# YOLO Detector
# ============================================================

class YOLODetector:
    """YOLO object detector for landmark detection."""
    
    INDOOR_CLASSES = {
        'person', 'chair', 'couch', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'refrigerator', 'sink', 'potted plant', 'clock',
        'book', 'vase', 'oven', 'microwave'
    }
    
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.65):
        self.model = None
        self.conf = conf
        self.class_names = {}
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.class_names = self.model.names
            print(f"YOLO loaded: {model_name}")
        except Exception as e:
            print(f"YOLO not available: {e}")
    
    def detect(self, rgb: np.ndarray, depth: np.ndarray, 
               agent_pos: np.ndarray, agent_yaw: float,
               camera_K: np.ndarray) -> List[Tuple[str, np.ndarray, float]]:
        """
        Detect objects and compute world positions.
        
        Returns:
            List of (class_name, world_position, confidence)
        """
        if self.model is None:
            return []
        
        results = self.model(rgb, verbose=False, conf=self.conf)
        
        if not results or len(results) == 0:
            return []
        
        detections = []
        result = results[0]
        
        if result.boxes is None:
            return []
        
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            class_name = self.class_names.get(cls_id, "unknown")
            
            # Filter to indoor classes
            if class_name.lower() not in self.INDOOR_CLASSES:
                continue
            
            # Get bounding box center
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Get depth at center
            margin = 10
            region = depth[max(0, cy-margin):min(depth.shape[0], cy+margin),
                         max(0, cx-margin):min(depth.shape[1], cx+margin)]
            valid = region[(region > 0.1) & (region < 8.0)]
            
            if len(valid) < 10:
                continue
            
            obj_depth = np.median(valid)
            
            # Backproject to camera frame
            fx, fy = camera_K[0, 0], camera_K[1, 1]
            cx_cam, cy_cam = camera_K[0, 2], camera_K[1, 2]
            
            x_cam = (cx - cx_cam) * obj_depth / fx
            z_cam = obj_depth
            
            # Transform to world
            cos_yaw = np.cos(agent_yaw)
            sin_yaw = np.sin(agent_yaw)
            
            # Camera to world (camera Z forward -> world -Z at yaw=0)
            x_world = x_cam * cos_yaw + z_cam * sin_yaw + agent_pos[0]
            z_world = -x_cam * sin_yaw + z_cam * cos_yaw + agent_pos[2]
            
            # Wait, this is wrong. Let me fix:
            # At yaw=0, camera forward is -Z in world
            # Camera frame: X-right, Y-down, Z-forward
            
            # Body frame (camera Z -> -Z in body at yaw=0)
            x_body = x_cam
            z_body = -z_cam
            
            # Rotate by yaw
            x_world = x_body * cos_yaw - z_body * sin_yaw + agent_pos[0]
            z_world = x_body * sin_yaw + z_body * cos_yaw + agent_pos[2]
            y_world = agent_pos[1]  # Same height
            
            world_pos = np.array([x_world, y_world, z_world])
            detections.append((class_name, world_pos, conf))
        
        return detections
    
    def visualize(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Draw detections on image."""
        if self.model is None:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        results = self.model(rgb, verbose=False, conf=self.conf)
        
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                class_name = self.class_names.get(cls_id, "?")
                
                if class_name.lower() not in self.INDOOR_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Get depth
                if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    d = depth[cy, cx]
                    label = f"{class_name} {d:.1f}m"
                else:
                    label = class_name
                
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis


# ============================================================
# VLM Navigator (Qwen)
# ============================================================

class VLMNavigator:
    """VLM-based navigation using Qwen."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.available = False
        
        self._load_model()
    
    def _load_model(self):
        """Load Qwen VLM."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            
            print(f"Loading VLM: {model_name}...")
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            self.available = True
            print("VLM loaded successfully")
            
        except Exception as e:
            print(f"VLM not available: {e}")
            self.available = False
    
    def get_navigation_target(self, rgb: np.ndarray, command: str, 
                              available_objects: List[str]) -> Optional[str]:
        """
        Parse natural language command to navigation target.
        
        Args:
            rgb: Current view
            command: User command like "go to the bathroom" or "find a chair"
            available_objects: List of detected object classes
            
        Returns:
            Target object class or None
        """
        if not self.available:
            # Fallback: simple keyword matching
            return self._keyword_match(command, available_objects)
        
        try:
            from PIL import Image
            import torch
            
            # Convert to PIL
            pil_image = Image.fromarray(rgb)
            
            # Create prompt
            prompt = f"""You are a robot navigation assistant. 
The user wants to: "{command}"

Available objects that have been detected in the environment:
{', '.join(available_objects) if available_objects else 'None yet'}

Based on the user's request, which object should the robot navigate to?
Reply with ONLY the object name (one of the available objects), or "explore" if the target isn't found yet.
If the request is about a room (like bathroom, kitchen), suggest an object typically found there."""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract target from response
            response = response.split("assistant")[-1].strip().lower()
            
            # Match to available objects
            for obj in available_objects:
                if obj.lower() in response:
                    return obj
            
            if "explore" in response:
                return "explore"
            
            return self._keyword_match(command, available_objects)
            
        except Exception as e:
            print(f"VLM error: {e}")
            return self._keyword_match(command, available_objects)
    
    def _keyword_match(self, command: str, available: List[str]) -> Optional[str]:
        """Fallback keyword matching."""
        command = command.lower()
        
        # Room -> object mapping
        room_objects = {
            'bathroom': ['toilet', 'sink'],
            'restroom': ['toilet', 'sink'],
            'kitchen': ['refrigerator', 'oven', 'microwave', 'sink'],
            'bedroom': ['bed'],
            'living': ['couch', 'tv', 'chair'],
            'office': ['chair', 'laptop', 'desk'],
        }
        
        # Check room keywords
        for room, objects in room_objects.items():
            if room in command:
                for obj in objects:
                    if obj in available:
                        return obj
        
        # Direct object match
        for obj in available:
            if obj.lower() in command:
                return obj
        
        return None


# ============================================================
# Landmark-based Pose Correction
# ============================================================

class LandmarkCorrector:
    """Correct pose using detected landmarks vs semantic map."""
    
    def __init__(self, semantic_map: SemanticMap):
        self.semantic_map = semantic_map
        self.min_landmarks = 2  # Need at least 2 landmarks for correction
    
    def compute_correction(
        self,
        detections: List[Tuple[str, np.ndarray, float]],
        current_pos: np.ndarray,
        current_yaw: float
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Compute pose correction from landmark observations.
        
        Args:
            detections: List of (class, observed_world_pos, confidence)
            current_pos: Current estimated position
            current_yaw: Current estimated yaw
            
        Returns:
            (position_correction, yaw_correction) or None
        """
        if len(detections) < self.min_landmarks:
            return None
        
        # Match detections to map landmarks
        matches = []
        
        for class_name, obs_pos, conf in detections:
            map_lm = self.semantic_map.find_nearest(class_name, current_pos)
            
            if map_lm is not None and map_lm.observations >= 3:
                # Distance between observed and expected
                expected_pos = map_lm.position
                matches.append((obs_pos, expected_pos, conf))
        
        if len(matches) < self.min_landmarks:
            return None
        
        # Compute average correction
        corrections = []
        weights = []
        
        for obs_pos, exp_pos, conf in matches:
            # The difference tells us how much our position estimate is off
            # If we see object at obs_pos but map says it's at exp_pos,
            # then we are offset by (exp_pos - obs_pos)
            correction = exp_pos - obs_pos
            corrections.append(correction[[0, 2]])  # X-Z only
            weights.append(conf)
        
        corrections = np.array(corrections)
        weights = np.array(weights)
        weights /= weights.sum()
        
        avg_correction = np.average(corrections, axis=0, weights=weights)
        
        # Only apply if correction is reasonable
        if np.linalg.norm(avg_correction) > 2.0:
            return None  # Too large, probably wrong
        
        pos_correction = np.array([avg_correction[0], 0, avg_correction[1]])
        
        return pos_correction, 0.0  # No yaw correction for now


# ============================================================
# Main SLAM System
# ============================================================

class InteractiveSLAM:
    """Interactive SLAM with user control and semantic localization."""
    
    FORWARD = 0.25
    TURN = np.radians(10.0)
    
    def __init__(self, use_yolo: bool = True, use_vlm: bool = False):
        # Pose
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.prev_y = 0.0
        self.current_floor = 0
        
        # Semantic map
        self.semantic_map = SemanticMap(landmarks=defaultdict(list))
        
        # Detectors
        self.yolo = YOLODetector() if use_yolo else None
        self.vlm = VLMNavigator() if use_vlm else None
        
        # Landmark corrector
        self.landmark_corrector = LandmarkCorrector(self.semantic_map)
        
        # Camera
        self.camera_K = self._default_camera_K()
        
        # Stats
        self.frame_count = 0
        self.total_distance = 0.0
        self.gt_errors = []
        self.corrections_applied = 0
        
        # Navigation
        self.nav_target: Optional[str] = None
        self.nav_target_pos: Optional[np.ndarray] = None
    
    def _default_camera_K(self) -> np.ndarray:
        """Default camera intrinsics."""
        width, height = 640, 480
        hfov = np.radians(90)
        fx = width / (2.0 * np.tan(hfov / 2))
        return np.array([
            [fx, 0, width/2],
            [0, fx, height/2],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def initialize(self, position: np.ndarray, yaw: float):
        """Initialize with known pose."""
        self.position = position.copy()
        self.yaw = yaw
        self.prev_y = position[1]
        self.current_floor = 0
        self.frame_count = 0
        self.total_distance = 0.0
        self.gt_errors.clear()
        self.semantic_map.landmarks.clear()
        self.corrections_applied = 0
        print(f"SLAM initialized: ({position[0]:.2f}, {position[2]:.2f})")
    
    def update(
        self,
        action: Optional[str],
        rgb: np.ndarray,
        depth: np.ndarray,
        gt_pose: Tuple[np.ndarray, float],
        use_landmarks: bool = True
    ) -> Dict:
        """
        Update SLAM after action execution.
        
        Returns dict with all state info.
        """
        self.frame_count += 1
        gt_pos, gt_yaw = gt_pose
        
        # 1. Dead reckoning
        self._apply_action(action)
        
        # 2. Floor change detection
        self._check_floor_change(gt_pos[1])
        
        # 3. YOLO detection
        detections = []
        if self.yolo is not None:
            detections = self.yolo.detect(
                rgb, depth, self.position, self.yaw, self.camera_K
            )
            
            # Add to semantic map
            for class_name, world_pos, conf in detections:
                self.semantic_map.add_landmark(
                    class_name, world_pos, conf, self.frame_count
                )
        
        # 4. Landmark-based correction
        if use_landmarks and len(detections) >= 2:
            correction = self.landmark_corrector.compute_correction(
                detections, self.position, self.yaw
            )
            
            if correction is not None:
                pos_corr, yaw_corr = correction
                
                # Apply partial correction
                alpha = 0.3
                self.position += alpha * pos_corr
                self.yaw += alpha * yaw_corr
                self.corrections_applied += 1
        
        # 5. Compute error
        error_2d = np.sqrt(
            (self.position[0] - gt_pos[0])**2 +
            (self.position[2] - gt_pos[2])**2
        )
        self.gt_errors.append(error_2d)
        
        # 6. Check navigation target
        nav_info = self._check_nav_target()
        
        return {
            'position': self.position.copy(),
            'yaw': self.yaw,
            'gt_position': gt_pos,
            'gt_yaw': gt_yaw,
            'error_2d': error_2d,
            'floor': self.current_floor,
            'detections': detections,
            'landmarks_total': len(self.semantic_map.get_all_landmarks()),
            'corrections': self.corrections_applied,
            'nav_target': self.nav_target,
            'nav_info': nav_info,
        }
    
    def _apply_action(self, action: Optional[str]):
        """Apply dead reckoning."""
        if action is None:
            return
        
        if action == 'move_forward':
            self.position[0] -= np.sin(self.yaw) * self.FORWARD
            self.position[2] -= np.cos(self.yaw) * self.FORWARD
            self.total_distance += self.FORWARD
        elif action == 'turn_left':
            self.yaw += self.TURN
        elif action == 'turn_right':
            self.yaw -= self.TURN
        
        while self.yaw > np.pi: self.yaw -= 2*np.pi
        while self.yaw < -np.pi: self.yaw += 2*np.pi
    
    def _check_floor_change(self, current_y: float):
        """Check for floor transition."""
        if abs(current_y - self.prev_y) > 0.3:
            self.current_floor += 1 if current_y > self.prev_y else -1
            self.position[1] = current_y
            print(f"Floor change -> {self.current_floor}")
        self.prev_y = current_y
    
    def _check_nav_target(self) -> Optional[str]:
        """Check if near navigation target."""
        if self.nav_target is None:
            return None
        
        target_lm = self.semantic_map.find_nearest(self.nav_target, self.position)
        
        if target_lm is None:
            return f"Searching for {self.nav_target}..."
        
        dist = np.linalg.norm(target_lm.position[[0, 2]] - self.position[[0, 2]])
        
        if dist < 1.0:
            result = f"REACHED {self.nav_target}!"
            self.nav_target = None
            return result
        
        return f"{self.nav_target}: {dist:.1f}m away"
    
    def set_nav_target(self, target: str):
        """Set navigation target."""
        self.nav_target = target.lower()
        print(f"Navigation target: {target}")
    
    def get_available_objects(self) -> List[str]:
        """Get list of detected object classes."""
        return list(self.semantic_map.landmarks.keys())
    
    def correct_to_gt(self, gt_pos: np.ndarray, gt_yaw: float):
        """Correct pose to ground truth."""
        self.position = gt_pos.copy()
        self.yaw = gt_yaw
        self.prev_y = gt_pos[1]
        print("Pose corrected to GT")
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        stats = {
            'frames': self.frame_count,
            'distance': self.total_distance,
            'landmarks': len(self.semantic_map.get_all_landmarks()),
            'corrections': self.corrections_applied,
        }
        
        if self.gt_errors:
            stats['error_mean'] = np.mean(self.gt_errors)
            stats['error_max'] = np.max(self.gt_errors)
            stats['error_current'] = self.gt_errors[-1]
            stats['drift'] = self.gt_errors[-1] / max(self.total_distance, 0.01) * 100
        
        return stats


# ============================================================
# Habitat Setup
# ============================================================

def setup_habitat(scene_path: str):
    """Setup Habitat simulator."""
    import habitat_sim
    
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [480, 640]
    rgb_spec.position = [0.0, 1.25, 0.0]
    rgb_spec.hfov = 90
    
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [480, 640]
    depth_spec.position = [0.0, 1.25, 0.0]
    depth_spec.hfov = 90
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)),
    }
    
    return habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))


def get_gt_pose(sim):
    """Get ground truth from Habitat."""
    state = sim.get_agent(0).get_state()
    pos = np.array(state.position)
    
    x, y, z, w = state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return pos, yaw


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Interactive SLAM Test')
    parser.add_argument('--scene', required=True, help='Path to scene')
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO')
    parser.add_argument('--vlm', action='store_true', help='Enable VLM (Qwen)')
    args = parser.parse_args()
    
    print("="*60)
    print("Interactive SLAM Test")
    print("="*60)
    print("Controls:")
    print("  W/↑ = Forward    A/← = Left    D/→ = Right")
    print("  R = Reset        C = Correct   L = Toggle YOLO vis")
    print("  V = VLM command  M = Map       Q = Quit")
    print("="*60)
    
    # Setup
    sim = setup_habitat(args.scene)
    
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
    
    gt_pos, gt_yaw = get_gt_pose(sim)
    
    # SLAM
    slam = InteractiveSLAM(
        use_yolo=not args.no_yolo,
        use_vlm=args.vlm
    )
    slam.initialize(gt_pos, gt_yaw)
    
    # Window
    window = 'Interactive SLAM'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1000, 600)
    
    show_yolo = True
    last_action = None
    
    while True:
        # Observations
        obs = sim.get_sensor_observations()
        rgb = obs['rgb'][:, :, :3]
        depth = obs['depth']
        
        gt_pos, gt_yaw = get_gt_pose(sim)
        
        # Update SLAM
        state = slam.update(last_action, rgb, depth, (gt_pos, gt_yaw), use_landmarks=True)
        last_action = None
        
        # Visualization
        if show_yolo and slam.yolo is not None:
            rgb_vis = slam.yolo.visualize(rgb, depth)
        else:
            rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        rgb_vis = cv2.resize(rgb_vis, (400, 300))
        
        # Depth vis
        depth_vis = cv2.applyColorMap(
            (np.clip(depth / 5.0, 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS
        )
        depth_vis = cv2.resize(depth_vis, (400, 300))
        
        # Info panel
        info = np.zeros((300, 400, 3), dtype=np.uint8)
        info[:] = [30, 30, 30]
        
        stats = slam.get_stats()
        
        err_color = (0, 255, 0) if state['error_2d'] < 0.1 else \
                   (0, 255, 255) if state['error_2d'] < 0.5 else (0, 0, 255)
        
        lines = [
            f"SLAM: ({state['position'][0]:+.2f}, {state['position'][2]:+.2f})",
            f"GT:   ({gt_pos[0]:+.2f}, {gt_pos[2]:+.2f})",
            f"Error: {state['error_2d']:.4f}m",
            f"Floor: {state['floor']}",
            "",
            f"Distance: {stats['distance']:.1f}m",
            f"Drift: {stats.get('drift', 0):.2f}%",
            f"Landmarks: {state['landmarks_total']}",
            f"Corrections: {state['corrections']}",
            "",
            f"Objects: {', '.join(slam.get_available_objects()[:5])}",
        ]
        
        if state['nav_info']:
            lines.append(f"Nav: {state['nav_info']}")
        
        for i, line in enumerate(lines):
            color = err_color if 'Error' in line else (200, 200, 200)
            cv2.putText(info, line, (10, 22 + i*22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Error bar
        bar_len = min(int(state['error_2d'] * 300), 380)
        cv2.rectangle(info, (10, 280), (10 + bar_len, 295), err_color, -1)
        
        # Combine
        top = np.hstack([rgb_vis, depth_vis])
        
        # Simple map view
        map_vis = np.zeros((300, 400, 3), dtype=np.uint8)
        map_vis[:] = [40, 40, 40]
        
        # Draw landmarks
        scale = 20  # pixels per meter
        cx, cy = 100, 150  # center
        
        for lm in slam.semantic_map.get_all_landmarks():
            mx = int(cx + (lm.position[0] - state['position'][0]) * scale)
            my = int(cy + (lm.position[2] - state['position'][2]) * scale)
            if 0 <= mx < 200 and 0 <= my < 300:
                cv2.circle(map_vis, (mx, my), 4, (0, 255, 0), -1)
        
        # Draw agent
        cv2.circle(map_vis, (cx, cy), 6, (255, 0, 0), -1)
        dx = int(15 * np.sin(state['yaw']))
        dy = int(15 * np.cos(state['yaw']))
        cv2.arrowedLine(map_vis, (cx, cy), (cx - dx, cy - dy), (255, 0, 0), 2)
        
        # Nav target
        if slam.nav_target:
            target_lm = slam.semantic_map.find_nearest(slam.nav_target, state['position'])
            if target_lm is not None:
                tx = int(cx + (target_lm.position[0] - state['position'][0]) * scale)
                ty = int(cy + (target_lm.position[2] - state['position'][2]) * scale)
                if 0 <= tx < 200 and 0 <= ty < 300:
                    cv2.circle(map_vis, (tx, ty), 8, (0, 0, 255), 2)
        
        bottom = np.hstack([info, map_vis])
        vis = np.vstack([top, bottom])
        
        cv2.imshow(window, vis)
        key = cv2.waitKey(0) & 0xFF
        
        # Handle input
        action = None
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('w') or key == 82:  # up
            action = 'move_forward'
        elif key == ord('a') or key == 81:  # left
            action = 'turn_left'
        elif key == ord('d') or key == 83:  # right
            action = 'turn_right'
        elif key == ord('r'):
            slam.initialize(gt_pos, gt_yaw)
        elif key == ord('c'):
            slam.correct_to_gt(gt_pos, gt_yaw)
        elif key == ord('l'):
            show_yolo = not show_yolo
            print(f"YOLO visualization: {'ON' if show_yolo else 'OFF'}")
        elif key == ord('v'):
            # VLM command
            print("\nEnter navigation command (or press Enter to cancel):")
            command = input("> ").strip()
            if command:
                available = slam.get_available_objects()
                if slam.vlm is not None and slam.vlm.available:
                    target = slam.vlm.get_navigation_target(rgb, command, available)
                else:
                    # Fallback
                    target = slam.vlm._keyword_match(command, available) if slam.vlm else None
                
                if target:
                    slam.set_nav_target(target)
                else:
                    print("Could not determine target. Try exploring more.")
        elif key == ord('m'):
            # Print map info
            print("\n=== Semantic Map ===")
            for cls, lms in slam.semantic_map.landmarks.items():
                print(f"  {cls}: {len(lms)} instances")
            print()
        
        if action:
            sim.step(action)
            last_action = action
        
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()
    sim.close()
    
    # Final stats
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)
    stats = slam.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
