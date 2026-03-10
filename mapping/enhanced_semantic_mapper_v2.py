"""
Enhanced Semantic Mapper V2 - Fixed Issues
==========================================

Fixes:
1. Duplicate detection - Stricter merging with IoU + distance + temporal
2. Semantic occupancy clipping - Objects only where depth shows obstacles
3. Floor detection - Relative height change detection
4. Real-time detection visualization

"""

import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict

from mapping.fixed_occupancy_map import FixedOccupancyMap, MapConfig


try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Color scheme by category
CATEGORY_COLORS = {
    'chair': (65, 105, 225),
    'couch': (30, 144, 255),
    'bench': (70, 130, 180),
    'dining table': (210, 105, 30),
    'desk': (205, 133, 63),
    'bed': (148, 0, 211),
    'toilet': (0, 255, 255),
    'sink': (64, 224, 208),
    'refrigerator': (192, 192, 192),
    'oven': (169, 169, 169),
    'microwave': (211, 211, 211),
    'tv': (0, 255, 0),
    'laptop': (50, 205, 50),
    'potted plant': (0, 100, 0),
    'bottle': (255, 255, 0),
    'cup': (255, 215, 0),
    'book': (220, 20, 60),
    'clock': (255, 0, 0),
    'backpack': (255, 0, 255),
    'stairs': (255, 140, 0),
    'person': (255, 192, 203),
}

# Reduced footprints for accuracy
OBJECT_FOOTPRINTS = {
    'chair': (0.45, 0.45),
    'couch': (1.8, 0.8),
    'dining table': (1.2, 0.75),
    'bed': (1.8, 1.4),
    'toilet': (0.4, 0.6),
    'sink': (0.5, 0.4),
    'refrigerator': (0.7, 0.6),
    'tv': (1.0, 0.08),
    'laptop': (0.35, 0.25),
    'potted plant': (0.25, 0.25),
    'stairs': (0.9, 1.5),
}


@dataclass
class Detection:
    """Single detection."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    mask: Optional[np.ndarray]
    position_3d: Optional[np.ndarray]
    depth_at_object: float = 0.0


@dataclass
class TrackedObject:
    """Tracked object with strict merging."""
    object_id: int
    class_name: str
    positions: List[np.ndarray] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_seen_frame: int = 0
    floor_level: int = 0
    creation_frame: int = 0
    
    @property
    def observations(self) -> int:
        return len(self.positions)
    
    @property
    def mean_position(self) -> np.ndarray:
        if not self.positions:
            return np.zeros(3)
        return np.median(self.positions, axis=0)
    
    @property
    def position_std(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        return np.std([np.linalg.norm(p - self.mean_position) for p in self.positions])
    
    @property
    def mean_confidence(self) -> float:
        return np.mean(self.confidences) if self.confidences else 0.0
    
    @property 
    def footprint(self) -> Tuple[float, float]:
        return OBJECT_FOOTPRINTS.get(self.class_name.lower(), (0.3, 0.3))
    
    def add_observation(self, pos, conf, bbox, frame):
        self.positions.append(pos.copy())
        self.confidences.append(conf)
        self.last_bbox = bbox
        self.last_seen_frame = frame
        # Keep recent only
        if len(self.positions) > 20:
            self.positions = self.positions[-20:]
            self.confidences = self.confidences[-20:]


class EnhancedSemanticMapperV2:
    """Fixed semantic mapper."""
    
    def __init__(
        self,
        map_config: MapConfig = None,
        model_size: str = 'n',
        confidence_threshold: float = 0.4,
        merge_distance: float = 1.0,
        merge_iou_threshold: float = 0.3,
        min_observations: int = 3,
        stale_threshold: int = 100,
        floor_change_threshold: float = 1.5,
        yolo_model = None,
    ):
        self.map_config = map_config or MapConfig()
        self.resolution = self.map_config.resolution_cm
        
        self.conf_threshold = confidence_threshold
        self.merge_distance = merge_distance
        self.merge_iou_threshold = merge_iou_threshold
        self.min_observations = min_observations
        self.stale_threshold = stale_threshold
        self.floor_change_threshold = floor_change_threshold
        
        # YOLO - use provided model or load new one
        if yolo_model is not None:
            self.model = yolo_model
        elif YOLO_AVAILABLE:
            try:
                self.model = YOLO(f'yolov8{model_size}-seg.pt')
                print(f"Loaded YOLOv8{model_size}-seg")
            except Exception as e:
                print(f"YOLO failed: {e}")
                self.model = None
        else:
            self.model = None
        
        self.occupancy_map = FixedOccupancyMap(self.map_config)
        
        # Multi-floor
        self.current_floor = 0
        self.floor_base_heights = {0: None}
        self.last_agent_height = None
        
        # Tracking
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 0
        self.frame_count = 0
        
        # Categories
        self.category_to_idx = {cat: i for i, cat in enumerate(CATEGORY_COLORS.keys())}
        self.num_categories = len(CATEGORY_COLORS)
        
        # Camera
        self.img_width = 640
        self.img_height = 480
        self.fx = self.img_width / 2.0
        self.fy = self.fx
        self.cx = self.img_width / 2
        self.cy = self.img_height / 2
        
        # Multi-floor semantic maps - separate map per floor
        map_size = self.occupancy_map.map_size
        self.floor_semantic_maps: Dict[int, np.ndarray] = {}
        self.floor_obstacle_maps: Dict[int, np.ndarray] = {}
        self.floor_explored_maps: Dict[int, np.ndarray] = {}
        self._init_floor_maps(0)
        
        # For visualization
        self.last_detections: List[Detection] = []
        self.last_rgb = None
        
        self.stairs_detected = []
    
    def _init_floor_maps(self, floor_id: int):
        """Initialize maps for a new floor."""
        map_size = self.occupancy_map.map_size
        if floor_id not in self.floor_semantic_maps:
            self.floor_semantic_maps[floor_id] = np.zeros((map_size, map_size, self.num_categories), dtype=np.float32)
            self.floor_obstacle_maps[floor_id] = np.zeros((map_size, map_size), dtype=np.float32)
            self.floor_explored_maps[floor_id] = np.zeros((map_size, map_size), dtype=np.float32)
            print(f"Initialized maps for floor {floor_id}")
    
    def reset(self):
        self.occupancy_map.reset()
        self.tracked_objects.clear()
        self.next_object_id = 0
        self.frame_count = 0
        self.current_floor = 0
        self.floor_base_heights = {0: None}
        self.last_detections = []
        self.stairs_detected = []
        
        # Reset all floor maps
        self.floor_semantic_maps.clear()
        self.floor_obstacle_maps.clear()
        self.floor_explored_maps.clear()
        self._init_floor_maps(0)
    
    def _get_yaw(self, quat):
        x, y, z, w = quat
        return np.arctan2(2*(w*y + z*x), 1 - 2*(x*x + y*y))
    
    def _detect_floor_change(self, agent_height):
        """Detect floor based on RELATIVE height change."""
        if self.floor_base_heights[0] is None:
            self.floor_base_heights[0] = agent_height
            self.last_agent_height = agent_height
            return 0
        
        current_base = self.floor_base_heights.get(self.current_floor, agent_height)
        height_diff = agent_height - current_base
        
        new_floor = self.current_floor
        
        if height_diff > self.floor_change_threshold:
            # Going UP
            new_floor = self.current_floor + 1
            if new_floor not in self.floor_base_heights:
                self.floor_base_heights[new_floor] = agent_height
                print(f"FLOOR UP: {self.current_floor} -> {new_floor}")
        elif height_diff < -self.floor_change_threshold:
            # Going DOWN
            new_floor = self.current_floor - 1
            if new_floor not in self.floor_base_heights:
                self.floor_base_heights[new_floor] = agent_height
                print(f"FLOOR DOWN: {self.current_floor} -> {new_floor}")
        
        # If floor changed, save current floor's occupancy data and init new floor
        if new_floor != self.current_floor:
            # Save current floor's obstacle/explored maps before switching
            self.floor_obstacle_maps[self.current_floor] = self.occupancy_map.get_obstacle_map().copy()
            self.floor_explored_maps[self.current_floor] = self.occupancy_map.get_explored_map().copy()
            
            # Init new floor if needed
            if new_floor not in self.floor_semantic_maps:
                self._init_floor_maps(new_floor)
            
            # Reset occupancy map for new floor (fresh start)
            # But restore if we've been to this floor before
            if self.floor_obstacle_maps[new_floor].sum() > 0:
                # Restore previous data for this floor
                self.occupancy_map.obstacle_map = self.floor_obstacle_maps[new_floor].copy()
                self.occupancy_map.explored_map = self.floor_explored_maps[new_floor].copy()
            else:
                # Fresh floor - reset occupancy tracking
                self.occupancy_map.obstacle_map.fill(0)
                self.occupancy_map.explored_map.fill(0)
        
        self.last_agent_height = agent_height
        return new_floor
    
    def _bbox_iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2-x1) * (y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)
    
    def _match_detection(self, det):
        """Strict matching — prevents cross-class confusion."""
        if det.position_3d is None:
            return -1
        
        best_match, best_score = -1, 0.0
        det_class = det.class_name.lower()
        
        # Classes that YOLO confuses with each other
        CONFUSABLE = {
            frozenset({'couch', 'bed'}),
            frozenset({'dining table', 'bed'}),
            frozenset({'bench', 'couch'}),
            frozenset({'bench', 'bed'}),
        }
        
        for oid, obj in self.tracked_objects.items():
            obj_class = obj.class_name.lower()
            
            if obj_class != det_class:
                pair = frozenset({det_class, obj_class})
                if pair in CONFUSABLE:
                    continue  # Don't let couch match with bed etc.
                continue
            
            if obj.floor_level != self.current_floor:
                continue
            
            frames_since = self.frame_count - obj.last_seen_frame
            if frames_since > self.stale_threshold:
                continue
            
            dist = np.linalg.norm(det.position_3d - obj.mean_position)
            if dist > self.merge_distance:
                continue
            
            iou = self._bbox_iou(det.bbox, obj.last_bbox) if frames_since < 10 else 0
            score = (1 - dist/self.merge_distance) * 0.6 + iou * 0.4
            
            if score > best_score:
                best_score = score
                best_match = oid
        
        return best_match
    
    def _compute_3d_position(self, cx, cy, depth, mask, agent_pos, yaw):
        if mask is not None and mask.sum() > 10:
            valid = depth[mask > 0]
            valid = valid[(valid > 0.1) & (valid < 8)]
        else:
            m = 15
            region = depth[max(0,cy-m):cy+m, max(0,cx-m):cx+m]
            valid = region[(region > 0.1) & (region < 8)]
        
        if len(valid) == 0:
            return None, 0.0
        
        d = np.median(valid)
        z_cam = d
        x_cam = (cx - self.cx) * d / self.fx
        y_cam = (cy - self.cy) * d / self.fy
        
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        wx = agent_pos[0] + x_cam * cos_y - z_cam * sin_y
        wz = agent_pos[2] - x_cam * sin_y - z_cam * cos_y
        wy = agent_pos[1] - y_cam
        
        return np.array([wx, wy, wz]), d
    
    def _update_semantic_map(self):
        """Update semantic map for current floor - CLIP to obstacle map!"""
        # Ensure floor maps exist
        if self.current_floor not in self.floor_semantic_maps:
            self._init_floor_maps(self.current_floor)
        
        map_size = self.occupancy_map.map_size
        semantic_map = self.floor_semantic_maps[self.current_floor]
        
        # Handle occupancy map resize: if semantic map is smaller, grow it
        if semantic_map.shape[0] < map_size or semantic_map.shape[1] < map_size:
            new_sem = np.zeros((map_size, map_size, self.num_categories), dtype=np.float32)
            old_h, old_w = semantic_map.shape[:2]
            new_sem[:old_h, :old_w, :] = semantic_map
            self.floor_semantic_maps[self.current_floor] = new_sem
            semantic_map = new_sem
            # Also resize floor obstacle/explored maps if they exist
            for maps_dict in [self.floor_obstacle_maps, self.floor_explored_maps]:
                if self.current_floor in maps_dict:
                    old_map = maps_dict[self.current_floor]
                    if old_map.shape[0] < map_size or old_map.shape[1] < map_size:
                        new_map = np.zeros((map_size, map_size), dtype=old_map.dtype)
                        oh, ow = old_map.shape[:2]
                        new_map[:oh, :ow] = old_map
                        maps_dict[self.current_floor] = new_map
        
        semantic_map.fill(0)
        
        obstacle_map = self.occupancy_map.get_obstacle_map()
        
        for obj in self.get_confirmed_objects().values():
            if obj.floor_level != self.current_floor:
                continue
            
            pos = obj.mean_position
            mx, my = self.occupancy_map._world_to_map(pos[0], pos[2])
            
            if not (0 <= mx < map_size and 0 <= my < map_size):
                continue
            
            cat_idx = self.category_to_idx.get(obj.class_name.lower())
            if cat_idx is None:
                continue
            
            w_m, d_m = obj.footprint
            w_px = max(int(w_m * 100 / self.resolution), 2)
            d_px = max(int(d_m * 100 / self.resolution), 2)
            
            hw, hd = w_px // 2, d_px // 2
            y1, y2 = max(0, my-hd), min(map_size, my+hd+1)
            x1, x2 = max(0, mx-hw), min(map_size, mx+hw+1)
            
            if y2 <= y1 or x2 <= x1:
                continue
            
            # CLIP: Only mark where obstacles exist
            obs_region = obstacle_map[y1:y2, x1:x2]
            if obs_region.size == 0:
                continue
            kernel = np.ones((3,3), np.uint8)
            obs_dilated = cv2.dilate((obs_region > 0).astype(np.uint8), kernel)
            
            footprint = np.ones((y2-y1, x2-x1), np.float32) * obj.mean_confidence
            # Safety: ensure shapes match after dilation
            fh, fw = footprint.shape
            oh, ow = obs_dilated.shape
            minh, minw = min(fh, oh), min(fw, ow)
            footprint[:minh, :minw] *= obs_dilated[:minh, :minw]
            
            sem_h, sem_w = semantic_map.shape[:2]
            sy2, sx2 = min(y2, sem_h), min(x2, sem_w)
            fy2, fx2 = sy2 - y1, sx2 - x1
            if sy2 > y1 and sx2 > x1 and fy2 > 0 and fx2 > 0:
                semantic_map[y1:sy2, x1:sx2, cat_idx] = np.maximum(
                    semantic_map[y1:sy2, x1:sx2, cat_idx], footprint[:fy2, :fx2]
                )
        
        # Save current floor's obstacle and explored maps
        self.floor_obstacle_maps[self.current_floor] = obstacle_map.copy()
        self.floor_explored_maps[self.current_floor] = self.occupancy_map.get_explored_map().copy()
    
    def _remove_duplicates(self):
        """Remove duplicates — same class close together, or cross-class conflicts."""
        to_remove = set()
        objs = list(self.tracked_objects.items())
        
        for i, (id1, o1) in enumerate(objs):
            if id1 in to_remove:
                continue
            for id2, o2 in objs[i+1:]:
                if id2 in to_remove:
                    continue
                if o1.floor_level != o2.floor_level:
                    continue
                
                dist = np.linalg.norm(o1.mean_position - o2.mean_position)
                
                # Same class, close → keep higher observations
                if o1.class_name == o2.class_name and dist < 0.8:
                    to_remove.add(id2 if o1.observations >= o2.observations else id1)
                    if id1 in to_remove:
                        break
                
                # Different class, very close → cross-class conflict
                # Keep the one with more observations
                elif o1.class_name != o2.class_name and dist < 0.5:
                    loser = id2 if o1.observations >= o2.observations else id1
                    to_remove.add(loser)
                    if id1 in to_remove:
                        break
        
        for oid in to_remove:
            del self.tracked_objects[oid]
        if to_remove:
            print(f"Removed {len(to_remove)} duplicates")
    
    def update(self, rgb, depth, agent_position, agent_rotation):
        self.frame_count += 1
        self.last_rgb = rgb.copy()
        
        # Floor detection
        prev_floor = self.current_floor
        self.current_floor = self._detect_floor_change(agent_position[1])
        
        # Base occupancy
        self.occupancy_map.update(depth, agent_position, agent_rotation)
        
        # Detection
        self.last_detections = []
        
        if self.model is not None:
            yaw = self._get_yaw(agent_rotation)
            results = self.model(rgb, conf=self.conf_threshold, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None:
                result = results[0]
                
                for i in range(len(result.boxes)):
                    cls_id = int(result.boxes.cls[i])
                    cls_name = self.model.names[cls_id].lower()
                    conf = float(result.boxes.conf[i])
                    
                    box = result.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    
                    mask = None
                    if result.masks is not None and i < len(result.masks):
                        mask = result.masks[i].data.cpu().numpy().squeeze()
                        if mask.shape != (rgb.shape[0], rgb.shape[1]):
                            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
                        mask = (mask > 0.5).astype(np.uint8)
                    
                    pos_3d, obj_depth = self._compute_3d_position(
                        cx, cy, depth, mask, agent_position, yaw
                    )
                    
                    det = Detection(cls_name, conf, (x1,y1,x2,y2), mask, pos_3d, obj_depth)
                    self.last_detections.append(det)
                    
                    if pos_3d is not None:
                        matched = self._match_detection(det)
                        
                        if matched >= 0:
                            self.tracked_objects[matched].add_observation(
                                pos_3d, conf, (x1,y1,x2,y2), self.frame_count
                            )
                        else:
                            new_id = self.next_object_id
                            self.next_object_id += 1
                            obj = TrackedObject(new_id, cls_name, floor_level=self.current_floor,
                                               creation_frame=self.frame_count)
                            obj.add_observation(pos_3d, conf, (x1,y1,x2,y2), self.frame_count)
                            self.tracked_objects[new_id] = obj
                        
                        if cls_name == 'stairs':
                            self._record_stairs(pos_3d)
        
        if self.frame_count % 50 == 0:
            self._remove_duplicates()
        
        self._update_semantic_map()
        
        return {
            'frame': self.frame_count,
            'current_floor': self.current_floor,
            'detections': len(self.last_detections),
            'confirmed_objects': len(self.get_confirmed_objects()),
            'tracked': len(self.tracked_objects),
            'explored_percent': self.occupancy_map.get_statistics().get('explored_percent', 0)
        }
    
    def _record_stairs(self, pos):
        for s in self.stairs_detected:
            if np.linalg.norm(np.array(s['position']) - pos) < 2.0:
                return
        self.stairs_detected.append({
            'position': pos.tolist(),
            'map_pos': self.occupancy_map._world_to_map(pos[0], pos[2]),
            'floor': self.current_floor
        })
    
    def get_confirmed_objects(self):
        return {oid: obj for oid, obj in self.tracked_objects.items()
                if obj.observations >= self.min_observations and obj.position_std < 1.0}
    
    def visualize_detections(self, rgb=None):
        """Show YOLO detections with masks and labels."""
        if rgb is None:
            rgb = self.last_rgb
        if rgb is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        vis = rgb.copy()
        
        for det in self.last_detections:
            x1, y1, x2, y2 = det.bbox
            color = CATEGORY_COLORS.get(det.class_name, (255, 255, 255))
            
            # Draw mask
            if det.mask is not None:
                mask_overlay = np.zeros_like(vis)
                mask_overlay[det.mask > 0] = color
                vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
            
            # Draw bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.depth_at_object > 0:
                label += f" {det.depth_at_object:.1f}m"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(vis, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        cv2.putText(vis, f"Detections: {len(self.last_detections)}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis
    
    def _resize_if_needed(self, m, target_size):
        if m.shape[0] != target_size:
            return cv2.resize(
                m,
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST
            )
        return m

    def visualize(self, floor_id=None, show_legend=True, show_trajectory=True):
        """
        Create map visualization for specified floor.
        
        Args:
            floor_id: Which floor (None = current)
            show_legend: Add legend panel
            show_trajectory: Show agent path and current position
        """
        if floor_id is None:
            floor_id = self.current_floor
        
        map_size = self.occupancy_map.map_size
        vis = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        vis[:] = [40, 40, 40]

        
        # Use floor-specific maps if viewing different floor
        if floor_id == self.current_floor:
            explored = self.occupancy_map.get_explored_map()
            obstacles = self.occupancy_map.get_obstacle_map()
        else:
            explored = self.floor_explored_maps.get(floor_id, np.zeros((map_size, map_size)))
            obstacles = self.floor_obstacle_maps.get(floor_id, np.zeros((map_size, map_size)))
        
        explored = self._resize_if_needed(explored, map_size)
        obstacles = self._resize_if_needed(obstacles, map_size)

        vis[explored > 0] = [180, 180, 180]
        vis[obstacles > 0] = [80, 80, 80]
        
        # Objects for this floor
        for obj in self.get_confirmed_objects().values():
            if obj.floor_level != floor_id:
                continue
            
            pos = obj.mean_position
            mx, my = self.occupancy_map._world_to_map(pos[0], pos[2])
            if not (0 <= mx < map_size and 0 <= my < map_size):
                continue
            
            color = CATEGORY_COLORS.get(obj.class_name, (255, 255, 255))
            w_m, d_m = obj.footprint
            w_px = max(int(w_m * 100 / self.resolution), 3)
            d_px = max(int(d_m * 100 / self.resolution), 3)
            
            hw, hd = w_px//2, d_px//2
            y1, y2 = max(0, my-hd), min(map_size, my+hd)
            x1, x2 = max(0, mx-hw), min(map_size, mx+hw)
            
            # Only where obstacles (but stairs get full footprint)
            if obj.class_name.lower() == 'stairs':
                # Stairs: draw full footprint, they're traversable
                vis[y1:y2, x1:x2] = color
            else:
                obs_region = obstacles[y1:y2, x1:x2]
                for dy in range(y2-y1):
                    for dx in range(x2-x1):
                        if obs_region[dy, dx] > 0 or (dy == hd and dx == hw):
                            vis[y1+dy, x1+dx] = color
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), 1)
        
        # Stairs markers (floor transition points)
        for s in self.stairs_detected:
            if s['floor'] == floor_id:
                sx, sy = s['map_pos']
                if 0 <= sx < map_size and 0 <= sy < map_size:
                    cv2.circle(vis, (sx, sy), 8, (255, 140, 0), 2)
                    cv2.putText(vis, "S", (sx-4, sy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 140, 0), 1)
        
        # Trajectory (optional)
        if show_trajectory:
            for pos in self.occupancy_map.trajectory[-500:]:
                mx, my = self.occupancy_map._world_to_map(pos[0], pos[2])
                if 0 <= mx < map_size and 0 <= my < map_size:
                    cv2.circle(vis, (mx, my), 1, (0, 200, 0), -1)
            
            # Agent position
            if self.occupancy_map.trajectory:
                pos = self.occupancy_map.trajectory[-1]
                mx, my = self.occupancy_map._world_to_map(pos[0], pos[2])
                if 0 <= mx < map_size and 0 <= my < map_size:
                    cv2.circle(vis, (mx, my), 5, (255, 100, 0), -1)
        
        if show_legend:
            vis = self._add_legend(vis, floor_id)
        
        return vis
    
    def _add_legend(self, vis, floor_id=None):
        if floor_id is None:
            floor_id = self.current_floor
        
        confirmed = self.get_confirmed_objects()
        present = set(o.class_name for o in confirmed.values() if o.floor_level == floor_id)
        
        lw, lh, pad = 150, 16, 6
        items = [("LEGEND", None, True), ("Unexplored", (40,40,40), False),
                 ("Explored", (180,180,180), False), ("Obstacle", (80,80,80), False),
                 ("Stairs", (255,140,0), False), ("", None, False), ("OBJECTS", None, True)]
        
        for cls in sorted(present):
            cnt = len([o for o in confirmed.values() if o.class_name == cls and o.floor_level == floor_id])
            items.append((f"{cls} ({cnt})", CATEGORY_COLORS.get(cls, (255,255,255)), False))
        
        leg_h = len(items) * lh + 2*pad + 30
        leg = np.zeros((max(vis.shape[0], leg_h), lw, 3), dtype=np.uint8)
        leg[:] = [25, 25, 25]
        
        y = pad
        for txt, col, hdr in items:
            if hdr:
                cv2.putText(leg, txt, (pad, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            elif col:
                cv2.rectangle(leg, (pad, y+2), (pad+10, y+12), col, -1)
                cv2.putText(leg, txt, (pad+14, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200,200,200), 1)
            y += lh
        
        cv2.putText(leg, f"Floor: {floor_id}", (pad, leg.shape[0]-pad-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)
        
        if leg.shape[0] < vis.shape[0]:
            pad_arr = np.zeros((vis.shape[0]-leg.shape[0], lw, 3), np.uint8)
            pad_arr[:] = [25,25,25]
            leg = np.vstack([leg, pad_arr])
        elif leg.shape[0] > vis.shape[0]:
            pad_arr = np.zeros((leg.shape[0]-vis.shape[0], vis.shape[1], 3), np.uint8)
            pad_arr[:] = [40,40,40]
            vis = np.vstack([vis, pad_arr])
        
        return np.hstack([vis, leg])
    
    def save(self, output_dir):
        """Save maps with proper structure - separate directories per floor."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current floor data first
        self.floor_obstacle_maps[self.current_floor] = self.occupancy_map.get_obstacle_map().copy()
        self.floor_explored_maps[self.current_floor] = self.occupancy_map.get_explored_map().copy()
        
        # Save each floor separately
        for floor_id in self.floor_semantic_maps.keys():
            floor_dir = output_dir / f'floor_{floor_id}'
            floor_dir.mkdir(exist_ok=True)
            
            # Get floor-specific data
            semantic_map = self.floor_semantic_maps[floor_id]
            obstacle_map = self.floor_obstacle_maps.get(floor_id, np.zeros_like(semantic_map[:,:,0]))
            explored_map = self.floor_explored_maps.get(floor_id, np.zeros_like(semantic_map[:,:,0]))
            
            # Create navigable map (obstacles with stairs marked as traversable)
            navigable_map = obstacle_map.copy()
            for stair in self.stairs_detected:
                if stair['floor'] == floor_id:
                    sx, sy = stair['map_pos']
                    # Clear obstacle in stair region (make traversable)
                    radius = 15  # Clear ~75cm radius around stairs
                    y1, y2 = max(0, sy-radius), min(navigable_map.shape[0], sy+radius)
                    x1, x2 = max(0, sx-radius), min(navigable_map.shape[1], sx+radius)
                    navigable_map[y1:y2, x1:x2] = 0  # Mark as free
            
            # PyTorch tensors
            torch.save(torch.from_numpy(obstacle_map).float(), floor_dir / 'occupancy_map.pt')
            torch.save(torch.from_numpy(navigable_map).float(), floor_dir / 'navigable_map.pt')  # For navigation!
            torch.save(torch.from_numpy(explored_map).float(), floor_dir / 'explored_map.pt')
            torch.save(torch.from_numpy(semantic_map).permute(2,0,1).float(), floor_dir / 'semantic_map.pt')
            
            # Combined tensor [obstacles, explored, navigable, semantics...]
            combined = torch.zeros(3 + self.num_categories, obstacle_map.shape[0], obstacle_map.shape[1])
            combined[0] = torch.from_numpy(obstacle_map).float()
            combined[1] = torch.from_numpy(explored_map).float()
            combined[2] = torch.from_numpy(navigable_map).float()
            combined[3:] = torch.from_numpy(semantic_map).permute(2,0,1).float()
            torch.save(combined, floor_dir / 'combined_map.pt')
            
            # PNG visualizations
            # With trajectory
            vis_traj = self.visualize(floor_id=floor_id, show_legend=True, show_trajectory=True)
            cv2.imwrite(str(floor_dir / 'semantic_map_with_trajectory.png'), cv2.cvtColor(vis_traj, cv2.COLOR_RGB2BGR))
            
            # Without trajectory (clean map)
            vis_clean = self.visualize(floor_id=floor_id, show_legend=True, show_trajectory=False)
            cv2.imwrite(str(floor_dir / 'semantic_map.png'), cv2.cvtColor(vis_clean, cv2.COLOR_RGB2BGR))
            
            # Without legend (for processing)
            vis_no_legend = self.visualize(floor_id=floor_id, show_legend=False, show_trajectory=False)
            cv2.imwrite(str(floor_dir / 'semantic_map_raw.png'), cv2.cvtColor(vis_no_legend, cv2.COLOR_RGB2BGR))
            
            cv2.imwrite(str(floor_dir / 'occupancy_map.png'), (obstacle_map * 255).astype(np.uint8))
            cv2.imwrite(str(floor_dir / 'navigable_map.png'), (navigable_map * 255).astype(np.uint8))
            cv2.imwrite(str(floor_dir / 'explored_map.png'), (explored_map * 255).astype(np.uint8))
            
            # Per-category heatmaps
            categories_dir = floor_dir / 'categories'
            categories_dir.mkdir(exist_ok=True)
            for cat_name, cat_idx in self.category_to_idx.items():
                cat_map = semantic_map[:, :, cat_idx]
                if cat_map.max() > 0:
                    cat_vis = (cat_map * 255).astype(np.uint8)
                    cat_color = cv2.applyColorMap(cat_vis, cv2.COLORMAP_JET)
                    cv2.imwrite(str(categories_dir / f'{cat_name}.png'), cat_color)
            
            # Floor-specific objects
            floor_objs = {str(oid): {
                'class': o.class_name,
                'position': [float(x) for x in o.mean_position],  # Ensure JSON serializable
                'confidence': float(o.mean_confidence),
                'observations': o.observations,
                'footprint': list(o.footprint)
            } for oid, o in self.get_confirmed_objects().items() if o.floor_level == floor_id}
            
            with open(floor_dir / 'objects.json', 'w') as f:
                json.dump(floor_objs, f, indent=2)
            
            # Auto-generate map_annotations.json for known-env navigation
            # (can be refined later with map_editor_v2)
            occ_map_size = self.occupancy_map.map_size
            map_annotations = {
                'map_size': [occ_map_size, occ_map_size],
                'resolution_cm': self.resolution,
                'rooms': {},  # Empty — user fills via map_editor_v2
                'objects': [],
                'metadata': {
                    'tool': 'auto-generated from exploration',
                    'floor': floor_id,
                },
            }
            for oid, obj_data in floor_objs.items():
                pos = obj_data['position']
                mx, my = self.occupancy_map._world_to_map(pos[0], pos[2])
                map_annotations['objects'].append({
                    'label': obj_data['class'],
                    'position': [int(mx), int(my)],
                    'room': None,
                })
            
            # Only write if no existing annotations (don't overwrite manual edits)
            ann_path = floor_dir / 'map_annotations.json'
            if not ann_path.exists():
                with open(ann_path, 'w') as f:
                    json.dump(map_annotations, f, indent=2)
                print(f"  Auto-generated {ann_path}")
        
        # Global metadata
        all_objs = {str(oid): {
            'class': o.class_name,
            'position': [float(x) for x in o.mean_position],
            'confidence': float(o.mean_confidence),
            'observations': o.observations,
            'floor': o.floor_level,
            'footprint': list(o.footprint)
        } for oid, o in self.get_confirmed_objects().items()}
        
        with open(output_dir / 'objects.json', 'w') as f:
            json.dump(all_objs, f, indent=2)
        
        with open(output_dir / 'stairs.json', 'w') as f:
            json.dump(self.stairs_detected, f, indent=2)
        
        with open(output_dir / 'floor_info.json', 'w') as f:
            json.dump({
                'num_floors': len(self.floor_semantic_maps),
                'floor_ids': list(self.floor_semantic_maps.keys()),
                'base_heights': {str(k): float(v) for k, v in self.floor_base_heights.items() if v is not None},
                'objects_per_floor': {
                    str(fid): len([o for o in self.get_confirmed_objects().values() if o.floor_level == fid])
                    for fid in self.floor_semantic_maps.keys()
                }
            }, f, indent=2)
        
        with open(output_dir / 'category_mapping.json', 'w') as f:
            json.dump({
                'category_to_idx': self.category_to_idx,
                'colors': {k: list(v) for k, v in CATEGORY_COLORS.items()},
                'num_channels': self.num_categories
            }, f, indent=2)
        
        with open(output_dir / 'statistics.json', 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
        
        print(f"\nSaved to {output_dir}:")
        print(f"  - {len(self.floor_semantic_maps)} floor(s)")
        for fid in sorted(self.floor_semantic_maps.keys()):
            n_obj = len([o for o in self.get_confirmed_objects().values() if o.floor_level == fid])
            print(f"    Floor {fid}: {n_obj} objects")
    
    def get_statistics(self):
        confirmed = self.get_confirmed_objects()
        counts = defaultdict(int)
        floor_counts = defaultdict(int)
        for o in confirmed.values():
            counts[o.class_name] += 1
            floor_counts[o.floor_level] += 1
        return {
            'total_floors': len(self.floor_semantic_maps),
            'current_floor': self.current_floor,
            'confirmed_objects': len(confirmed),
            'tracked': len(self.tracked_objects),
            'class_counts': dict(counts),
            'objects_per_floor': dict(floor_counts),
            'stairs_found': len(self.stairs_detected),
            'explored_percent': self.occupancy_map.get_statistics().get('explored_percent', 0),
            'origin_x': float(self.occupancy_map.origin_x),
            'origin_z': float(self.occupancy_map.origin_z),
            'resolution_cm': self.resolution,
            'map_size': self.occupancy_map.map_size
        }