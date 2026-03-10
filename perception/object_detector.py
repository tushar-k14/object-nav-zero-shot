"""
Object Detector
===============

YOLO-based object detection with 3D position extraction.

Features:
1. YOLOv8 instance segmentation for accurate masks
2. Depth-based 3D position extraction
3. Object tracking across frames (proximity-based merging)
4. Bounding box, mask, category, and confidence

This module focuses on OBJECTS only - no walls, floors, or ceilings.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geometry import (
    CameraIntrinsics,
    unproject_pixel_to_3d,
    merge_nearby_positions
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available. Install: pip install ultralytics")


# ObjectNav relevant categories
OBJECTNAV_CATEGORIES = {
    'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'refrigerator', 'sink', 'oven', 'microwave', 'potted plant', 'clock',
    'vase', 'book', 'bottle', 'cup', 'bowl', 'keyboard', 'cell phone',
    'backpack', 'umbrella', 'handbag', 'suitcase', 'sports ball',
    'baseball bat', 'tennis racket', 'wine glass', 'fork', 'knife',
    'spoon', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'pizza', 'donut', 'cake', 'mouse', 'remote', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
}


@dataclass
class Detection:
    """Single object detection."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None  # HxW binary mask
    centroid_2d: Tuple[int, int] = (0, 0)  # Pixel centroid
    position_3d: Optional[np.ndarray] = None  # World position [x, y, z]
    depth_median: float = 0.0  # Median depth of object
    
    def to_dict(self) -> Dict:
        return {
            'class': self.class_name,
            'confidence': float(self.confidence),
            'bbox': list(self.bbox),
            'centroid_2d': list(self.centroid_2d),
            'position_3d': self.position_3d.tolist() if self.position_3d is not None else None,
            'depth': float(self.depth_median)
        }


@dataclass
class TrackedObject:
    """Object tracked across multiple observations."""
    object_id: int
    class_name: str
    positions: List[np.ndarray] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    observations: int = 0
    last_seen_frame: int = 0
    
    @property
    def mean_position(self) -> np.ndarray:
        """Get mean position across observations."""
        if len(self.positions) == 0:
            return np.array([0, 0, 0])
        return np.mean(self.positions, axis=0)
    
    @property
    def mean_confidence(self) -> float:
        """Get mean confidence."""
        if len(self.confidences) == 0:
            return 0.0
        return float(np.mean(self.confidences))
    
    def add_observation(self, position: np.ndarray, confidence: float, frame: int):
        """Add new observation."""
        self.positions.append(position.copy())
        self.confidences.append(confidence)
        self.observations += 1
        self.last_seen_frame = frame
    
    def to_dict(self) -> Dict:
        return {
            'id': self.object_id,
            'class': self.class_name,
            'position': self.mean_position.tolist(),
            'confidence': self.mean_confidence,
            'observations': self.observations,
            'last_seen': self.last_seen_frame
        }


class ObjectDetector:
    """
    YOLO-based object detector with 3D position extraction.
    
    Key parameters for reducing duplicate detections:
    - confidence_threshold: Min detection confidence (default 0.3)
    - merge_threshold: Distance to merge same-class objects (default 1.0m)
    - min_observations: Min observations before object is "confirmed"
    """
    
    def __init__(
        self,
        model_size: str = 'x',
        confidence_threshold: float = 0.3,  # Increased from 0.15
        use_segmentation: bool = True,
        filter_categories: bool = True,
        merge_threshold: float = 1.0,  # Increased from 0.5
        min_observations: int = 2  # Require multiple sightings
    ):
        """
        Initialize detector.
        
        Args:
            model_size: YOLO model size (n/s/m/l/x)
            confidence_threshold: Detection confidence threshold (0.3+ recommended)
            use_segmentation: Use YOLOv8-seg for masks
            filter_categories: Only detect ObjectNav-relevant categories
            merge_threshold: Distance (meters) to merge same-class objects
            min_observations: Minimum observations to confirm object
        """
        self.conf_threshold = confidence_threshold
        self.use_seg = use_segmentation
        self.filter_categories = filter_categories
        self.merge_threshold = merge_threshold
        self.min_observations = min_observations
        
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLOv8 not available. Install: pip install ultralytics")
        
        # Load model
        if use_segmentation:
            model_name = f'yolov8{model_size}-seg.pt'
        else:
            model_name = f'yolov8{model_size}.pt'
        
        print(f"Loading YOLO model: {model_name}")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Merge threshold: {merge_threshold}m")
        self.model = YOLO(model_name)
        self.class_names = self.model.names
        
        # Object tracking
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 0
        self.frame_count = 0
    
    def detect(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        agent_position: Optional[np.ndarray] = None,
        agent_rotation: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None
    ) -> List[Detection]:
        """
        Detect objects in RGB image.
        
        If depth and pose are provided, also computes 3D positions.
        
        Args:
            rgb: HxWx3 RGB image
            depth: Optional HxW depth image
            agent_position: Optional [x,y,z] agent world position
            agent_rotation: Optional [x,y,z,w] quaternion
            intrinsics: Optional camera intrinsics
            
        Returns:
            List of Detection objects
        """
        self.frame_count += 1
        h, w = rgb.shape[:2]
        
        # Run YOLO
        results = self.model(rgb, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        if len(results) == 0:
            return detections
        
        result = results[0]
        
        # Get boxes
        if result.boxes is None:
            return detections
        
        boxes = result.boxes.data.cpu().numpy()
        
        # Get masks if available
        if self.use_seg and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
        else:
            masks = [None] * len(boxes)
        
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            x1, y1, x2, y2, conf, cls_id = box[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)
            
            class_name = self.class_names.get(cls_id, 'unknown')
            
            # Filter categories
            if self.filter_categories:
                if class_name.lower() not in OBJECTNAV_CATEGORIES:
                    continue
            
            # Process mask
            if mask is not None:
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
            else:
                mask_binary = None
            
            # Compute centroid
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Compute 3D position if depth available
            position_3d = None
            depth_median = 0.0
            
            if depth is not None and intrinsics is not None:
                # Get depth values within detection
                if mask_binary is not None:
                    # Use mask for accurate depth
                    depth_values = depth[mask_binary > 0]
                else:
                    # Use bounding box
                    depth_values = depth[y1:y2, x1:x2].flatten()
                
                # Filter valid depth
                valid_depth = depth_values[(depth_values > 0.1) & (depth_values < 10.0)]
                
                if len(valid_depth) > 0:
                    depth_median = float(np.median(valid_depth))
                    
                    # Unproject centroid to 3D
                    if agent_position is not None and agent_rotation is not None:
                        position_3d = unproject_pixel_to_3d(
                            cx, cy, depth_median,
                            intrinsics, agent_position, agent_rotation
                        )
            
            detection = Detection(
                class_name=class_name,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                mask=mask_binary,
                centroid_2d=(cx, cy),
                position_3d=position_3d,
                depth_median=depth_median
            )
            
            detections.append(detection)
        
        return detections
    
    def detect_and_track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        agent_position: np.ndarray,
        agent_rotation: np.ndarray,
        intrinsics = None  # CameraIntrinsics, optional
    ) -> Tuple[List[Detection], List[int]]:
        """
        Detect objects and track them across frames.
        
        Args:
            rgb: RGB image
            depth: Depth image
            agent_position: Agent world position
            agent_rotation: Agent rotation quaternion
            intrinsics: Camera intrinsics (uses defaults if None)
            
        Returns:
            (detections, object_ids) - matched object IDs for each detection
        """
        # Use default intrinsics if not provided
        if intrinsics is None:
            intrinsics = CameraIntrinsics.habitat_default(
                width=rgb.shape[1], height=rgb.shape[0]
            )
        
        detections = self.detect(
            rgb, depth, agent_position, agent_rotation, intrinsics
        )
        
        object_ids = []
        
        for det in detections:
            if det.position_3d is None:
                object_ids.append(-1)
                continue
            
            # Try to match with existing tracked objects
            matched_id = self._match_to_tracked(det)
            
            if matched_id >= 0:
                # Update existing object
                self.tracked_objects[matched_id].add_observation(
                    det.position_3d, det.confidence, self.frame_count
                )
            else:
                # Create new tracked object
                matched_id = self.next_object_id
                self.next_object_id += 1
                
                obj = TrackedObject(
                    object_id=matched_id,
                    class_name=det.class_name
                )
                obj.add_observation(det.position_3d, det.confidence, self.frame_count)
                self.tracked_objects[matched_id] = obj
            
            object_ids.append(matched_id)
        
        return detections, object_ids
    
    def _match_to_tracked(self, detection: Detection) -> int:
        """
        Try to match detection to existing tracked object.
        Uses merge_threshold for distance-based matching.
        
        Returns object_id if matched, -1 otherwise.
        """
        if detection.position_3d is None:
            return -1
        
        best_match = -1
        best_distance = self.merge_threshold  # Use instance variable
        
        for obj_id, obj in self.tracked_objects.items():
            # Must be same class
            if obj.class_name.lower() != detection.class_name.lower():
                continue
            
            # Check 3D distance
            dist = np.linalg.norm(detection.position_3d - obj.mean_position)
            
            if dist < best_distance:
                best_distance = dist
                best_match = obj_id
        
        return best_match
    
    def get_tracked_objects(self, min_observations: int = None) -> List[TrackedObject]:
        """Get tracked objects with minimum observations."""
        if min_observations is None:
            min_observations = self.min_observations
        return [
            obj for obj in self.tracked_objects.values()
            if obj.observations >= min_observations
        ]
    
    def get_objects_by_class(self, class_name: str) -> List[TrackedObject]:
        """Get all tracked objects of a specific class."""
        return [
            obj for obj in self.tracked_objects.values()
            if obj.class_name.lower() == class_name.lower()
        ]
    
    def visualize(
        self,
        rgb: np.ndarray,
        detections: List[Detection],
        object_ids: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Create visualization with bounding boxes and labels.
        
        Args:
            rgb: RGB image
            detections: List of detections
            object_ids: Optional list of tracked object IDs
            
        Returns:
            Annotated RGB image
        """
        vis = rgb.copy()
        
        # Generate colors for classes
        np.random.seed(42)
        colors = {}
        
        for i, det in enumerate(detections):
            if det.class_name not in colors:
                colors[det.class_name] = tuple(map(int, np.random.randint(50, 255, 3)))
            
            color = colors[det.class_name]
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw mask if available
            if det.mask is not None:
                mask_colored = np.zeros_like(vis)
                mask_colored[det.mask > 0] = color
                vis = cv2.addWeighted(vis, 1.0, mask_colored, 0.3, 0)
            
            # Label
            label = f"{det.class_name} {det.confidence:.2f}"
            if object_ids is not None and i < len(object_ids) and object_ids[i] >= 0:
                label = f"[{object_ids[i]}] {label}"
            if det.depth_median > 0:
                label += f" {det.depth_median:.1f}m"
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def reset_tracking(self):
        """Reset object tracking."""
        self.tracked_objects.clear()
        self.next_object_id = 0
        self.frame_count = 0
