"""
Lightweight Visual Odometry for Navigation
==========================================

Simple, fast visual odometry for real-time navigation.
Optimized for speed on CPU.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class TrackingState(Enum):
    NOT_INITIALIZED = "not_initialized"
    OK = "ok"
    LOST = "lost"


@dataclass
class VOResult:
    """Visual odometry result."""
    success: bool
    position: np.ndarray      # [x, y, z] estimated position
    rotation: np.ndarray      # [3, 3] rotation matrix
    inliers: int
    tracking_state: TrackingState


class LightweightVO:
    """
    Fast visual odometry using ORB features.
    
    Optimizations:
    - Fewer features (300)
    - Downscaled image for detection
    - FLANN matcher for speed
    - Frame skipping option
    """
    
    def __init__(
        self,
        # Camera params (Habitat defaults)
        width: int = 640,
        height: int = 480,
        hfov: float = 90.0,
        # VO params - tuned for speed
        n_features: int = 300,
        scale_factor: float = 1.2,
        n_levels: int = 4,           # Reduced pyramid levels
        min_inliers: int = 12,
        detection_scale: float = 0.5, # Downscale for detection
    ):
        # Camera intrinsics
        self.width = width
        self.height = height
        self.fx = width / (2.0 * np.tan(np.radians(hfov / 2)))
        self.fy = self.fx
        self.cx = width / 2
        self.cy = height / 2
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Scaled intrinsics for detection
        self.detection_scale = detection_scale
        self.det_width = int(width * detection_scale)
        self.det_height = int(height * detection_scale)
        self.K_scaled = self.K.copy()
        self.K_scaled[0, :] *= detection_scale
        self.K_scaled[1, :] *= detection_scale
        
        # VO params
        self.n_features = n_features
        self.min_inliers = min_inliers
        
        # ORB detector - optimized settings
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            patchSize=21,
            fastThreshold=15
        )
        
        # BF matcher with crossCheck for accuracy
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # State
        self.state = TrackingState.NOT_INITIALIZED
        self.current_R = np.eye(3)
        self.current_t = np.zeros(3)
        
        # Previous frame data
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        
        # Stats
        self.frame_count = 0
        self.lost_count = 0
    
    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray
    ) -> VOResult:
        """
        Process RGB-D frame and estimate pose.
        """
        self.frame_count += 1
        
        # Convert to grayscale
        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        
        # Downscale for faster detection
        gray_small = cv2.resize(gray, (self.det_width, self.det_height))
        depth_small = cv2.resize(depth, (self.det_width, self.det_height), interpolation=cv2.INTER_NEAREST)
        
        # Detect ORB features on downscaled image
        keypoints, descriptors = self.orb.detectAndCompute(gray_small, None)
        
        if keypoints is None or len(keypoints) < self.min_inliers:
            return self._make_result(False, 0)
        
        # Convert keypoints to array (in downscaled coords)
        kp_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        
        # Get 3D points from downscaled depth
        points_3d = self._backproject_scaled(kp_array, depth_small)
        
        # Filter invalid depth
        valid = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 8.0)
        if valid.sum() < self.min_inliers:
            return self._make_result(False, 0)
        
        kp_array = kp_array[valid]
        points_3d = points_3d[valid]
        descriptors = descriptors[valid]
        
        # First frame - initialize
        if self.state == TrackingState.NOT_INITIALIZED:
            self.prev_keypoints = kp_array
            self.prev_descriptors = descriptors
            self.prev_points_3d = points_3d
            self.state = TrackingState.OK
            return self._make_result(True, len(kp_array))
        
        # Match with previous frame
        if len(self.prev_descriptors) < self.min_inliers:
            return self._make_result(False, 0)
        
        matches = self.matcher.match(self.prev_descriptors, descriptors)
        
        # Sort by distance and take best matches
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:min(len(matches), 100)]  # Top 100 matches
        
        if len(matches) < self.min_inliers:
            self.lost_count += 1
            if self.lost_count > 5:
                self.state = TrackingState.LOST
            return self._make_result(False, len(matches))
        
        # Get matched 3D-2D correspondences
        pts_3d = np.array([self.prev_points_3d[m.queryIdx] for m in matches])
        pts_2d = np.array([kp_array[m.trainIdx] for m in matches])
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.astype(np.float64),
            pts_2d.astype(np.float64),
            self.K_scaled,  # Use scaled intrinsics
            None,
            flags=cv2.SOLVEPNP_EPNP,  # Faster than ITERATIVE
            reprojectionError=2.0,
            iterationsCount=50,
            confidence=0.95
        )
        
        if not success or inliers is None or len(inliers) < self.min_inliers:
            self.lost_count += 1
            if self.lost_count > 5:
                self.state = TrackingState.LOST
            return self._make_result(False, len(inliers) if inliers is not None else 0)
        
        # Update pose
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        
        # Sanity check - reject large motions (likely errors)
        if np.linalg.norm(t) > 0.5:  # Max 0.5m per frame
            self.lost_count += 1
            return self._make_result(False, len(inliers))
        
        # Accumulate pose
        self.current_t = self.current_t + self.current_R @ t
        self.current_R = self.current_R @ R
        
        # Update state
        self.prev_keypoints = kp_array
        self.prev_descriptors = descriptors
        self.prev_points_3d = points_3d
        self.state = TrackingState.OK
        self.lost_count = 0
        
        return self._make_result(True, len(inliers))
    
    def _backproject_scaled(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Backproject 2D pixels to 3D using scaled intrinsics."""
        u = pixels[:, 0].astype(int)
        v = pixels[:, 1].astype(int)
        
        # Clamp to image bounds
        u = np.clip(u, 0, self.det_width - 1)
        v = np.clip(v, 0, self.det_height - 1)
        
        # Get depths
        z = depth[v, u]
        
        # Backproject using scaled intrinsics
        fx = self.K_scaled[0, 0]
        fy = self.K_scaled[1, 1]
        cx = self.K_scaled[0, 2]
        cy = self.K_scaled[1, 2]
        
        x = (pixels[:, 0] - cx) * z / fx
        y = (pixels[:, 1] - cy) * z / fy
        
        return np.stack([x, y, z], axis=1)
    
    def _make_result(self, success: bool, inliers: int) -> VOResult:
        """Create VOResult."""
        return VOResult(
            success=success,
            position=self.current_t.copy(),
            rotation=self.current_R.copy(),
            inliers=inliers,
            tracking_state=self.state
        )
    
    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current pose (position, rotation)."""
        return self.current_t.copy(), self.current_R.copy()
    
    def reset(self):
        """Reset odometry."""
        self.state = TrackingState.NOT_INITIALIZED
        self.current_R = np.eye(3)
        self.current_t = np.zeros(3)
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        self.frame_count = 0
        self.lost_count = 0


class PoseTracker:
    """
    Tracks pose using either Visual Odometry or Ground Truth.
    """
    
    def __init__(self, use_vo: bool = False):
        self.use_vo = use_vo
        self.vo = LightweightVO() if use_vo else None
        
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        
        # For comparison
        self.gt_positions: List[np.ndarray] = []
        self.est_positions: List[np.ndarray] = []
    
    def update(
        self,
        rgb: np.ndarray = None,
        depth: np.ndarray = None,
        gt_position: np.ndarray = None,
        gt_rotation: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update pose estimate."""
        if self.use_vo and rgb is not None and depth is not None:
            result = self.vo.process_frame(rgb, depth)
            self.position = result.position
            self.rotation = result.rotation
            
            self.est_positions.append(self.position.copy())
            if gt_position is not None:
                self.gt_positions.append(gt_position.copy())
        else:
            if gt_position is not None:
                self.position = gt_position.copy()
            if gt_rotation is not None:
                self.rotation = gt_rotation.copy()
        
        return self.position, self.rotation
    
    def compute_drift(self) -> Optional[float]:
        """Compute drift between VO and GT."""
        if len(self.gt_positions) < 2 or len(self.est_positions) < 2:
            return None
        
        n = min(len(self.gt_positions), len(self.est_positions))
        gt = np.array(self.gt_positions[:n])
        est = np.array(self.est_positions[:n])
        
        errors = np.linalg.norm(gt - est, axis=1)
        return float(np.mean(errors))
    
    def reset(self):
        """Reset tracker."""
        if self.vo:
            self.vo.reset()
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.gt_positions.clear()
        self.est_positions.clear()