"""
Visual Odometry
===============

Hybrid visual odometry combining:
1. ORB feature matching for robust motion estimation
2. RGB-D ICP for geometric refinement

This provides robustness in textured areas (ORB) and
accurate alignment in geometric areas (ICP).
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Pose, OdometryResult, TrackingState
from core.transforms import CameraIntrinsics, depth_to_pointcloud, transform_pointcloud
from slam.feature_extractor import FeatureExtractor


@dataclass
class VOConfig:
    """Visual Odometry configuration."""
    # Feature extraction
    n_features: int = 1000
    match_ratio: float = 0.75
    
    # Motion estimation
    min_inliers: int = 20
    ransac_threshold: float = 3.0
    
    # ICP refinement
    use_icp: bool = True
    icp_max_iterations: int = 30
    icp_tolerance: float = 1e-6
    icp_max_correspondence_dist: float = 0.1  # meters
    
    # Depth filtering
    min_depth: float = 0.1
    max_depth: float = 10.0
    
    # Keyframe selection
    keyframe_translation: float = 0.1  # meters
    keyframe_rotation: float = 0.1     # radians


class VisualOdometry:
    """
    Hybrid visual odometry using ORB features + ICP refinement.
    """
    
    def __init__(self, camera: CameraIntrinsics, config: VOConfig = None):
        """
        Initialize visual odometry.
        
        Args:
            camera: Camera intrinsic parameters
            config: VO configuration
        """
        self.camera = camera
        self.config = config or VOConfig()
        
        # Feature extractor
        self.extractor = FeatureExtractor(
            n_features=self.config.n_features,
            match_ratio=self.config.match_ratio
        )
        
        # State
        self.state = TrackingState.NOT_INITIALIZED
        self.current_pose = Pose.identity()
        
        # Previous frame data
        self.prev_rgb = None
        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        
        # Keyframe data
        self.keyframe_pose = None
        self.keyframe_rgb = None
        self.keyframe_depth = None
        
        # Statistics
        self.frame_count = 0
        self.total_distance = 0.0
    
    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp: float = 0.0
    ) -> OdometryResult:
        """
        Process a new RGB-D frame and estimate motion.
        
        Args:
            rgb: RGB image [H, W, 3]
            depth: Depth image [H, W] in meters
            timestamp: Frame timestamp
            
        Returns:
            OdometryResult with relative pose
        """
        self.frame_count += 1
        
        # Extract features with 3D positions
        keypoints, descriptors, points_3d = self.extractor.extract_with_depth(
            rgb, depth,
            self.camera.fx, self.camera.fy,
            self.camera.cx, self.camera.cy,
            self.config.min_depth, self.config.max_depth
        )
        
        # First frame - initialize
        if self.state == TrackingState.NOT_INITIALIZED:
            self._initialize(rgb, depth, keypoints, descriptors, points_3d, timestamp)
            return OdometryResult(
                success=True,
                relative_pose=Pose.identity(),
                inlier_count=len(keypoints),
                total_matches=len(keypoints),
                method="init"
            )
        
        # Try feature-based motion estimation
        result = self._estimate_motion_features(
            keypoints, descriptors, points_3d
        )
        
        # Refine with ICP if enabled and feature estimation succeeded
        if result.success and self.config.use_icp:
            result = self._refine_with_icp(
                depth, result.relative_pose
            )
        
        # Update pose if successful
        if result.success:
            self.current_pose = self.current_pose.compose(result.relative_pose)
            self.current_pose.timestamp = timestamp
            
            # Update distance traveled
            if result.relative_pose is not None:
                self.total_distance += np.linalg.norm(result.relative_pose.position)
            
            self.state = TrackingState.OK
        else:
            self.state = TrackingState.LOST
        
        # Update previous frame
        self.prev_rgb = rgb
        self.prev_depth = depth
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_points_3d = points_3d
        
        return result
    
    def _initialize(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        points_3d: np.ndarray,
        timestamp: float
    ):
        """Initialize odometry with first frame."""
        self.prev_rgb = rgb
        self.prev_depth = depth
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_points_3d = points_3d
        
        self.keyframe_rgb = rgb.copy()
        self.keyframe_depth = depth.copy()
        self.keyframe_pose = Pose.identity()
        self.keyframe_pose.timestamp = timestamp
        
        self.current_pose = Pose.identity()
        self.current_pose.timestamp = timestamp
        
        self.state = TrackingState.OK
    
    def _estimate_motion_features(
        self,
        curr_keypoints: np.ndarray,
        curr_descriptors: np.ndarray,
        curr_points_3d: np.ndarray
    ) -> OdometryResult:
        """
        Estimate motion using feature matching and PnP.
        
        Uses 3D-2D correspondences: previous 3D points matched to current 2D points.
        """
        if self.prev_descriptors is None or len(self.prev_descriptors) == 0:
            return OdometryResult(success=False, relative_pose=None, method="feature")
        
        if len(curr_descriptors) == 0:
            return OdometryResult(success=False, relative_pose=None, method="feature")
        
        # Match features
        matches = self.extractor.match(self.prev_descriptors, curr_descriptors)
        
        if len(matches) < self.config.min_inliers:
            return OdometryResult(
                success=False,
                relative_pose=None,
                total_matches=len(matches),
                method="feature"
            )
        
        # Get 3D points from previous frame and 2D points from current frame
        pts_3d = np.array([self.prev_points_3d[m.idx1] for m in matches])
        pts_2d = np.array([curr_keypoints[m.idx2] for m in matches])
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.astype(np.float64),
            pts_2d.astype(np.float64),
            self.camera.K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.config.ransac_threshold,
            iterationsCount=100
        )
        
        if not success or inliers is None:
            return OdometryResult(
                success=False,
                relative_pose=None,
                total_matches=len(matches),
                method="feature"
            )
        
        inlier_count = len(inliers)
        
        if inlier_count < self.config.min_inliers:
            return OdometryResult(
                success=False,
                relative_pose=None,
                inlier_count=inlier_count,
                total_matches=len(matches),
                method="feature"
            )
        
        # Convert to pose
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        
        # This gives camera pose in previous frame's coordinate system
        # We need relative motion: T_prev_to_curr
        relative_pose = Pose(
            position=t,
            rotation=R,
            timestamp=0.0
        )
        
        return OdometryResult(
            success=True,
            relative_pose=relative_pose,
            inlier_count=inlier_count,
            total_matches=len(matches),
            method="feature"
        )
    
    def _refine_with_icp(
        self,
        curr_depth: np.ndarray,
        initial_pose: Pose
    ) -> OdometryResult:
        """
        Refine motion estimate using point-to-plane ICP.
        """
        if self.prev_depth is None:
            return OdometryResult(
                success=True,
                relative_pose=initial_pose,
                method="feature"
            )
        
        # Generate point clouds (subsampled for speed)
        prev_cloud = depth_to_pointcloud(
            self.prev_depth, self.camera,
            max_depth=self.config.max_depth,
            subsample=4
        )
        
        curr_cloud = depth_to_pointcloud(
            curr_depth, self.camera,
            max_depth=self.config.max_depth,
            subsample=4
        )
        
        if len(prev_cloud) < 100 or len(curr_cloud) < 100:
            return OdometryResult(
                success=True,
                relative_pose=initial_pose,
                method="feature"
            )
        
        # Transform current cloud by initial estimate
        T_init = initial_pose.to_matrix()
        curr_cloud_transformed = transform_pointcloud(curr_cloud, np.linalg.inv(T_init))
        
        # Run ICP
        T_refined, inlier_count = self._icp(
            prev_cloud,
            curr_cloud_transformed,
            np.eye(4),  # Start from identity since we pre-transformed
            self.config.icp_max_iterations,
            self.config.icp_tolerance,
            self.config.icp_max_correspondence_dist
        )
        
        # Combine initial and refined transforms
        T_total = T_init @ T_refined
        refined_pose = Pose.from_matrix(T_total)
        
        return OdometryResult(
            success=True,
            relative_pose=refined_pose,
            inlier_count=inlier_count,
            method="hybrid"
        )
    
    def _icp(
        self,
        source: np.ndarray,
        target: np.ndarray,
        T_init: np.ndarray,
        max_iterations: int,
        tolerance: float,
        max_dist: float
    ) -> Tuple[np.ndarray, int]:
        """
        Point-to-point ICP implementation.
        
        Args:
            source: [N, 3] source point cloud
            target: [M, 3] target point cloud
            T_init: Initial 4x4 transformation
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            max_dist: Maximum correspondence distance
            
        Returns:
            T: Refined transformation
            inlier_count: Number of inlier correspondences
        """
        T = T_init.copy()
        source_h = np.hstack([source, np.ones((len(source), 1))])
        
        prev_error = float('inf')
        
        for iteration in range(max_iterations):
            # Transform source points
            source_transformed = (T @ source_h.T).T[:, :3]
            
            # Find nearest neighbors (simple brute force for now)
            # In production, use KD-tree
            dists = np.linalg.norm(
                source_transformed[:, np.newaxis, :] - target[np.newaxis, :, :],
                axis=2
            )
            nearest_idx = np.argmin(dists, axis=1)
            nearest_dist = dists[np.arange(len(source)), nearest_idx]
            
            # Filter by distance
            valid = nearest_dist < max_dist
            if valid.sum() < 10:
                break
            
            src_valid = source_transformed[valid]
            tgt_valid = target[nearest_idx[valid]]
            
            # Compute centroids
            src_centroid = np.mean(src_valid, axis=0)
            tgt_centroid = np.mean(tgt_valid, axis=0)
            
            # Center the points
            src_centered = src_valid - src_centroid
            tgt_centered = tgt_valid - tgt_centroid
            
            # Compute rotation using SVD
            H = src_centered.T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = tgt_centroid - R @ src_centroid
            
            # Update transformation
            T_delta = np.eye(4)
            T_delta[:3, :3] = R
            T_delta[:3, 3] = t
            T = T_delta @ T
            
            # Check convergence
            error = np.mean(nearest_dist[valid])
            if abs(prev_error - error) < tolerance:
                break
            prev_error = error
        
        inlier_count = int(valid.sum()) if 'valid' in dir() else 0
        return T, inlier_count
    
    def get_pose(self) -> Pose:
        """Get current estimated pose."""
        return self.current_pose
    
    def get_state(self) -> TrackingState:
        """Get current tracking state."""
        return self.state
    
    def reset(self):
        """Reset odometry to initial state."""
        self.state = TrackingState.NOT_INITIALIZED
        self.current_pose = Pose.identity()
        self.prev_rgb = None
        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        self.keyframe_pose = None
        self.frame_count = 0
        self.total_distance = 0.0
    
    def is_keyframe(self) -> bool:
        """Check if current frame should be a keyframe."""
        if self.keyframe_pose is None:
            return True
        
        # Compute relative motion from last keyframe
        T_kf = self.keyframe_pose.to_matrix()
        T_curr = self.current_pose.to_matrix()
        T_rel = np.linalg.inv(T_kf) @ T_curr
        
        # Translation magnitude
        translation = np.linalg.norm(T_rel[:3, 3])
        
        # Rotation magnitude (angle of rotation)
        R_rel = T_rel[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        
        return (translation > self.config.keyframe_translation or 
                angle > self.config.keyframe_rotation)
    
    def set_keyframe(self):
        """Mark current frame as keyframe."""
        self.keyframe_pose = Pose(
            position=self.current_pose.position.copy(),
            rotation=self.current_pose.rotation.copy() if isinstance(self.current_pose.rotation, np.ndarray) and self.current_pose.rotation.shape == (3,3) else self.current_pose.rotation,
            timestamp=self.current_pose.timestamp
        )
        if self.prev_rgb is not None:
            self.keyframe_rgb = self.prev_rgb.copy()
        if self.prev_depth is not None:
            self.keyframe_depth = self.prev_depth.copy()
