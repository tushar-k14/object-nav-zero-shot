"""
Visual SLAM System
==================

Integrates all SLAM components:
- Visual Odometry (ORB + ICP)
- Loop Closure Detection (NetVLAD-style)
- Pose Graph Optimization

Provides ground truth comparison for evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Pose, Keyframe, TrackingState
from core.transforms import CameraIntrinsics
from slam.visual_odometry import VisualOdometry, VOConfig
from slam.loop_closure import LoopClosureDetector, LoopClosureConfig
from slam.pose_graph import PoseGraph, EdgeType


@dataclass
class SLAMConfig:
    """SLAM system configuration."""
    # Visual Odometry
    vo_config: VOConfig = field(default_factory=VOConfig)
    
    # Loop Closure
    lc_config: LoopClosureConfig = field(default_factory=LoopClosureConfig)
    
    # Keyframe selection
    keyframe_translation: float = 0.2   # meters
    keyframe_rotation: float = 0.15     # radians
    
    # Optimization
    optimize_every_n_keyframes: int = 10
    optimize_on_loop_closure: bool = True
    
    # Ground truth comparison
    record_ground_truth: bool = True


@dataclass
class SLAMResult:
    """Result from SLAM processing."""
    # Estimated pose
    estimated_pose: Pose
    
    # Tracking status
    tracking_state: TrackingState
    
    # Keyframe info
    is_keyframe: bool
    keyframe_id: Optional[int]
    
    # Loop closure info
    loop_detected: bool
    loop_keyframe_id: Optional[int]
    
    # Optimization
    optimized: bool
    
    # Timing
    processing_time_ms: float


class VisualSLAM:
    """
    Complete Visual SLAM system.
    """
    
    def __init__(
        self,
        camera: CameraIntrinsics,
        config: SLAMConfig = None
    ):
        """
        Initialize SLAM system.
        
        Args:
            camera: Camera intrinsic parameters
            config: SLAM configuration
        """
        self.camera = camera
        self.config = config or SLAMConfig()
        
        # Components
        self.visual_odometry = VisualOdometry(camera, self.config.vo_config)
        self.loop_closure = LoopClosureDetector(self.config.lc_config)
        self.pose_graph = PoseGraph()
        
        # Keyframes
        self.keyframes: Dict[int, Keyframe] = {}
        self.current_keyframe_id = -1
        self.last_keyframe_pose = None
        
        # Ground truth tracking
        self.ground_truth_poses: List[Pose] = []
        self.estimated_poses: List[Pose] = []
        self.timestamps: List[float] = []
        
        # State
        self.frame_count = 0
        self.keyframe_count = 0
        self.loop_closure_count = 0
        self.optimization_count = 0
        
        # Timing
        self.total_processing_time = 0.0
    
    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp: float = None,
        ground_truth_pose: Optional[Pose] = None
    ) -> SLAMResult:
        """
        Process a new RGB-D frame.
        
        Args:
            rgb: RGB image [H, W, 3]
            depth: Depth image [H, W] in meters
            timestamp: Frame timestamp
            ground_truth_pose: Optional ground truth pose for evaluation
            
        Returns:
            SLAMResult
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = self.frame_count * 0.033  # Assume 30 FPS
        
        self.frame_count += 1
        
        # Visual Odometry
        vo_result = self.visual_odometry.process_frame(rgb, depth, timestamp)
        estimated_pose = self.visual_odometry.get_pose()
        tracking_state = self.visual_odometry.get_state()
        
        # Record for evaluation
        if self.config.record_ground_truth:
            self.estimated_poses.append(Pose(
                position=estimated_pose.position.copy(),
                rotation=estimated_pose.rotation.copy() if isinstance(estimated_pose.rotation, np.ndarray) else estimated_pose.rotation,
                timestamp=timestamp
            ))
            self.timestamps.append(timestamp)
            
            if ground_truth_pose is not None:
                self.ground_truth_poses.append(ground_truth_pose)
        
        # Check for keyframe
        is_keyframe = self._should_create_keyframe(estimated_pose)
        keyframe_id = None
        loop_detected = False
        loop_keyframe_id = None
        optimized = False
        
        if is_keyframe and tracking_state == TrackingState.OK:
            keyframe_id = self._create_keyframe(
                rgb, depth, estimated_pose, timestamp,
                self.visual_odometry.prev_keypoints,
                self.visual_odometry.prev_descriptors
            )
            
            # Check for loop closure
            if keyframe_id > self.config.lc_config.min_keyframe_gap:
                lc_result = self.loop_closure.detect(
                    keyframe_id,
                    rgb,
                    self.visual_odometry.prev_keypoints,
                    self.visual_odometry.prev_descriptors
                )
                
                if lc_result.detected and lc_result.verified:
                    loop_detected = True
                    loop_keyframe_id = lc_result.match_keyframe_id
                    self.loop_closure_count += 1
                    
                    # Add loop closure edge to pose graph
                    self._add_loop_closure_constraint(
                        keyframe_id,
                        lc_result.match_keyframe_id,
                        lc_result.relative_pose
                    )
                    
                    # Optimize on loop closure
                    if self.config.optimize_on_loop_closure:
                        self._optimize()
                        optimized = True
            
            # Periodic optimization
            if (self.keyframe_count % self.config.optimize_every_n_keyframes == 0 
                and not optimized):
                self._optimize()
                optimized = True
        
        processing_time = (time.time() - start_time) * 1000
        self.total_processing_time += processing_time
        
        return SLAMResult(
            estimated_pose=estimated_pose,
            tracking_state=tracking_state,
            is_keyframe=is_keyframe,
            keyframe_id=keyframe_id,
            loop_detected=loop_detected,
            loop_keyframe_id=loop_keyframe_id,
            optimized=optimized,
            processing_time_ms=processing_time
        )
    
    def _should_create_keyframe(self, current_pose: Pose) -> bool:
        """Check if we should create a new keyframe."""
        if self.last_keyframe_pose is None:
            return True
        
        # Compute relative motion
        translation = np.linalg.norm(
            current_pose.position - self.last_keyframe_pose.position
        )
        
        # Rotation difference
        if isinstance(current_pose.rotation, np.ndarray) and current_pose.rotation.shape == (3, 3):
            R1 = self.last_keyframe_pose.rotation
            R2 = current_pose.rotation
            if isinstance(R1, np.ndarray) and R1.shape == (3, 3):
                R_diff = R1.T @ R2
                angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            else:
                angle = 0
        else:
            angle = 0
        
        return (translation > self.config.keyframe_translation or 
                angle > self.config.keyframe_rotation)
    
    def _create_keyframe(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: Pose,
        timestamp: float,
        keypoints: np.ndarray,
        descriptors: np.ndarray
    ) -> int:
        """Create a new keyframe."""
        keyframe_id = self.keyframe_count
        self.keyframe_count += 1
        
        # Create keyframe
        keyframe = Keyframe(
            id=keyframe_id,
            pose=Pose(
                position=pose.position.copy(),
                rotation=pose.rotation.copy() if isinstance(pose.rotation, np.ndarray) else pose.rotation,
                timestamp=timestamp
            ),
            rgb=rgb.copy(),
            depth=depth.copy(),
            keypoints=keypoints.copy() if keypoints is not None else None,
            descriptors=descriptors.copy() if descriptors is not None else None
        )
        
        self.keyframes[keyframe_id] = keyframe
        
        # Add to loop closure database
        self.loop_closure.add_keyframe(
            keyframe_id, rgb, keypoints, descriptors
        )
        
        # Add node to pose graph
        node_id = self.pose_graph.add_node(keyframe.pose)
        
        # Add odometry edge to previous keyframe
        if self.current_keyframe_id >= 0:
            prev_pose = self.keyframes[self.current_keyframe_id].pose
            relative_pose = prev_pose.inverse().compose(pose)
            
            self.pose_graph.add_odometry_edge(
                self.current_keyframe_id,
                keyframe_id,
                relative_pose
            )
        
        self.current_keyframe_id = keyframe_id
        self.last_keyframe_pose = Pose(
            position=pose.position.copy(),
            rotation=pose.rotation.copy() if isinstance(pose.rotation, np.ndarray) else pose.rotation,
            timestamp=timestamp
        )
        
        # Update visual odometry keyframe
        self.visual_odometry.set_keyframe()
        
        return keyframe_id
    
    def _add_loop_closure_constraint(
        self,
        from_keyframe: int,
        to_keyframe: int,
        relative_pose: Optional[Pose]
    ):
        """Add a loop closure constraint to the pose graph."""
        if relative_pose is None:
            # Estimate relative pose from keyframe poses
            from_pose = self.keyframes[from_keyframe].pose
            to_pose = self.keyframes[to_keyframe].pose
            relative_pose = from_pose.inverse().compose(to_pose)
        
        self.pose_graph.add_loop_closure_edge(
            to_keyframe,  # From older
            from_keyframe,  # To newer
            relative_pose
        )
        
        print(f"Loop closure: {from_keyframe} <-> {to_keyframe}")
    
    def _optimize(self):
        """Run pose graph optimization."""
        result = self.pose_graph.optimize()
        self.optimization_count += 1
        
        if result.success:
            # Update keyframe poses from optimized graph
            optimized_poses = self.pose_graph.get_optimized_poses()
            
            for kf_id, pose in optimized_poses.items():
                if kf_id in self.keyframes:
                    self.keyframes[kf_id].pose = pose
            
            # Update current pose estimate
            if self.current_keyframe_id >= 0:
                self.visual_odometry.current_pose = self.keyframes[self.current_keyframe_id].pose
            
            print(f"Optimization: {result.iterations} iters, "
                  f"error {result.initial_error:.2f} -> {result.final_error:.2f} "
                  f"({result.improvement*100:.1f}% improvement)")
    
    def get_current_pose(self) -> Pose:
        """Get current estimated pose."""
        return self.visual_odometry.get_pose()
    
    def get_tracking_state(self) -> TrackingState:
        """Get current tracking state."""
        return self.visual_odometry.get_state()
    
    def get_trajectory(self) -> np.ndarray:
        """Get estimated trajectory as array of positions."""
        return np.array([p.position for p in self.estimated_poses])
    
    def get_ground_truth_trajectory(self) -> Optional[np.ndarray]:
        """Get ground truth trajectory if available."""
        if not self.ground_truth_poses:
            return None
        return np.array([p.position for p in self.ground_truth_poses])
    
    def compute_ate(self) -> Optional[float]:
        """
        Compute Absolute Trajectory Error (ATE).
        
        Returns:
            RMSE of position error, or None if no ground truth
        """
        if len(self.ground_truth_poses) == 0:
            return None
        
        n = min(len(self.estimated_poses), len(self.ground_truth_poses))
        
        if n == 0:
            return None
        
        # Align trajectories (find best rigid transformation)
        est_positions = np.array([self.estimated_poses[i].position for i in range(n)])
        gt_positions = np.array([self.ground_truth_poses[i].position for i in range(n)])
        
        # Compute alignment using Umeyama method
        aligned_est = self._align_trajectories(est_positions, gt_positions)
        
        # Compute RMSE
        errors = np.linalg.norm(aligned_est - gt_positions, axis=1)
        ate = np.sqrt(np.mean(errors ** 2))
        
        return ate
    
    def compute_rpe(self, delta: int = 1) -> Optional[Tuple[float, float]]:
        """
        Compute Relative Pose Error (RPE).
        
        Args:
            delta: Frame delta for relative error computation
            
        Returns:
            (translation_error, rotation_error) or None if no ground truth
        """
        if len(self.ground_truth_poses) < delta + 1:
            return None
        
        n = min(len(self.estimated_poses), len(self.ground_truth_poses))
        
        trans_errors = []
        rot_errors = []
        
        for i in range(n - delta):
            # Ground truth relative pose
            gt_from = self.ground_truth_poses[i]
            gt_to = self.ground_truth_poses[i + delta]
            gt_rel = gt_from.inverse().compose(gt_to)
            
            # Estimated relative pose
            est_from = self.estimated_poses[i]
            est_to = self.estimated_poses[i + delta]
            est_rel = est_from.inverse().compose(est_to)
            
            # Error
            error_rel = gt_rel.inverse().compose(est_rel)
            
            trans_errors.append(np.linalg.norm(error_rel.position))
            
            # Rotation error
            if isinstance(error_rel.rotation, np.ndarray) and error_rel.rotation.shape == (3, 3):
                angle = np.arccos(np.clip((np.trace(error_rel.rotation) - 1) / 2, -1, 1))
                rot_errors.append(angle)
        
        if not trans_errors:
            return None
        
        rpe_trans = np.sqrt(np.mean(np.array(trans_errors) ** 2))
        rpe_rot = np.sqrt(np.mean(np.array(rot_errors) ** 2)) if rot_errors else 0
        
        return rpe_trans, rpe_rot
    
    def _align_trajectories(
        self,
        estimated: np.ndarray,
        ground_truth: np.ndarray
    ) -> np.ndarray:
        """
        Align estimated trajectory to ground truth using Umeyama method.
        
        Args:
            estimated: [N, 3] estimated positions
            ground_truth: [N, 3] ground truth positions
            
        Returns:
            [N, 3] aligned estimated positions
        """
        # Compute centroids
        est_centroid = np.mean(estimated, axis=0)
        gt_centroid = np.mean(ground_truth, axis=0)
        
        # Center the points
        est_centered = estimated - est_centroid
        gt_centered = ground_truth - gt_centroid
        
        # Compute rotation using SVD
        H = est_centered.T @ gt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute scale
        scale = np.sum(S) / np.sum(est_centered ** 2)
        
        # Compute translation
        t = gt_centroid - scale * R @ est_centroid
        
        # Apply transformation
        aligned = scale * (estimated @ R.T) + t
        
        return aligned
    
    def get_statistics(self) -> Dict:
        """Get SLAM statistics."""
        stats = {
            'frame_count': self.frame_count,
            'keyframe_count': self.keyframe_count,
            'loop_closure_count': self.loop_closure_count,
            'optimization_count': self.optimization_count,
            'tracking_state': self.visual_odometry.get_state().value,
            'total_distance': self.visual_odometry.total_distance,
            'avg_processing_time_ms': self.total_processing_time / max(1, self.frame_count),
        }
        
        # Add evaluation metrics if ground truth available
        ate = self.compute_ate()
        if ate is not None:
            stats['ate_rmse'] = ate
        
        rpe = self.compute_rpe()
        if rpe is not None:
            stats['rpe_translation'] = rpe[0]
            stats['rpe_rotation_rad'] = rpe[1]
        
        # Add pose graph stats
        stats.update(self.pose_graph.get_statistics())
        
        # Add loop closure stats
        stats.update(self.loop_closure.get_statistics())
        
        return stats
    
    def save_results(self, output_dir: str):
        """Save SLAM results to directory."""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trajectory
        trajectory = self.get_trajectory()
        if len(trajectory) > 0:
            np.save(os.path.join(output_dir, 'estimated_trajectory.npy'), trajectory)
        
        gt_trajectory = self.get_ground_truth_trajectory()
        if gt_trajectory is not None and len(gt_trajectory) > 0:
            np.save(os.path.join(output_dir, 'ground_truth_trajectory.npy'), gt_trajectory)
        
        # Save pose graph
        self.pose_graph.save(os.path.join(output_dir, 'pose_graph.json'))
        
        # Save statistics
        stats = self.get_statistics()
        with open(os.path.join(output_dir, 'slam_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"SLAM results saved to {output_dir}")
    
    def reset(self):
        """Reset SLAM system."""
        self.visual_odometry.reset()
        self.loop_closure = LoopClosureDetector(self.config.lc_config)
        self.pose_graph = PoseGraph()
        
        self.keyframes.clear()
        self.current_keyframe_id = -1
        self.last_keyframe_pose = None
        
        self.ground_truth_poses.clear()
        self.estimated_poses.clear()
        self.timestamps.clear()
        
        self.frame_count = 0
        self.keyframe_count = 0
        self.loop_closure_count = 0
        self.optimization_count = 0
        self.total_processing_time = 0.0
