"""
Robust SLAM System
==================

Production-ready SLAM for ObjectNav with multiple correction methods:

1. Dead Reckoning (DR) - Primary, perfect in simulation
2. Visual Odometry (VO) - Frame-to-frame motion
3. Loop Closure - Detect revisited locations
4. Particle Filter - Map-based correction
5. Ground Truth Comparison - Accuracy metrics
6. User Correction - Manual pose fixes

Design principles:
- DR is always the backbone (fast, accurate in sim)
- Other methods provide corrections
- Confidence-weighted fusion
- Graceful degradation
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class LocalizationState(Enum):
    """Localization confidence state."""
    EXCELLENT = "excellent"  # <0.1m error
    GOOD = "good"           # <0.3m error
    UNCERTAIN = "uncertain" # <1.0m error
    LOST = "lost"           # >1.0m error


@dataclass
class PoseEstimate:
    """Pose estimation result with confidence."""
    position: np.ndarray      # [x, y, z]
    yaw: float               # radians
    confidence: float        # 0-1
    state: LocalizationState
    
    # For debugging
    dr_position: np.ndarray = None
    vo_correction: np.ndarray = None
    lc_correction: np.ndarray = None
    gt_error: float = None
    
    def copy(self) -> 'PoseEstimate':
        return PoseEstimate(
            position=self.position.copy(),
            yaw=self.yaw,
            confidence=self.confidence,
            state=self.state,
        )


@dataclass
class LoopClosureCandidate:
    """A candidate location for loop closure."""
    position: np.ndarray
    yaw: float
    descriptor: np.ndarray  # Visual descriptor
    depth_signature: np.ndarray  # Depth pattern
    frame_id: int
    floor_id: int = 0


class RobustSLAM:
    """
    Robust SLAM system combining multiple localization methods.
    
    Usage:
        slam = RobustSLAM()
        slam.initialize(start_position, start_yaw)
        
        # Each frame:
        result = slam.update(
            action='move_forward',
            rgb=rgb_image,
            depth=depth_image,
            gt_pose=(gt_pos, gt_yaw)  # Optional, for metrics
        )
        
        print(f"Position: {result.position}, Confidence: {result.confidence}")
        print(f"GT Error: {result.gt_error}m")
        
        # User correction
        slam.correct_pose(user_position, user_yaw)
    """
    
    # Habitat action parameters
    FORWARD_DIST = 0.25  # meters
    TURN_ANGLE = np.radians(10.0)  # radians
    
    def __init__(
        self,
        use_vo: bool = False,
        use_loop_closure: bool = True,
        use_particle_filter: bool = False,
        camera_K: np.ndarray = None,
    ):
        """
        Initialize SLAM system.
        
        Args:
            use_vo: Enable visual odometry (slower, for real robot prep)
            use_loop_closure: Enable loop closure detection
            use_particle_filter: Enable particle filter (slow but robust)
            camera_K: Camera intrinsic matrix for VO
        """
        self.use_vo = use_vo
        self.use_loop_closure = use_loop_closure
        self.use_particle_filter = use_particle_filter
        self.camera_K = camera_K
        
        # State
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.confidence = 1.0
        self.initialized = False
        self.frame_count = 0
        
        # Dead reckoning accumulator
        self.dr_position = np.zeros(3)
        self.dr_yaw = 0.0
        
        # Visual odometry
        self.vo_enabled = use_vo and camera_K is not None
        self.prev_gray = None
        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.orb = None
        self.matcher = None
        
        if self.vo_enabled:
            self._init_vo()
        
        # Loop closure
        self.lc_candidates: List[LoopClosureCandidate] = []
        self.lc_cooldown = 0  # Frames until next LC check
        self.lc_min_interval = 30  # Min frames between LC
        self.lc_distance_threshold = 1.5  # meters
        
        # Particle filter (optional, for robustness)
        self.pf_enabled = use_particle_filter
        self.particles = None
        self.particle_weights = None
        self.n_particles = 100
        
        # Ground truth tracking
        self.gt_errors: List[float] = []
        self.total_distance = 0.0
        
        # User corrections
        self.user_corrections: List[Tuple[int, np.ndarray, float]] = []
        
        # Timing
        self.timing_stats = {
            'dr': [],
            'vo': [],
            'lc': [],
            'total': [],
        }
    
    def _init_vo(self):
        """Initialize visual odometry components."""
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=4)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def initialize(self, position: np.ndarray, yaw: float):
        """
        Initialize SLAM with known pose.
        
        Args:
            position: [x, y, z] starting position
            yaw: starting yaw in radians
        """
        self.position = position.copy()
        self.yaw = yaw
        self.dr_position = position.copy()
        self.dr_yaw = yaw
        self.confidence = 1.0
        self.initialized = True
        self.frame_count = 0
        
        self.lc_candidates.clear()
        self.gt_errors.clear()
        self.total_distance = 0.0
        
        if self.pf_enabled:
            self._init_particles(position, yaw)
        
        print(f"SLAM initialized at ({position[0]:.2f}, {position[2]:.2f}), yaw={np.degrees(yaw):.1f}°")
    
    def _init_particles(self, position: np.ndarray, yaw: float, spread: float = 0.1):
        """Initialize particle filter around position."""
        self.particles = np.zeros((self.n_particles, 4))  # x, y, z, yaw
        self.particles[:, :3] = position + np.random.randn(self.n_particles, 3) * spread
        self.particles[:, 3] = yaw + np.random.randn(self.n_particles) * 0.1
        self.particle_weights = np.ones(self.n_particles) / self.n_particles
    
    def update(
        self,
        action: Optional[str],
        rgb: np.ndarray = None,
        depth: np.ndarray = None,
        gt_pose: Tuple[np.ndarray, float] = None,
        occupancy_map: np.ndarray = None,
        map_coords = None,  # (world_to_map function, origin, resolution)
    ) -> PoseEstimate:
        """
        Update SLAM with new observation.
        
        Args:
            action: 'move_forward', 'turn_left', 'turn_right', or None
            rgb: RGB image (for VO and loop closure)
            depth: Depth image (for loop closure and PF)
            gt_pose: Optional (position, yaw) ground truth for metrics
            occupancy_map: Optional map for particle filter
            map_coords: Optional coordinate conversion for PF
            
        Returns:
            PoseEstimate with position, confidence, and debug info
        """
        if not self.initialized:
            return PoseEstimate(
                position=np.zeros(3),
                yaw=0.0,
                confidence=0.0,
                state=LocalizationState.LOST
            )
        
        t_start = time.time()
        self.frame_count += 1
        
        # 1. Dead Reckoning (always)
        t_dr = time.time()
        self._apply_dead_reckoning(action)
        self.timing_stats['dr'].append(time.time() - t_dr)
        
        # 2. Visual Odometry (optional)
        vo_correction = None
        if self.vo_enabled and rgb is not None and depth is not None:
            t_vo = time.time()
            vo_correction = self._compute_vo(rgb, depth)
            self.timing_stats['vo'].append(time.time() - t_vo)
        
        # 3. Loop Closure (optional)
        lc_correction = None
        if self.use_loop_closure and rgb is not None and depth is not None:
            t_lc = time.time()
            lc_correction = self._check_loop_closure(rgb, depth)
            self.timing_stats['lc'].append(time.time() - t_lc)
        
        # 4. Particle Filter Update (optional)
        if self.pf_enabled and depth is not None and occupancy_map is not None:
            self._update_particle_filter(action, depth, occupancy_map, map_coords)
        
        # 5. Fuse estimates
        self._fuse_estimates(vo_correction, lc_correction)
        
        # 6. Ground truth comparison
        gt_error = None
        if gt_pose is not None:
            gt_pos, gt_yaw = gt_pose
            gt_error = np.linalg.norm(self.position[[0, 2]] - gt_pos[[0, 2]])
            self.gt_errors.append(gt_error)
        
        # Track distance
        if action == 'move_forward':
            self.total_distance += self.FORWARD_DIST
        
        # Determine state
        if self.confidence > 0.8:
            state = LocalizationState.EXCELLENT
        elif self.confidence > 0.5:
            state = LocalizationState.GOOD
        elif self.confidence > 0.2:
            state = LocalizationState.UNCERTAIN
        else:
            state = LocalizationState.LOST
        
        self.timing_stats['total'].append(time.time() - t_start)
        
        return PoseEstimate(
            position=self.position.copy(),
            yaw=self.yaw,
            confidence=self.confidence,
            state=state,
            dr_position=self.dr_position.copy(),
            vo_correction=vo_correction,
            lc_correction=lc_correction,
            gt_error=gt_error,
        )
    
    def _apply_dead_reckoning(self, action: Optional[str]):
        """Apply action to dead reckoning estimate."""
        if action is None:
            return
        
        if action == 'move_forward':
            # At yaw=0, forward is -Z direction
            dx = -np.sin(self.dr_yaw) * self.FORWARD_DIST
            dz = -np.cos(self.dr_yaw) * self.FORWARD_DIST
            self.dr_position[0] += dx
            self.dr_position[2] += dz
            
        elif action == 'turn_left':
            self.dr_yaw += self.TURN_ANGLE
            
        elif action == 'turn_right':
            self.dr_yaw -= self.TURN_ANGLE
        
        # Normalize yaw
        while self.dr_yaw > np.pi:
            self.dr_yaw -= 2 * np.pi
        while self.dr_yaw < -np.pi:
            self.dr_yaw += 2 * np.pi
        
        # DR is primary estimate in simulation
        self.position = self.dr_position.copy()
        self.yaw = self.dr_yaw
    
    def _compute_vo(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute visual odometry correction.
        
        Returns:
            [dx, dz, dyaw] correction or None
        """
        if self.orb is None:
            return None
        
        # Convert to grayscale
        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        
        # Detect features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if self.prev_gray is None or descriptors is None or len(keypoints) < 20:
            self.prev_gray = gray
            self.prev_depth = depth.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
        
        if self.prev_descriptors is None:
            self.prev_gray = gray
            self.prev_depth = depth.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
        
        # Match features
        try:
            matches = self.matcher.match(self.prev_descriptors, descriptors)
        except cv2.error:
            self.prev_gray = gray
            self.prev_depth = depth.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
        
        if len(matches) < 15:
            self.prev_gray = gray
            self.prev_depth = depth.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        # Get 3D-2D correspondences
        pts_3d = []
        pts_2d = []
        
        for m in matches:
            pt_prev = self.prev_keypoints[m.queryIdx].pt
            pt_curr = keypoints[m.trainIdx].pt
            
            px, py = int(pt_prev[0]), int(pt_prev[1])
            
            if 0 <= px < self.prev_depth.shape[1] and 0 <= py < self.prev_depth.shape[0]:
                d = self.prev_depth[py, px]
                if 0.1 < d < 5.0:
                    # Backproject
                    x = (px - self.camera_K[0, 2]) * d / self.camera_K[0, 0]
                    y = (py - self.camera_K[1, 2]) * d / self.camera_K[1, 1]
                    pts_3d.append([x, y, d])
                    pts_2d.append(pt_curr)
        
        if len(pts_3d) < 8:
            self.prev_gray = gray
            self.prev_depth = depth.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
        
        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)
        
        # PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.camera_K, None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=3.0,
            confidence=0.99,
        )
        
        correction = None
        
        if success and inliers is not None and len(inliers) > 8:
            t = tvec.flatten()
            
            # Sanity check
            if np.linalg.norm(t) < 0.5:
                # Camera motion to world motion
                # Simplified: assume mostly forward/turn motion
                correction = np.array([t[0], t[2], 0])  # dx, dz, dyaw
        
        self.prev_gray = gray
        self.prev_depth = depth.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return correction
    
    def _check_loop_closure(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[np.ndarray]:
        """
        Check for loop closure.
        
        Returns:
            [dx, dz, dyaw] correction or None
        """
        self.lc_cooldown = max(0, self.lc_cooldown - 1)
        
        # Compute descriptors for current frame
        depth_sig = self._compute_depth_signature(depth)
        
        # Store candidate every N frames
        if self.frame_count % 20 == 0:
            self.lc_candidates.append(LoopClosureCandidate(
                position=self.position.copy(),
                yaw=self.yaw,
                descriptor=None,  # Could add ORB-BOW here
                depth_signature=depth_sig,
                frame_id=self.frame_count,
            ))
            
            # Limit storage
            if len(self.lc_candidates) > 200:
                self.lc_candidates.pop(0)
        
        # Check for loop closure
        if self.lc_cooldown > 0 or len(self.lc_candidates) < 10:
            return None
        
        # Find candidates near current position
        for candidate in self.lc_candidates[:-10]:  # Skip recent
            dist = np.linalg.norm(
                self.position[[0, 2]] - candidate.position[[0, 2]]
            )
            
            if dist > self.lc_distance_threshold:
                continue
            
            # Compare depth signatures
            sig_diff = np.mean(np.abs(depth_sig - candidate.depth_signature))
            
            if sig_diff < 0.3:  # Similar depth pattern
                # Loop closure detected!
                correction = candidate.position - self.position
                
                print(f"Loop closure detected! Correction: {np.linalg.norm(correction):.3f}m")
                
                self.lc_cooldown = self.lc_min_interval
                self.confidence = min(1.0, self.confidence + 0.2)
                
                return correction[[0, 2, 2]]  # dx, dz, 0
        
        return None
    
    def _compute_depth_signature(self, depth: np.ndarray) -> np.ndarray:
        """Compute compact depth signature for loop closure."""
        h, w = depth.shape
        
        # Take horizontal slice at middle, downsample
        slice_depth = depth[h//2, ::max(1, w//16)][:16]
        
        # Normalize
        valid = slice_depth[(slice_depth > 0.1) & (slice_depth < 10)]
        if len(valid) < 4:
            return np.zeros(16)
        
        signature = np.clip(slice_depth / 5.0, 0, 2)
        return signature
    
    def _update_particle_filter(
        self,
        action: Optional[str],
        depth: np.ndarray,
        occupancy_map: np.ndarray,
        map_coords,
    ):
        """Update particle filter with observation."""
        if self.particles is None:
            return
        
        # Motion update
        if action == 'move_forward':
            noise = np.random.randn(self.n_particles) * 0.02
            self.particles[:, 0] -= np.sin(self.particles[:, 3]) * (self.FORWARD_DIST + noise)
            self.particles[:, 2] -= np.cos(self.particles[:, 3]) * (self.FORWARD_DIST + noise)
        elif action == 'turn_left':
            self.particles[:, 3] += self.TURN_ANGLE + np.random.randn(self.n_particles) * 0.02
        elif action == 'turn_right':
            self.particles[:, 3] -= self.TURN_ANGLE + np.random.randn(self.n_particles) * 0.02
        
        # Observation update (simplified: check if particles are in valid locations)
        if map_coords is not None:
            world_to_map, origin_x, origin_z, resolution, map_size = map_coords
            
            for i in range(self.n_particles):
                mx, my = world_to_map(self.particles[i, 0], self.particles[i, 2])
                
                if 0 <= mx < map_size and 0 <= my < map_size:
                    if occupancy_map[my, mx] > 0.5:
                        self.particle_weights[i] *= 0.1  # In obstacle
                    else:
                        self.particle_weights[i] *= 1.0  # Valid
                else:
                    self.particle_weights[i] *= 0.5  # Out of bounds
        
        # Normalize weights
        total = np.sum(self.particle_weights)
        if total > 0:
            self.particle_weights /= total
        else:
            self.particle_weights = np.ones(self.n_particles) / self.n_particles
        
        # Resample if effective particles low
        n_eff = 1.0 / np.sum(self.particle_weights ** 2)
        if n_eff < self.n_particles / 2:
            indices = np.random.choice(
                self.n_particles, self.n_particles,
                p=self.particle_weights
            )
            self.particles = self.particles[indices]
            self.particle_weights = np.ones(self.n_particles) / self.n_particles
    
    def _fuse_estimates(
        self,
        vo_correction: Optional[np.ndarray],
        lc_correction: Optional[np.ndarray],
    ):
        """Fuse different estimates with confidence weighting."""
        # Start with DR (already applied)
        
        # Apply VO correction (small weight, for validation)
        if vo_correction is not None:
            alpha = 0.1  # Low weight for VO
            self.position[0] += alpha * vo_correction[0]
            self.position[2] += alpha * vo_correction[1]
        
        # Apply loop closure correction (higher weight)
        if lc_correction is not None:
            alpha = 0.5  # Significant correction
            self.position[0] += alpha * lc_correction[0]
            self.position[2] += alpha * lc_correction[1]
            
            # Re-sync DR to corrected position
            self.dr_position = self.position.copy()
        
        # Apply particle filter estimate (if enabled)
        if self.pf_enabled and self.particles is not None:
            pf_pos = np.average(self.particles[:, :3], weights=self.particle_weights, axis=0)
            pf_yaw = np.arctan2(
                np.average(np.sin(self.particles[:, 3]), weights=self.particle_weights),
                np.average(np.cos(self.particles[:, 3]), weights=self.particle_weights)
            )
            
            # Blend with current estimate
            alpha = 0.2
            self.position = (1 - alpha) * self.position + alpha * pf_pos
            
            yaw_diff = pf_yaw - self.yaw
            while yaw_diff > np.pi: yaw_diff -= 2*np.pi
            while yaw_diff < -np.pi: yaw_diff += 2*np.pi
            self.yaw += alpha * yaw_diff
        
        # Update confidence based on corrections
        if lc_correction is not None:
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.confidence = max(0.3, self.confidence - 0.001)  # Slow decay
    
    def correct_pose(self, position: np.ndarray, yaw: float = None):
        """
        User-provided pose correction.
        
        Args:
            position: Corrected [x, y, z] position
            yaw: Corrected yaw (optional)
        """
        print(f"User correction: ({position[0]:.2f}, {position[2]:.2f})")
        
        self.position = position.copy()
        self.dr_position = position.copy()
        
        if yaw is not None:
            self.yaw = yaw
            self.dr_yaw = yaw
        
        self.confidence = 1.0
        
        # Re-initialize particles around corrected position
        if self.pf_enabled:
            self._init_particles(position, self.yaw, spread=0.2)
        
        self.user_corrections.append((self.frame_count, position.copy(), self.yaw))
    
    def get_pose(self) -> Tuple[np.ndarray, float]:
        """Get current pose estimate."""
        return self.position.copy(), self.yaw
    
    def get_stats(self) -> Dict:
        """Get SLAM statistics."""
        stats = {
            'frame_count': self.frame_count,
            'total_distance': self.total_distance,
            'confidence': self.confidence,
            'lc_candidates': len(self.lc_candidates),
            'user_corrections': len(self.user_corrections),
        }
        
        if self.gt_errors:
            stats['gt_error_mean'] = np.mean(self.gt_errors)
            stats['gt_error_max'] = np.max(self.gt_errors)
            stats['gt_error_current'] = self.gt_errors[-1]
            stats['drift_rate'] = (
                self.gt_errors[-1] / self.total_distance * 100
                if self.total_distance > 0 else 0
            )
        
        # Timing
        for key, times in self.timing_stats.items():
            if times:
                stats[f'time_{key}_ms'] = np.mean(times[-100:]) * 1000
        
        return stats
    
    def reset(self):
        """Reset SLAM state."""
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.dr_position = np.zeros(3)
        self.dr_yaw = 0.0
        self.confidence = 1.0
        self.initialized = False
        self.frame_count = 0
        
        self.prev_gray = None
        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        self.lc_candidates.clear()
        self.gt_errors.clear()
        self.user_corrections.clear()
        self.total_distance = 0.0
        
        if self.pf_enabled:
            self.particles = None
            self.particle_weights = None
        
        for key in self.timing_stats:
            self.timing_stats[key].clear()
