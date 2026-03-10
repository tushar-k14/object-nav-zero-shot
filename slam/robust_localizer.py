"""
Robust Localization for Navigation
==================================

Instead of pure visual odometry (which drifts badly), we use:

1. **Motion Model**: Predict position from actions (dead reckoning)
2. **Depth Matching**: Compare observed depth with expected depth from map
3. **Particle Filter**: Maintain multiple pose hypotheses

This is much more robust for navigation on pre-built maps.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class LocalizationState(Enum):
    OK = "ok"
    UNCERTAIN = "uncertain"
    LOST = "lost"


@dataclass
class LocalizationResult:
    """Localization result."""
    position: np.ndarray      # [x, y, z]
    yaw: float                # radians
    confidence: float         # 0-1
    state: LocalizationState


class ActionModel:
    """
    Simple motion model based on known actions.
    
    Habitat actions:
    - move_forward: 0.25m forward
    - turn_left: 10 degrees CCW (positive yaw)
    - turn_right: 10 degrees CW (negative yaw)
    
    Habitat coordinate system:
    - Y is up
    - At yaw=0, agent faces -Z direction
    - Positive yaw = counter-clockwise when viewed from above
    """
    
    # Action parameters
    FORWARD_DIST = 0.25   # meters
    TURN_ANGLE = np.radians(10.0)  # radians
    
    # Noise parameters (for particle filter)
    FORWARD_NOISE = 0.02  # std dev in meters
    TURN_NOISE = np.radians(2.0)  # std dev in radians
    
    @staticmethod
    def apply_action(
        position: np.ndarray,
        yaw: float,
        action: str,
        add_noise: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Apply action to pose.
        
        Args:
            position: [x, y, z] current position
            yaw: current yaw angle (radians), 0 = facing -Z
            action: 'move_forward', 'turn_left', 'turn_right'
            add_noise: add Gaussian noise for particle filter
            
        Returns:
            (new_position, new_yaw)
        """
        new_pos = position.copy()
        new_yaw = yaw
        
        if action == 'move_forward':
            dist = ActionModel.FORWARD_DIST
            if add_noise:
                dist += np.random.normal(0, ActionModel.FORWARD_NOISE)
            
            # Move in direction of yaw
            # At yaw=0, forward is -Z direction
            # dx = -sin(yaw) * dist  -- NO, this was wrong
            # dz = -cos(yaw) * dist
            #
            # Actually in Habitat:
            # - yaw=0: facing -Z, so dz = -dist, dx = 0
            # - yaw=90° (pi/2): facing -X, so dx = -dist, dz = 0
            # - yaw=-90° (-pi/2): facing +X, so dx = +dist, dz = 0
            #
            # This means:
            # dx = -sin(yaw) * dist
            # dz = -cos(yaw) * dist
            new_pos[0] -= np.sin(yaw) * dist  # X
            new_pos[2] -= np.cos(yaw) * dist  # Z
            
        elif action == 'turn_left':
            # Turn left = counter-clockwise = positive yaw change
            angle = ActionModel.TURN_ANGLE
            if add_noise:
                angle += np.random.normal(0, ActionModel.TURN_NOISE)
            new_yaw = yaw + angle
            
        elif action == 'turn_right':
            # Turn right = clockwise = negative yaw change
            angle = ActionModel.TURN_ANGLE
            if add_noise:
                angle += np.random.normal(0, ActionModel.TURN_NOISE)
            new_yaw = yaw - angle
        
        # Normalize yaw to [-pi, pi]
        while new_yaw > np.pi:
            new_yaw -= 2 * np.pi
        while new_yaw < -np.pi:
            new_yaw += 2 * np.pi
        
        return new_pos, new_yaw


class DepthMapMatcher:
    """
    Match observed depth to expected depth from occupancy map.
    
    Uses ray casting on the 2D occupancy map to predict what
    the depth sensor should see, then compares with actual depth.
    """
    
    def __init__(
        self,
        occupancy_map: np.ndarray,
        resolution_cm: float,
        origin_x: float,
        origin_z: float,
        map_size: int,
        # Camera params
        width: int = 640,
        height: int = 480,
        hfov: float = 90.0,
        max_range: float = 5.0,
    ):
        self.occupancy_map = occupancy_map  # 1 = obstacle, 0 = free
        self.resolution = resolution_cm
        self.origin_x = origin_x
        self.origin_z = origin_z
        self.map_size = map_size
        
        self.width = width
        self.height = height
        self.hfov_rad = np.radians(hfov)
        self.max_range = max_range
        
        # Precompute ray angles for horizontal slice
        self.n_rays = 32  # Number of rays to cast
        half_fov = self.hfov_rad / 2
        self.ray_angles = np.linspace(-half_fov, half_fov, self.n_rays)
    
    def world_to_map(self, wx: float, wz: float) -> Tuple[int, int]:
        """Convert world to map coordinates."""
        mx = int((wx - self.origin_x) * 100 / self.resolution + self.map_size // 2)
        my = int((wz - self.origin_z) * 100 / self.resolution + self.map_size // 2)
        return mx, my
    
    def cast_rays(self, position: np.ndarray, yaw: float) -> np.ndarray:
        """
        Cast rays from position and return expected depths.
        
        Args:
            position: [x, y, z] world position
            yaw: heading in radians
            
        Returns:
            [n_rays] expected depths in meters
        """
        expected_depths = np.full(self.n_rays, self.max_range)
        
        wx, wz = position[0], position[2]
        
        for i, angle in enumerate(self.ray_angles):
            # Ray direction in world frame
            ray_yaw = yaw + angle
            dx = np.sin(ray_yaw)
            dz = -np.cos(ray_yaw)
            
            # Ray march
            for dist in np.arange(0.1, self.max_range, 0.05):  # 5cm steps
                rx = wx + dx * dist
                rz = wz + dz * dist
                
                mx, my = self.world_to_map(rx, rz)
                
                # Check bounds
                if mx < 0 or mx >= self.map_size or my < 0 or my >= self.map_size:
                    expected_depths[i] = dist
                    break
                
                # Check obstacle
                if self.occupancy_map[my, mx] > 0:
                    expected_depths[i] = dist
                    break
        
        return expected_depths
    
    def extract_depth_slice(self, depth: np.ndarray) -> np.ndarray:
        """
        Extract horizontal depth slice from depth image.
        
        Takes the middle row and samples n_rays points.
        """
        h, w = depth.shape
        mid_row = depth[h // 2, :]  # Middle row
        
        # Sample to match n_rays
        indices = np.linspace(0, w - 1, self.n_rays).astype(int)
        observed = mid_row[indices]
        
        # Filter invalid depths
        observed[observed <= 0.1] = self.max_range
        observed[observed > self.max_range] = self.max_range
        
        return observed
    
    def compute_likelihood(
        self,
        position: np.ndarray,
        yaw: float,
        observed_depth: np.ndarray
    ) -> float:
        """
        Compute likelihood of pose given observed depth.
        
        Args:
            position: hypothesized position
            yaw: hypothesized yaw
            observed_depth: [n_rays] observed depths
            
        Returns:
            Likelihood score (higher = better match)
        """
        expected = self.cast_rays(position, yaw)
        
        # Compute error
        diff = np.abs(expected - observed_depth)
        
        # Use robust error (ignore outliers)
        diff = np.clip(diff, 0, 2.0)  # Cap at 2m
        
        # Gaussian likelihood
        sigma = 0.3  # meters
        likelihood = np.exp(-0.5 * (diff / sigma) ** 2)
        
        return np.mean(likelihood)


class ParticleFilterLocalizer:
    """
    Particle filter for robust localization.
    
    Maintains multiple pose hypotheses and weights them
    by how well they match the observed depth.
    """
    
    def __init__(
        self,
        occupancy_map: np.ndarray,
        resolution_cm: float,
        origin_x: float,
        origin_z: float,
        map_size: int,
        n_particles: int = 100,
        # Camera params
        width: int = 640,
        height: int = 480,
        hfov: float = 90.0,
    ):
        self.n_particles = n_particles
        
        # Depth matcher
        self.depth_matcher = DepthMapMatcher(
            occupancy_map, resolution_cm, origin_x, origin_z, map_size,
            width, height, hfov
        )
        
        # Particles: [n_particles, 4] = [x, y, z, yaw]
        self.particles = None
        self.weights = None
        
        # State
        self.initialized = False
        self.state = LocalizationState.LOST
        
        # Best estimate
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.confidence = 0.0
    
    def initialize(self, position: np.ndarray, yaw: float, spread: float = 0.5):
        """
        Initialize particles around a starting pose.
        
        Args:
            position: [x, y, z] starting position
            yaw: starting yaw
            spread: initial spread in meters
        """
        self.particles = np.zeros((self.n_particles, 4))
        
        # Add noise around initial pose
        self.particles[:, 0] = position[0] + np.random.normal(0, spread, self.n_particles)
        self.particles[:, 1] = position[1]  # Y stays constant
        self.particles[:, 2] = position[2] + np.random.normal(0, spread, self.n_particles)
        self.particles[:, 3] = yaw + np.random.normal(0, 0.2, self.n_particles)
        
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        self.position = position.copy()
        self.yaw = yaw
        self.confidence = 1.0
        self.initialized = True
        self.state = LocalizationState.OK
    
    def update(
        self,
        action: Optional[str],
        depth: np.ndarray
    ) -> LocalizationResult:
        """
        Update localization with action and observation.
        
        Args:
            action: action taken ('move_forward', 'turn_left', 'turn_right', or None)
            depth: observed depth image
            
        Returns:
            LocalizationResult
        """
        if not self.initialized:
            return LocalizationResult(
                position=self.position,
                yaw=self.yaw,
                confidence=0.0,
                state=LocalizationState.LOST
            )
        
        # 1. Motion update (prediction)
        if action is not None:
            for i in range(self.n_particles):
                pos = self.particles[i, :3]
                yaw = self.particles[i, 3]
                new_pos, new_yaw = ActionModel.apply_action(pos, yaw, action, add_noise=True)
                self.particles[i, :3] = new_pos
                self.particles[i, 3] = new_yaw
        
        # 2. Measurement update (correction)
        observed_depth = self.depth_matcher.extract_depth_slice(depth)
        
        for i in range(self.n_particles):
            pos = self.particles[i, :3]
            yaw = self.particles[i, 3]
            likelihood = self.depth_matcher.compute_likelihood(pos, yaw, observed_depth)
            self.weights[i] *= likelihood
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-10:
            self.weights /= weight_sum
        else:
            # All particles have zero weight - lost!
            self.weights = np.ones(self.n_particles) / self.n_particles
            self.state = LocalizationState.LOST
        
        # 3. Estimate pose (weighted mean)
        self.position = np.average(self.particles[:, :3], weights=self.weights, axis=0)
        
        # Circular mean for yaw
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 3]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 3]))
        self.yaw = np.arctan2(sin_sum, cos_sum)
        
        # 4. Compute confidence (effective number of particles)
        n_eff = 1.0 / np.sum(self.weights ** 2)
        self.confidence = n_eff / self.n_particles
        
        # Update state
        if self.confidence > 0.5:
            self.state = LocalizationState.OK
        elif self.confidence > 0.2:
            self.state = LocalizationState.UNCERTAIN
        else:
            self.state = LocalizationState.LOST
        
        # 5. Resample if needed
        if n_eff < self.n_particles / 2:
            self._resample()
        
        return LocalizationResult(
            position=self.position.copy(),
            yaw=self.yaw,
            confidence=self.confidence,
            state=self.state
        )
    
    def _resample(self):
        """Resample particles based on weights."""
        indices = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            p=self.weights
        )
        self.particles = self.particles[indices].copy()
        
        # Add small noise to avoid particle depletion
        self.particles[:, 0] += np.random.normal(0, 0.02, self.n_particles)
        self.particles[:, 2] += np.random.normal(0, 0.02, self.n_particles)
        self.particles[:, 3] += np.random.normal(0, 0.02, self.n_particles)
        
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def get_pose(self) -> Tuple[np.ndarray, float, float]:
        """Get current pose estimate."""
        return self.position.copy(), self.yaw, self.confidence


class HybridLocalizer:
    """
    Hybrid localization combining:
    1. Action-based dead reckoning (fast, drifts)
    2. Particle filter correction (robust, slower)
    3. Loop closure for drift correction on revisits
    
    Uses dead reckoning between corrections, with periodic
    particle filter updates to correct drift.
    """
    
    def __init__(
        self,
        occupancy_map: np.ndarray,
        resolution_cm: float,
        origin_x: float,
        origin_z: float,
        map_size: int,
        correction_interval: int = 5,  # Correct every N frames
        n_particles: int = 50,
    ):
        # Particle filter
        self.pf = ParticleFilterLocalizer(
            occupancy_map, resolution_cm, origin_x, origin_z, map_size,
            n_particles=n_particles
        )
        
        self.correction_interval = correction_interval
        self.frame_count = 0
        
        # Current estimate (dead reckoning between corrections)
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.confidence = 0.0
        
        # Last action for motion prediction
        self.last_action = None
        
        self.initialized = False
        
        # Loop closure: store visited locations
        self.visited_locations: List[Tuple[np.ndarray, float, np.ndarray]] = []
        self.loop_closure_threshold = 1.0  # meters
        self.loop_closure_cooldown = 50    # frames between loop closures
        self.frames_since_loop_closure = 0
    
    def initialize(self, position: np.ndarray, yaw: float):
        """Initialize at known pose."""
        self.position = position.copy()
        self.yaw = yaw
        self.pf.initialize(position, yaw, spread=0.3)
        self.initialized = True
        self.confidence = 1.0
        self.visited_locations.clear()
    
    def _compute_depth_signature(self, depth: np.ndarray) -> np.ndarray:
        """Compute a simple signature from depth for loop closure."""
        h, w = depth.shape
        slice_depth = depth[h//2, ::max(1, w//16)][:16]
        signature = np.clip(slice_depth / 5.0, 0, 2)
        return signature
    
    def _check_loop_closure(self, depth: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Check if we've revisited a previous location."""
        if self.frames_since_loop_closure < self.loop_closure_cooldown:
            return None
        
        current_sig = self._compute_depth_signature(depth)
        if np.sum(current_sig) < 0.1:
            return None
        
        for stored_pos, stored_yaw, stored_sig in self.visited_locations:
            dist = np.linalg.norm(self.position[[0, 2]] - stored_pos[[0, 2]])
            if dist > self.loop_closure_threshold:
                continue
            
            sig_diff = np.mean(np.abs(current_sig - stored_sig))
            if sig_diff < 0.3:
                return stored_pos, stored_yaw
        
        return None
    
    def update(
        self,
        action: Optional[str],
        depth: np.ndarray
    ) -> LocalizationResult:
        """Update localization."""
        if not self.initialized:
            return LocalizationResult(
                position=self.position,
                yaw=self.yaw,
                confidence=0.0,
                state=LocalizationState.LOST
            )
        
        self.frame_count += 1
        self.frames_since_loop_closure += 1
        
        # Dead reckoning
        if action is not None:
            self.position, self.yaw = ActionModel.apply_action(
                self.position, self.yaw, action, add_noise=False
            )
        
        # Check for loop closure
        loop_result = self._check_loop_closure(depth)
        if loop_result is not None:
            stored_pos, stored_yaw = loop_result
            correction = np.linalg.norm(self.position - stored_pos)
            print(f"LOOP CLOSURE! Correcting by {correction:.2f}m")
            
            alpha = 0.7
            self.position = (1 - alpha) * self.position + alpha * stored_pos
            
            yaw_diff = stored_yaw - self.yaw
            while yaw_diff > np.pi: yaw_diff -= 2*np.pi
            while yaw_diff < -np.pi: yaw_diff += 2*np.pi
            self.yaw = self.yaw + alpha * yaw_diff
            
            self.pf.initialize(self.position, self.yaw, spread=0.2)
            self.confidence = 0.9
            self.frames_since_loop_closure = 0
        
        # Periodic PF correction
        elif self.frame_count % self.correction_interval == 0:
            result = self.pf.update(action, depth)
            
            if result.confidence > 0.3:
                alpha = min(result.confidence, 0.5)
                self.position = (1 - alpha) * self.position + alpha * result.position
                
                diff = result.yaw - self.yaw
                while diff > np.pi: diff -= 2*np.pi
                while diff < -np.pi: diff += 2*np.pi
                self.yaw = self.yaw + alpha * diff
                
                self.confidence = result.confidence
            else:
                self.confidence *= 0.95
        else:
            if action is not None:
                for i in range(self.pf.n_particles):
                    pos = self.pf.particles[i, :3]
                    yaw = self.pf.particles[i, 3]
                    new_pos, new_yaw = ActionModel.apply_action(pos, yaw, action, add_noise=True)
                    self.pf.particles[i, :3] = new_pos
                    self.pf.particles[i, 3] = new_yaw
        
        # Store location for loop closure
        if self.frame_count % 20 == 0:
            sig = self._compute_depth_signature(depth)
            if np.sum(sig) > 0.1:
                self.visited_locations.append((self.position.copy(), self.yaw, sig))
                if len(self.visited_locations) > 200:
                    self.visited_locations.pop(0)
        
        # Determine state
        if self.confidence > 0.5:
            state = LocalizationState.OK
        elif self.confidence > 0.2:
            state = LocalizationState.UNCERTAIN
        else:
            state = LocalizationState.LOST
        
        return LocalizationResult(
            position=self.position.copy(),
            yaw=self.yaw,
            confidence=self.confidence,
            state=state
        )
    
    def get_pose(self) -> Tuple[np.ndarray, float]:
        """Get current pose."""
        return self.position.copy(), self.yaw