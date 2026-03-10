"""
Robust SLAM V2
==============

Fixed version with proper floor handling.

Key insight: Dead reckoning CANNOT track vertical movement (stairs).
We need to detect floor changes and re-sync the Y coordinate.

For 2D navigation, we only care about X-Z plane accuracy.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
from collections import deque


class LocalizationState(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    UNCERTAIN = "uncertain"
    LOST = "lost"


@dataclass 
class SLAMResult:
    """SLAM estimation result."""
    position: np.ndarray      # [x, y, z]
    yaw: float               # radians
    confidence: float        # 0-1
    state: LocalizationState
    current_floor: int
    
    # Errors (if GT provided)
    error_2d: Optional[float] = None  # X-Z plane error
    error_y: Optional[float] = None   # Height error
    error_yaw: Optional[float] = None # Yaw error (degrees)


class RobustSLAMv2:
    """
    Robust SLAM with floor change handling.
    
    Key features:
    1. Dead reckoning for X-Z movement (perfect in sim)
    2. Floor change detection from Y coordinate
    3. Y coordinate synced on floor change
    4. Loop closure for long-term drift
    5. GT comparison for validation
    
    Usage:
        slam = RobustSLAMv2()
        
        # Initialize with GT
        slam.initialize(gt_position, gt_yaw)
        
        # Each step:
        # 1. Execute action in simulator
        # 2. Update SLAM with action
        # 3. Optionally provide new GT for comparison
        
        result = slam.update(action='move_forward', gt_pose=(gt_pos, gt_yaw))
    """
    
    # Habitat parameters
    FORWARD_DIST = 0.25
    TURN_ANGLE = np.radians(10.0)
    
    # Floor detection
    FLOOR_CHANGE_THRESHOLD = 0.3  # Y change to detect floor transition
    FLOOR_HEIGHT = 2.5  # Approximate floor height
    
    def __init__(
        self,
        enable_loop_closure: bool = True,
        enable_vo: bool = False,
    ):
        # Pose state
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.initialized = False
        
        # Floor tracking
        self.current_floor = 0
        self.floor_heights: Dict[int, float] = {}  # floor_id -> base Y
        self.prev_y = 0.0
        
        # Loop closure
        self.enable_lc = enable_loop_closure
        self.lc_candidates: List[Tuple[np.ndarray, float, np.ndarray]] = []  # (pos, yaw, depth_sig)
        self.lc_cooldown = 0
        
        # Statistics
        self.frame_count = 0
        self.total_distance = 0.0
        self.floor_changes = 0
        self.gt_errors_2d: List[float] = []
        
        # Confidence
        self.confidence = 1.0
    
    def initialize(self, position: np.ndarray, yaw: float):
        """Initialize with known pose."""
        self.position = position.copy()
        self.yaw = yaw
        self.prev_y = position[1]
        self.initialized = True
        self.frame_count = 0
        self.total_distance = 0.0
        self.floor_changes = 0
        self.confidence = 1.0
        
        # Set initial floor
        self.current_floor = 0
        self.floor_heights = {0: position[1]}
        
        self.lc_candidates.clear()
        self.gt_errors_2d.clear()
        
        print(f"SLAM initialized: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), yaw={np.degrees(yaw):.1f}°")
    
    def update(
        self,
        action: Optional[str],
        gt_pose: Tuple[np.ndarray, float] = None,
        depth: np.ndarray = None,
    ) -> SLAMResult:
        """
        Update pose estimate after action execution.
        
        IMPORTANT: Call this AFTER executing the action in simulator.
        
        Args:
            action: action that was just executed
            gt_pose: optional (position, yaw) for validation
            depth: optional depth image for loop closure
            
        Returns:
            SLAMResult
        """
        if not self.initialized:
            return SLAMResult(
                position=np.zeros(3), yaw=0.0, confidence=0.0,
                state=LocalizationState.LOST, current_floor=0
            )
        
        self.frame_count += 1
        
        # 1. Apply dead reckoning (X-Z only)
        self._apply_action(action)
        
        # 2. Check floor change using GT (if available)
        if gt_pose is not None:
            gt_pos, gt_yaw = gt_pose
            self._check_floor_change(gt_pos[1])
        
        # 3. Loop closure check
        if self.enable_lc and depth is not None:
            self._check_loop_closure(depth)
        
        # 4. Compute errors
        error_2d = None
        error_y = None
        error_yaw = None
        
        if gt_pose is not None:
            gt_pos, gt_yaw = gt_pose
            error_2d = np.sqrt(
                (self.position[0] - gt_pos[0])**2 + 
                (self.position[2] - gt_pos[2])**2
            )
            error_y = abs(self.position[1] - gt_pos[1])
            
            yaw_diff = gt_yaw - self.yaw
            while yaw_diff > np.pi: yaw_diff -= 2*np.pi
            while yaw_diff < -np.pi: yaw_diff += 2*np.pi
            error_yaw = np.degrees(abs(yaw_diff))
            
            self.gt_errors_2d.append(error_2d)
        
        # 5. Determine state
        if error_2d is not None:
            if error_2d < 0.1:
                state = LocalizationState.EXCELLENT
            elif error_2d < 0.3:
                state = LocalizationState.GOOD
            elif error_2d < 1.0:
                state = LocalizationState.UNCERTAIN
            else:
                state = LocalizationState.LOST
        else:
            state = LocalizationState.GOOD if self.confidence > 0.5 else LocalizationState.UNCERTAIN
        
        return SLAMResult(
            position=self.position.copy(),
            yaw=self.yaw,
            confidence=self.confidence,
            state=state,
            current_floor=self.current_floor,
            error_2d=error_2d,
            error_y=error_y,
            error_yaw=error_yaw,
        )
    
    def _apply_action(self, action: Optional[str]):
        """Apply action to pose estimate."""
        if action is None:
            return
        
        if action == 'move_forward':
            # At yaw=0, forward is -Z
            self.position[0] -= np.sin(self.yaw) * self.FORWARD_DIST
            self.position[2] -= np.cos(self.yaw) * self.FORWARD_DIST
            self.total_distance += self.FORWARD_DIST
            
        elif action == 'turn_left':
            self.yaw += self.TURN_ANGLE
            
        elif action == 'turn_right':
            self.yaw -= self.TURN_ANGLE
        
        # Normalize yaw
        while self.yaw > np.pi:
            self.yaw -= 2 * np.pi
        while self.yaw < -np.pi:
            self.yaw += 2 * np.pi
    
    def _check_floor_change(self, current_y: float):
        """Check and handle floor transitions."""
        y_change = current_y - self.prev_y
        
        if abs(y_change) > self.FLOOR_CHANGE_THRESHOLD:
            # Floor change detected
            self.floor_changes += 1
            
            # Determine new floor
            if y_change > 0:
                self.current_floor += 1
            else:
                self.current_floor -= 1
            
            # Store floor height
            self.floor_heights[self.current_floor] = current_y
            
            # IMPORTANT: Sync Y coordinate (DR can't track vertical movement)
            self.position[1] = current_y
            
            print(f"Floor change: Y {self.prev_y:.2f} -> {current_y:.2f}, now floor {self.current_floor}")
        
        self.prev_y = current_y
    
    def _check_loop_closure(self, depth: np.ndarray):
        """Simple loop closure using depth signatures."""
        self.lc_cooldown = max(0, self.lc_cooldown - 1)
        
        # Compute depth signature
        sig = self._depth_signature(depth)
        
        # Store candidate periodically
        if self.frame_count % 20 == 0:
            self.lc_candidates.append((
                self.position.copy(),
                self.yaw,
                sig
            ))
            
            # Limit size
            if len(self.lc_candidates) > 200:
                self.lc_candidates.pop(0)
        
        # Check for loop closure
        if self.lc_cooldown > 0 or len(self.lc_candidates) < 10:
            return
        
        for cand_pos, cand_yaw, cand_sig in self.lc_candidates[:-10]:
            # Distance check
            dist = np.sqrt(
                (self.position[0] - cand_pos[0])**2 +
                (self.position[2] - cand_pos[2])**2
            )
            
            if dist > 2.0:
                continue
            
            # Signature similarity
            sig_diff = np.mean(np.abs(sig - cand_sig))
            
            if sig_diff < 0.25:
                # Loop closure!
                correction = cand_pos - self.position
                
                # Apply partial correction
                self.position[0] += 0.5 * correction[0]
                self.position[2] += 0.5 * correction[2]
                
                self.lc_cooldown = 50
                self.confidence = min(1.0, self.confidence + 0.1)
                
                print(f"Loop closure: correction {np.linalg.norm(correction):.3f}m")
                return
    
    def _depth_signature(self, depth: np.ndarray) -> np.ndarray:
        """Compute compact depth signature."""
        h, w = depth.shape
        
        # Middle row, downsampled
        row = depth[h//2, ::max(1, w//16)][:16]
        
        # Normalize
        valid = row[(row > 0.1) & (row < 10)]
        if len(valid) < 4:
            return np.zeros(16)
        
        return np.clip(row / 5.0, 0, 2)
    
    def correct_pose(self, position: np.ndarray, yaw: float = None):
        """User pose correction."""
        self.position = position.copy()
        if yaw is not None:
            self.yaw = yaw
        self.prev_y = position[1]
        self.confidence = 1.0
        print(f"Pose corrected to ({position[0]:.2f}, {position[2]:.2f})")
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        stats = {
            'frames': self.frame_count,
            'distance': self.total_distance,
            'floor_changes': self.floor_changes,
            'current_floor': self.current_floor,
            'confidence': self.confidence,
            'lc_candidates': len(self.lc_candidates),
        }
        
        if self.gt_errors_2d:
            stats['error_2d_mean'] = np.mean(self.gt_errors_2d)
            stats['error_2d_max'] = np.max(self.gt_errors_2d)
            stats['error_2d_current'] = self.gt_errors_2d[-1]
            stats['drift_rate'] = self.gt_errors_2d[-1] / max(self.total_distance, 0.01) * 100
        
        return stats
    
    def reset(self):
        """Reset state."""
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.initialized = False
        self.current_floor = 0
        self.floor_heights.clear()
        self.prev_y = 0.0
        self.frame_count = 0
        self.total_distance = 0.0
        self.floor_changes = 0
        self.confidence = 1.0
        self.lc_candidates.clear()
        self.gt_errors_2d.clear()
