"""
Dead Reckoning Localizer
========================

Simple action-based pose tracking for navigation.
Since Habitat actions are deterministic, dead reckoning
works very well with minimal drift.

This is much faster and more accurate than visual odometry
when we know the exact actions being executed.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DRResult:
    """Dead reckoning result."""
    position: np.ndarray  # [x, y, z]
    yaw: float           # radians
    quaternion: np.ndarray  # [x, y, z, w] for compatibility


class DeadReckoningLocalizer:
    """
    Fast, accurate pose tracking using known actions.
    
    Habitat action parameters:
    - move_forward: 0.25m
    - turn_left: 10 degrees CCW
    - turn_right: 10 degrees CW
    """
    
    FORWARD_DIST = 0.25
    TURN_ANGLE = np.radians(10.0)
    
    def __init__(self):
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.initialized = False
    
    def initialize(self, position: np.ndarray, rotation: np.ndarray):
        """
        Initialize from ground truth pose.
        
        Args:
            position: [x, y, z] starting position
            rotation: quaternion [x, y, z, w] or yaw angle
        """
        self.position = position.copy()
        
        if isinstance(rotation, (int, float)):
            self.yaw = float(rotation)
        elif len(rotation) == 4:
            # Extract yaw from quaternion
            x, y, z, w = rotation
            siny_cosp = 2.0 * (w * y + z * x)
            cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
            self.yaw = np.arctan2(siny_cosp, cosy_cosp)
        else:
            self.yaw = 0.0
        
        self.initialized = True
    
    def update(self, action: Optional[str]) -> DRResult:
        """
        Update pose based on action.
        
        Args:
            action: 'move_forward', 'turn_left', 'turn_right', or None
            
        Returns:
            DRResult with updated pose
        """
        if action == 'move_forward':
            # Move in direction of yaw
            # At yaw=0, forward is -Z direction
            self.position[0] -= np.sin(self.yaw) * self.FORWARD_DIST
            self.position[2] -= np.cos(self.yaw) * self.FORWARD_DIST
            
        elif action == 'turn_left':
            self.yaw += self.TURN_ANGLE
            
        elif action == 'turn_right':
            self.yaw -= self.TURN_ANGLE
        
        # Normalize yaw
        while self.yaw > np.pi:
            self.yaw -= 2 * np.pi
        while self.yaw < -np.pi:
            self.yaw += 2 * np.pi
        
        return DRResult(
            position=self.position.copy(),
            yaw=self.yaw,
            quaternion=self._yaw_to_quat(self.yaw)
        )
    
    def _yaw_to_quat(self, yaw: float) -> np.ndarray:
        """Convert yaw to quaternion [x, y, z, w]."""
        half_yaw = yaw / 2
        return np.array([0, np.sin(half_yaw), 0, np.cos(half_yaw)])
    
    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current pose as (position, quaternion)."""
        return self.position.copy(), self._yaw_to_quat(self.yaw)
    
    def get_pose_yaw(self) -> Tuple[np.ndarray, float]:
        """Get current pose as (position, yaw)."""
        return self.position.copy(), self.yaw
    
    def reset(self):
        """Reset to origin."""
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.initialized = False
