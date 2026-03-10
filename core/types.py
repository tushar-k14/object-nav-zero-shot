"""
Core Data Types
===============

Common data structures used across the ObjectNav system.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


@dataclass
class Pose:
    """6-DoF pose representation."""
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # quaternion [x, y, z, w] or rotation matrix [3,3]
    timestamp: float = 0.0
    
    @property
    def x(self) -> float:
        return self.position[0]
    
    @property
    def y(self) -> float:
        return self.position[1]
    
    @property
    def z(self) -> float:
        return self.position[2]
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, 3] = self.position
        
        if self.rotation.shape == (3, 3):
            T[:3, :3] = self.rotation
        else:
            # Quaternion to rotation matrix
            T[:3, :3] = quat_to_rotation_matrix(self.rotation)
        
        return T
    
    @classmethod
    def from_matrix(cls, T: np.ndarray, timestamp: float = 0.0) -> 'Pose':
        """Create from 4x4 transformation matrix."""
        position = T[:3, 3].copy()
        rotation = T[:3, :3].copy()
        return cls(position=position, rotation=rotation, timestamp=timestamp)
    
    @classmethod
    def identity(cls) -> 'Pose':
        """Create identity pose."""
        return cls(
            position=np.zeros(3),
            rotation=np.eye(3)
        )
    
    def inverse(self) -> 'Pose':
        """Return inverse pose."""
        T = self.to_matrix()
        T_inv = np.linalg.inv(T)
        return Pose.from_matrix(T_inv, self.timestamp)
    
    def compose(self, other: 'Pose') -> 'Pose':
        """Compose with another pose: self * other."""
        T1 = self.to_matrix()
        T2 = other.to_matrix()
        return Pose.from_matrix(T1 @ T2, other.timestamp)


@dataclass
class Keyframe:
    """A keyframe for SLAM."""
    id: int
    pose: Pose
    rgb: np.ndarray
    depth: np.ndarray
    keypoints: Optional[np.ndarray] = None      # [N, 2] pixel coordinates
    descriptors: Optional[np.ndarray] = None    # [N, D] feature descriptors
    keypoints_3d: Optional[np.ndarray] = None   # [N, 3] 3D points
    global_descriptor: Optional[np.ndarray] = None  # For loop closure
    
    # Connections
    covisible_keyframes: List[int] = field(default_factory=list)
    loop_candidates: List[int] = field(default_factory=list)


@dataclass
class MapPoint:
    """A 3D map point."""
    id: int
    position: np.ndarray  # [x, y, z] world coordinates
    descriptor: np.ndarray  # Representative descriptor
    observations: Dict[int, int] = field(default_factory=dict)  # keyframe_id -> keypoint_idx
    normal: Optional[np.ndarray] = None  # Viewing direction
    
    @property
    def num_observations(self) -> int:
        return len(self.observations)


@dataclass
class OdometryResult:
    """Result from visual odometry."""
    success: bool
    relative_pose: Optional[Pose]  # Transform from previous to current
    inlier_count: int = 0
    total_matches: int = 0
    method: str = ""  # "feature" or "icp" or "hybrid"
    covariance: Optional[np.ndarray] = None  # 6x6 uncertainty


@dataclass 
class LoopClosureResult:
    """Result from loop closure detection."""
    detected: bool
    query_keyframe_id: int
    match_keyframe_id: int = -1
    similarity_score: float = 0.0
    relative_pose: Optional[Pose] = None
    inlier_count: int = 0
    verified: bool = False  # Geometric verification passed


class TrackingState(Enum):
    """Visual odometry tracking state."""
    NOT_INITIALIZED = "not_initialized"
    OK = "ok"
    LOST = "lost"
    RELOCALIZATION = "relocalization"


# ============================================================================
# Utility Functions
# ============================================================================

def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    """
    x, y, z, w = q
    
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])
    
    return R


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [x, y, z, w].
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([x, y, z, w])


def get_yaw_from_quaternion(q: np.ndarray) -> float:
    """Extract yaw angle from quaternion [x, y, z, w]."""
    x, y, z, w = q
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    return np.arctan2(siny_cosp, cosy_cosp)