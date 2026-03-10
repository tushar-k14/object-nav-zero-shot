"""
Coordinate Transforms
=====================

Utilities for coordinate system conversions.
"""

import numpy as np
from typing import Tuple


class CameraIntrinsics:
    """Camera intrinsic parameters."""
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        hfov: float = 90.0,
        fx: float = None,
        fy: float = None,
        cx: float = None,
        cy: float = None
    ):
        self.width = width
        self.height = height
        self.hfov = hfov
        
        # Compute focal length from HFOV if not provided
        if fx is None:
            self.fx = width / (2.0 * np.tan(np.radians(hfov / 2)))
        else:
            self.fx = fx
            
        self.fy = fy if fy is not None else self.fx
        self.cx = cx if cx is not None else width / 2
        self.cy = cy if cy is not None else height / 2
    
    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D pixels.
        
        Args:
            points_3d: [N, 3] points in camera frame
            
        Returns:
            [N, 2] pixel coordinates
        """
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        
        # Avoid division by zero
        z = np.maximum(z, 1e-6)
        
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        
        return np.stack([u, v], axis=1)
    
    def unproject(self, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """
        Unproject 2D pixels to 3D points.
        
        Args:
            pixels: [N, 2] pixel coordinates (u, v)
            depths: [N] depth values
            
        Returns:
            [N, 3] points in camera frame
        """
        u = pixels[:, 0]
        v = pixels[:, 1]
        
        x = (u - self.cx) * depths / self.fx
        y = (v - self.cy) * depths / self.fy
        z = depths
        
        return np.stack([x, y, z], axis=1)


class MapCoordinates:
    """
    Handle world <-> map coordinate conversions.
    
    Matches the coordinate system used in occupancy mapping.
    """
    
    def __init__(
        self,
        map_size: int = 480,
        resolution_cm: float = 5.0,
        origin_x: float = 0.0,
        origin_z: float = 0.0
    ):
        self.map_size = map_size
        self.resolution = resolution_cm
        self.origin_x = origin_x
        self.origin_z = origin_z
    
    def world_to_map(self, wx: float, wz: float) -> Tuple[int, int]:
        """Convert world coordinates to map pixels."""
        mx = int((wx - self.origin_x) * 100 / self.resolution + self.map_size // 2)
        my = int((wz - self.origin_z) * 100 / self.resolution + self.map_size // 2)
        return (mx, my)
    
    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """Convert map pixels to world coordinates."""
        wx = (mx - self.map_size // 2) * self.resolution / 100.0 + self.origin_x
        wz = (my - self.map_size // 2) * self.resolution / 100.0 + self.origin_z
        return (wx, wz)
    
    def world_to_map_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Convert batch of world points to map coordinates.
        
        Args:
            points: [N, 2] or [N, 3] world coordinates (x, z) or (x, y, z)
            
        Returns:
            [N, 2] map coordinates (mx, my)
        """
        wx = points[:, 0]
        wz = points[:, 2] if points.shape[1] > 2 else points[:, 1]
        
        mx = ((wx - self.origin_x) * 100 / self.resolution + self.map_size // 2).astype(int)
        my = ((wz - self.origin_z) * 100 / self.resolution + self.map_size // 2).astype(int)
        
        return np.stack([mx, my], axis=1)


def depth_to_pointcloud(
    depth: np.ndarray,
    camera: CameraIntrinsics,
    max_depth: float = 10.0,
    subsample: int = 1
) -> np.ndarray:
    """
    Convert depth image to 3D point cloud in camera frame.
    
    Args:
        depth: [H, W] depth image in meters
        camera: Camera intrinsics
        max_depth: Maximum valid depth
        subsample: Subsample factor (1 = full resolution)
        
    Returns:
        [N, 3] point cloud in camera frame
    """
    h, w = depth.shape
    
    # Create pixel coordinate grid
    u = np.arange(0, w, subsample)
    v = np.arange(0, h, subsample)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    
    # Get depths at these pixels
    d = depth[v, u]
    
    # Filter valid depths
    valid = (d > 0.1) & (d < max_depth)
    u, v, d = u[valid], v[valid], d[valid]
    
    # Unproject to 3D
    x = (u - camera.cx) * d / camera.fx
    y = (v - camera.cy) * d / camera.fy
    z = d
    
    return np.stack([x, y, z], axis=1)


def transform_pointcloud(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform point cloud by 4x4 transformation matrix.
    
    Args:
        points: [N, 3] points
        T: [4, 4] transformation matrix
        
    Returns:
        [N, 3] transformed points
    """
    # Convert to homogeneous
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Transform
    points_t = (T @ points_h.T).T
    
    return points_t[:, :3]


# ================================================================
# Standalone helper functions
# ================================================================

def get_yaw(quaternion) -> float:
    """
    Extract yaw angle from quaternion [x, y, z, w].
    Works with numpy arrays and Magnum quaternions.
    """
    if hasattr(quaternion, 'scalar'):
        # Magnum quaternion
        x, y, z = quaternion.vector.x, quaternion.vector.y, quaternion.vector.z
        w = quaternion.scalar
    else:
        x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    siny_cosp = 2.0 * (w * y - z * x)
    cosy_cosp = 1.0 - 2.0 * (y * y + x * x)  # Note: Habitat Y-up
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return yaw


def world_to_map(
    wx: float, wz: float,
    origin_x: float, origin_z: float,
    resolution_cm: float, map_size: int,
) -> Tuple[int, int]:
    """Convert world (x, z) to map pixel (mx, my)."""
    mx = int((wx - origin_x) * 100 / resolution_cm + map_size // 2)
    my = int((wz - origin_z) * 100 / resolution_cm + map_size // 2)
    return (mx, my)


def map_to_world(
    mx: int, my: int,
    origin_x: float, origin_z: float,
    resolution_cm: float, map_size: int,
) -> Tuple[float, float]:
    """Convert map pixel (mx, my) to world (x, z)."""
    wx = (mx - map_size // 2) * resolution_cm / 100.0 + origin_x
    wz = (my - map_size // 2) * resolution_cm / 100.0 + origin_z
    return (wx, wz)


def angle_to_target(
    agent_x: float, agent_z: float,
    agent_yaw: float,
    target_x: float, target_z: float,
) -> float:
    """
    Compute signed angle from agent heading to target.

    Returns angle in radians, positive = turn left, negative = turn right.
    Assumes Habitat coordinate system (Y-up, -Z forward).
    """
    dx = target_x - agent_x
    dz = target_z - agent_z
    target_angle = np.arctan2(-dx, -dz)  # Habitat: -Z is forward
    diff = target_angle - agent_yaw
    # Normalize to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return float(diff)


def compute_3d_position(
    px: int, py: int,
    depth_value: float,
    camera: CameraIntrinsics,
    agent_position: np.ndarray,
    agent_yaw: float,
) -> np.ndarray:
    """
    Backproject 2D pixel + depth to 3D world position.

    Args:
        px, py: pixel coordinates
        depth_value: depth in meters
        camera: camera intrinsics
        agent_position: [x, y, z] world position
        agent_yaw: agent heading in radians

    Returns:
        [x, y, z] world position
    """
    # Camera frame (Habitat: x-right, y-down, z-forward)
    cam_x = (px - camera.cx) * depth_value / camera.fx
    cam_z = depth_value

    # Rotate to world frame
    cos_y, sin_y = np.cos(agent_yaw), np.sin(agent_yaw)
    world_x = agent_position[0] + cos_y * cam_x - sin_y * cam_z
    world_z = agent_position[2] + sin_y * cam_x + cos_y * cam_z
    world_y = agent_position[1]

    return np.array([world_x, world_y, world_z])