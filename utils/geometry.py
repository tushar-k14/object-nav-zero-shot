"""
Geometry Utilities
==================

Core geometric operations for ObjectNav:
1. Depth unprojection (2D pixel + depth → 3D point)
2. Coordinate frame transforms (camera → world)
3. Quaternion operations
4. Visibility checks (distance, angle, occlusion)
5. Map coordinate conversions
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    
    @classmethod
    def from_fov(cls, hfov_deg: float, width: int, height: int) -> 'CameraIntrinsics':
        """Create from horizontal field of view."""
        hfov_rad = np.radians(hfov_deg)
        fx = (width / 2) / np.tan(hfov_rad / 2)
        fy = fx  # Square pixels
        return cls(fx=fx, fy=fy, cx=width/2, cy=height/2, width=width, height=height)
    
    @classmethod
    def habitat_default(cls, width: int = 640, height: int = 480) -> 'CameraIntrinsics':
        """Default Habitat camera (90° HFOV)."""
        return cls.from_fov(90.0, width, height)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    
    Args:
        q: Quaternion in [x, y, z, w] format (Habitat convention)
        
    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = q
    
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    min_depth: float = 0.1,
    max_depth: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to 3D point cloud in camera frame.
    
    Camera frame (Habitat convention):
        X: right
        Y: down  
        Z: forward (into scene)
    
    Args:
        depth: HxW depth image in meters
        intrinsics: Camera intrinsic parameters
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        
    Returns:
        points: Nx3 array of 3D points in camera frame
        valid_mask: HxW boolean mask of valid pixels
    """
    h, w = depth.shape
    
    # Create pixel coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    # Valid depth mask
    valid_mask = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)
    
    # Get valid pixels
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    d_valid = depth[valid_mask]
    
    # Back-project to 3D
    # X = (u - cx) * d / fx
    # Y = (v - cy) * d / fy
    # Z = d
    X = (u_valid - intrinsics.cx) * d_valid / intrinsics.fx
    Y = (v_valid - intrinsics.cy) * d_valid / intrinsics.fy
    Z = d_valid
    
    points = np.stack([X, Y, Z], axis=1)
    
    return points, valid_mask


def transform_points_camera_to_world(
    points_camera: np.ndarray,
    agent_position: np.ndarray,
    agent_rotation: np.ndarray
) -> np.ndarray:
    """
    Transform points from camera frame to world frame.
    
    Handles the Y-axis flip between:
        Camera: Y-down
        World: Y-up
    
    Args:
        points_camera: Nx3 points in camera frame
        agent_position: [x, y, z] agent position in world
        agent_rotation: [x, y, z, w] quaternion
        
    Returns:
        Nx3 points in world frame
    """
    if len(points_camera) == 0:
        return points_camera
    
    # Flip Y axis: camera Y-down → world Y-up
    points_adjusted = points_camera.copy()
    points_adjusted[:, 1] = -points_adjusted[:, 1]
    
    # Get rotation matrix
    R = quaternion_to_rotation_matrix(agent_rotation)
    
    # Apply rotation and translation
    points_world = (R @ points_adjusted.T).T + agent_position
    
    return points_world


def unproject_pixel_to_3d(
    u: int, v: int, 
    depth: float,
    intrinsics: CameraIntrinsics,
    agent_position: np.ndarray,
    agent_rotation: np.ndarray
) -> np.ndarray:
    """
    Unproject a single pixel to 3D world coordinates.
    
    Args:
        u, v: Pixel coordinates
        depth: Depth value at pixel
        intrinsics: Camera parameters
        agent_position: Agent world position
        agent_rotation: Agent rotation quaternion
        
    Returns:
        [x, y, z] world position
    """
    # Camera frame
    X = (u - intrinsics.cx) * depth / intrinsics.fx
    Y = (v - intrinsics.cy) * depth / intrinsics.fy
    Z = depth
    
    point_camera = np.array([[X, Y, Z]])
    point_world = transform_points_camera_to_world(
        point_camera, agent_position, agent_rotation
    )
    
    return point_world[0]


def world_to_map_coords(
    points_world: np.ndarray,
    map_size: int,
    resolution_cm: float,
    origin: np.ndarray = None
) -> np.ndarray:
    """
    Convert world XZ coordinates to map pixel coordinates.
    
    Map convention:
        - Map center = world origin (or specified origin)
        - X world → U map (columns)
        - Z world → V map (rows)
        - Y world (height) is ignored
    
    Args:
        points_world: Nx3 world points
        map_size: Map size in pixels
        resolution_cm: Centimeters per pixel
        origin: Optional world origin [x, z]
        
    Returns:
        Nx2 map coordinates [u, v]
    """
    if origin is None:
        origin = np.array([0.0, 0.0])
    
    map_center = map_size // 2
    
    # Extract XZ (horizontal plane)
    x = points_world[:, 0] - origin[0]
    z = points_world[:, 2] - origin[1]
    
    # Convert to map pixels
    u = (x * 100 / resolution_cm + map_center).astype(np.int32)
    v = (z * 100 / resolution_cm + map_center).astype(np.int32)
    
    return np.stack([u, v], axis=1)


def map_to_world_coords(
    map_coords: np.ndarray,
    map_size: int,
    resolution_cm: float,
    height: float = 0.0,
    origin: np.ndarray = None
) -> np.ndarray:
    """
    Convert map coordinates back to world XZ coordinates.
    
    Args:
        map_coords: Nx2 map coordinates [u, v]
        map_size: Map size in pixels
        resolution_cm: Centimeters per pixel
        height: Y coordinate to assign
        origin: Optional world origin
        
    Returns:
        Nx3 world coordinates [x, y, z]
    """
    if origin is None:
        origin = np.array([0.0, 0.0])
    
    map_center = map_size // 2
    
    u = map_coords[:, 0]
    v = map_coords[:, 1]
    
    x = (u - map_center) * resolution_cm / 100 + origin[0]
    z = (v - map_center) * resolution_cm / 100 + origin[1]
    y = np.full_like(x, height)
    
    return np.stack([x, y, z], axis=1)


def check_visibility(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    occupancy_map: np.ndarray,
    map_size: int,
    resolution_cm: float,
    max_distance: float = 5.0,
    fov_deg: float = 90.0,
    observer_heading: Optional[np.ndarray] = None
) -> Tuple[bool, float]:
    """
    Check if target is visible from observer position.
    
    Visibility requires:
    1. Distance within max_distance
    2. Target within FOV (if heading provided)
    3. No occlusion (ray-cast through occupancy map)
    
    Args:
        observer_pos: [x, y, z] observer world position
        target_pos: [x, y, z] target world position
        occupancy_map: HxW occupancy map (1 = obstacle)
        map_size: Map size in pixels
        resolution_cm: Map resolution
        max_distance: Maximum visibility distance
        fov_deg: Field of view in degrees
        observer_heading: Optional [dx, dz] heading direction
        
    Returns:
        (is_visible, distance)
    """
    # 1. Distance check
    diff = target_pos - observer_pos
    distance = np.sqrt(diff[0]**2 + diff[2]**2)  # XZ distance
    
    if distance > max_distance:
        return False, distance
    
    # 2. FOV check (if heading provided)
    if observer_heading is not None:
        # Normalize heading
        heading_norm = observer_heading / (np.linalg.norm(observer_heading) + 1e-8)
        
        # Direction to target (XZ plane)
        to_target = np.array([diff[0], diff[2]])
        to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-8)
        
        # Angle between heading and target direction
        cos_angle = np.dot(heading_norm, to_target_norm)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        if angle > fov_deg / 2:
            return False, distance
    
    # 3. Occlusion check (ray-cast)
    # Convert positions to map coordinates
    obs_map = world_to_map_coords(
        observer_pos.reshape(1, 3), map_size, resolution_cm
    )[0]
    tgt_map = world_to_map_coords(
        target_pos.reshape(1, 3), map_size, resolution_cm
    )[0]
    
    # Bresenham's line algorithm
    is_occluded = _ray_cast_occlusion(
        obs_map[0], obs_map[1],
        tgt_map[0], tgt_map[1],
        occupancy_map
    )
    
    return not is_occluded, distance


def _ray_cast_occlusion(
    x0: int, y0: int,
    x1: int, y1: int,
    occupancy_map: np.ndarray
) -> bool:
    """
    Check if ray from (x0,y0) to (x1,y1) passes through obstacles.
    Uses Bresenham's line algorithm.
    
    Returns True if occluded.
    """
    h, w = occupancy_map.shape
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        # Check bounds
        if 0 <= x < w and 0 <= y < h:
            # Skip start and end points
            if (x, y) != (x0, y0) and (x, y) != (x1, y1):
                if occupancy_map[y, x] > 0.5:
                    return True  # Occluded
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return False  # Not occluded


def compute_object_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute 3D Euclidean distance between two positions."""
    return float(np.linalg.norm(pos1 - pos2))


def merge_nearby_positions(
    positions: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge positions that are within threshold distance.
    
    Args:
        positions: Nx3 array of positions
        threshold: Distance threshold for merging
        
    Returns:
        merged_positions: Mx3 array of merged positions
        labels: N array mapping original positions to merged indices
    """
    if len(positions) == 0:
        return positions, np.array([])
    
    n = len(positions)
    labels = np.arange(n)
    
    # Simple greedy merging
    for i in range(n):
        if labels[i] != i:
            continue
        
        for j in range(i + 1, n):
            if labels[j] != j:
                continue
            
            dist = compute_object_distance(positions[i], positions[j])
            if dist < threshold:
                labels[j] = i
    
    # Compute merged positions (mean of each cluster)
    unique_labels = np.unique(labels)
    merged = []
    label_map = {}
    
    for new_idx, old_idx in enumerate(unique_labels):
        mask = labels == old_idx
        merged.append(positions[mask].mean(axis=0))
        label_map[old_idx] = new_idx
    
    # Remap labels
    new_labels = np.array([label_map[l] for l in labels])
    
    return np.array(merged), new_labels
