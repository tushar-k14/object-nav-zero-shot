"""
Vectorized Occupancy Map Update
================================

Drop-in replacement for FixedOccupancyMap.update() that uses
numpy vectorized operations instead of Python pixel loops.

Performance: ~10-20x faster (19,200 iterations in Python → single numpy ops)

Usage:
    from mapping.fast_occupancy import patch_occupancy_map
    patch_occupancy_map(mapper.occupancy_map)
    # Now mapper.occupancy_map.update() uses the fast version

Or apply manually:
    import types
    from mapping.fast_occupancy import fast_update
    mapper.occupancy_map.update = types.MethodType(fast_update, mapper.occupancy_map)
"""

import numpy as np
import cv2
from typing import Dict


def fast_update(
    self,
    depth: np.ndarray,
    agent_position: np.ndarray,
    agent_rotation: np.ndarray,
) -> Dict:
    """
    Vectorized map update. Drop-in replacement for FixedOccupancyMap.update().

    Replaces the nested Python loop over pixels with numpy array operations.
    Identical output, ~10-20x faster.
    """
    self.frame_count += 1

    # Set origin on first frame
    self._set_origin(agent_position[0], agent_position[2])

    # Check resize
    map_x, map_y = self._world_to_map(agent_position[0], agent_position[2])
    self._check_and_resize(map_x, map_y)
    map_x, map_y = self._world_to_map(agent_position[0], agent_position[2])

    self.trajectory.append(agent_position.copy())

    yaw = self._get_yaw_from_quaternion(agent_rotation)
    agent_height = agent_position[1]

    local_size = self.config.local_map_size
    local_obstacle = np.zeros((local_size, local_size), dtype=np.float32)
    local_explored = np.zeros((local_size, local_size), dtype=np.float32)

    # ============================================================
    # VECTORIZED depth projection (replaces Python loop)
    # ============================================================
    # Subsample: every 4th pixel (same as original)
    step = 4
    v_coords = np.arange(0, self.img_height, step)
    u_coords = np.arange(0, self.img_width, step)
    vv, uu = np.meshgrid(v_coords, u_coords, indexing='ij')  # [H/4, W/4]
    vv = vv.ravel()
    uu = uu.ravel()

    # Get depth values at subsampled pixels
    d = depth[vv, uu]  # [N]

    # Filter valid depths
    valid = (d > 0.1) & (d <= 5.0) & np.isfinite(d)
    d = d[valid]
    uu = uu[valid]
    vv = vv[valid]

    if len(d) == 0:
        # No valid depth — just mark agent position
        if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
            cv2.circle(self.visited_map, (map_x, map_y), 2, 1, -1)
            cv2.circle(self.explored_map, (map_x, map_y), 5, 1.0, -1)
        return {'frame': self.frame_count, 'agent_map_pos': (map_x, map_y), 'yaw_deg': np.degrees(yaw)}

    # Back-project to camera frame
    z_cam = d
    x_cam = (uu - self.cx) * d / self.fx
    y_cam = (vv - self.cy) * d / self.fy

    # Height classification
    point_height = agent_height - y_cam
    rel_height = (point_height - (agent_height - self.config.agent_height_cm / 100)) * 100  # cm

    # Convert to local map coords
    local_x = (x_cam * 100 / self.resolution).astype(np.int32) + local_size // 2
    local_y = local_size // 2 - (z_cam * 100 / self.resolution).astype(np.int32)

    # Bounds check
    in_bounds = (local_x >= 0) & (local_x < local_size) & (local_y >= 0) & (local_y < local_size)

    local_x = local_x[in_bounds]
    local_y = local_y[in_bounds]
    rel_height = rel_height[in_bounds]

    # Classify: obstacles
    is_obstacle = (rel_height > self.config.min_obstacle_height_cm) & \
                  (rel_height < self.config.max_obstacle_height_cm)
    # Classify: floor (fully explored)
    is_floor = rel_height < self.config.floor_threshold_cm

    # Write to local maps using np.maximum.at for accumulation
    if np.any(is_obstacle):
        obs_y = local_y[is_obstacle]
        obs_x = local_x[is_obstacle]
        np.maximum.at(local_obstacle, (obs_y, obs_x), 1.0)

    if np.any(is_floor):
        floor_y = local_y[is_floor]
        floor_x = local_x[is_floor]
        np.maximum.at(local_explored, (floor_y, floor_x), 1.0)

    # Non-floor points: mark as partially explored (0.3)
    not_floor = ~is_floor
    if np.any(not_floor):
        nf_y = local_y[not_floor]
        nf_x = local_x[not_floor]
        np.maximum.at(local_explored, (nf_y, nf_x), 0.3)

    # ============================================================
    # Rotate + place into global map (same as original)
    # ============================================================
    rot_angle_deg = np.degrees(yaw)
    center = (local_size // 2, local_size // 2)
    M = cv2.getRotationMatrix2D(center, rot_angle_deg, 1.0)

    rotated_obstacle = cv2.warpAffine(
        local_obstacle, M, (local_size, local_size),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )
    rotated_explored = cv2.warpAffine(
        local_explored, M, (local_size, local_size),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )

    # Place into global map
    half_size = local_size // 2
    g_x1 = max(0, map_x - half_size)
    g_x2 = min(self.map_size, map_x + half_size)
    g_y1 = max(0, map_y - half_size)
    g_y2 = min(self.map_size, map_y + half_size)

    l_x1 = half_size - (map_x - g_x1)
    l_x2 = half_size + (g_x2 - map_x)
    l_y1 = half_size - (map_y - g_y1)
    l_y2 = half_size + (g_y2 - map_y)

    if g_x2 > g_x1 and g_y2 > g_y1:
        self.obstacle_map[g_y1:g_y2, g_x1:g_x2] = np.maximum(
            self.obstacle_map[g_y1:g_y2, g_x1:g_x2],
            rotated_obstacle[l_y1:l_y2, l_x1:l_x2],
        )
        self.explored_map[g_y1:g_y2, g_x1:g_x2] = np.maximum(
            self.explored_map[g_y1:g_y2, g_x1:g_x2],
            rotated_explored[l_y1:l_y2, l_x1:l_x2],
        )

    # Mark agent position
    if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
        cv2.circle(self.visited_map, (map_x, map_y), 2, 1, -1)
        cv2.circle(self.explored_map, (map_x, map_y), 5, 1.0, -1)

    return {
        'frame': self.frame_count,
        'agent_map_pos': (map_x, map_y),
        'yaw_deg': np.degrees(yaw),
    }


def patch_occupancy_map(occ_map):
    """
    Monkey-patch a FixedOccupancyMap instance to use the fast vectorized update.

    Usage:
        from mapping.fast_occupancy import patch_occupancy_map
        patch_occupancy_map(mapper.occupancy_map)
    """
    import types
    occ_map.update = types.MethodType(fast_update, occ_map)
