"""
Fixed Occupancy Map
===================

1. Using Habitat's actual coordinate system (Y-up, -Z forward)
2. Proper egocentric to allocentric transform
3. Accumulating in local frame then rotating
4. Using morphological cleanup

Habitat uses Y-up coordinate system where:
- X: right
- Y: up  
- Z: backward (agent faces -Z)

When projecting to top-down map:
- Map X = World X
- Map Y = World Z (flipped because -Z is forward)
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class MapConfig:
    """Map configuration."""
    map_size_cm: int = 4800          # 24m x 24m total
    resolution_cm: int = 5           # 5cm per pixel
    agent_height_cm: int = 125       # 1.25m camera height
    min_obstacle_height_cm: int = 20 # 20cm min obstacle
    max_obstacle_height_cm: int = 150 # 1.5m max obstacle
    floor_threshold_cm: int = 10     # 10cm floor detection
    vision_range_cm: int = 500       # 5m max depth
    
    @property
    def map_size(self) -> int:
        return self.map_size_cm // self.resolution_cm
    
    @property
    def local_map_size(self) -> int:
        """Size of local egocentric map."""
        return (self.vision_range_cm * 2) // self.resolution_cm


class FixedOccupancyMap:
    """
    Building local egocentric map first, then rotate and
    place into global map.
    
    Dynamic resizing
    """
    
    def __init__(self, config: MapConfig = None):
        self.config = config or MapConfig()
        self.map_size = self.config.map_size
        self.resolution = self.config.resolution_cm
        
        # Global maps (allocentric - world frame)
        self.obstacle_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.explored_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.visited_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        
        # Camera parameters (Habitat default: 90° HFOV, 640x480)
        self.img_width = 640
        self.img_height = 480
        self.hfov = 90.0
        self.fx = self.img_width / (2.0 * np.tan(np.radians(self.hfov / 2)))
        self.fy = self.fx
        self.cx = self.img_width / 2
        self.cy = self.img_height / 2
        
        # Tracking
        self.trajectory = []
        self.frame_count = 0
        
        # Origin offset (center of map in world coords)
        self.origin_x = 0.0
        self.origin_z = 0.0
        self.origin_set = False
        
        # Boundary margin for triggering resize (pixels from edge)
        self.resize_margin = 50
        self.max_map_size = 2000  # Maximum map size to prevent memory issues
    
    def reset(self):
        """Reset map."""
        self.obstacle_map.fill(0)
        self.explored_map.fill(0)
        self.visited_map.fill(0)
        self.trajectory = []
        self.frame_count = 0
        self.origin_set = False
    
    def _set_origin(self, agent_x: float, agent_z: float):
        """Set map origin to agent's starting position."""
        if not self.origin_set:
            self.origin_x = agent_x
            self.origin_z = agent_z
            self.origin_set = True
    
    def _check_and_resize(self, map_x: int, map_y: int):
        """Check if position is near boundary and resize if needed."""
        need_resize = False
        expand_direction = [0, 0, 0, 0]  # left, right, top, bottom
        
        if map_x < self.resize_margin:
            need_resize = True
            expand_direction[0] = 1  # Expand left
        elif map_x >= self.map_size - self.resize_margin:
            need_resize = True
            expand_direction[1] = 1  # Expand right
            
        if map_y < self.resize_margin:
            need_resize = True
            expand_direction[2] = 1  # Expand top
        elif map_y >= self.map_size - self.resize_margin:
            need_resize = True
            expand_direction[3] = 1  # Expand bottom
        
        if need_resize and self.map_size < self.max_map_size:
            self._expand_map(expand_direction)
    
    def _expand_map(self, direction: list):
        """
        Expand map in specified directions.
        
        Args:
            direction: [left, right, top, bottom] - 1 to expand, 0 to not
        """
        expand_size = 100  # Pixels to expand
        
        left, right, top, bottom = direction
        
        # Always expand symmetrically to keep map square
        expand_x = (left + right) * expand_size
        expand_y = (top + bottom) * expand_size
        expand_total = max(expand_x, expand_y)  # Keep square
        
        new_size = self.map_size + expand_total
        
        # Cap at max size
        if new_size > self.max_map_size:
            return
        
        # Calculate offsets - center the old map in the new one if expanding asymmetrically
        if expand_x > expand_y:
            x_offset = left * expand_size
            y_offset = (expand_total - expand_y) // 2 + top * expand_size
        elif expand_y > expand_x:
            x_offset = (expand_total - expand_x) // 2 + left * expand_size
            y_offset = top * expand_size
        else:
            x_offset = left * expand_size
            y_offset = top * expand_size
        
        old_h, old_w = self.obstacle_map.shape
        
        # Create new square maps
        new_obstacle = np.zeros((new_size, new_size), dtype=np.float32)
        new_explored = np.zeros((new_size, new_size), dtype=np.float32)
        new_visited = np.zeros((new_size, new_size), dtype=np.uint8)
        
        # Copy old data (use actual array shape, not self.map_size)
        new_obstacle[y_offset:y_offset+old_h, x_offset:x_offset+old_w] = self.obstacle_map
        new_explored[y_offset:y_offset+old_h, x_offset:x_offset+old_w] = self.explored_map
        new_visited[y_offset:y_offset+old_h, x_offset:x_offset+old_w] = self.visited_map
        
        # Update maps
        self.obstacle_map = new_obstacle
        self.explored_map = new_explored
        self.visited_map = new_visited
        
        # Update origin to account for expansion
        self.origin_x -= (x_offset * self.resolution) / 100.0
        self.origin_z -= (y_offset * self.resolution) / 100.0
        
        self.map_size = new_size
    
    def _world_to_map(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """Convert world coordinates to map pixel coordinates."""
        # Offset from origin, convert to cm, then to pixels
        map_x = int((world_x - self.origin_x) * 100 / self.resolution + self.map_size // 2)
        map_y = int((world_z - self.origin_z) * 100 / self.resolution + self.map_size // 2)
        return map_x, map_y
    
    def _get_yaw_from_quaternion(self, quat: np.ndarray) -> float:
        """
        Extract yaw angle from quaternion [x, y, z, w].
        
        In Habitat, agent faces -Z direction at yaw=0.
        Returns angle in radians.
        """
        x, y, z, w = quat
        # Yaw around Y axis
        siny_cosp = 2.0 * (w * y + z * x)
        cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw
    
    def update(
        self,
        depth: np.ndarray,
        agent_position: np.ndarray,
        agent_rotation: np.ndarray
    ) -> Dict:
        """
        Update map with depth observation.
        
        Args:
            depth: HxW depth in meters
            agent_position: [x, y, z] world position (Habitat Y-up)
            agent_rotation: [x, y, z, w] quaternion
        
        Returns:
            Statistics dict
        """
        self.frame_count += 1
        
        # Set origin on first frame
        self._set_origin(agent_position[0], agent_position[2])
        
        # Check if we need to resize map based on agent position
        map_x, map_y = self._world_to_map(agent_position[0], agent_position[2])
        self._check_and_resize(map_x, map_y)
        
        # Recalculate after potential resize
        map_x, map_y = self._world_to_map(agent_position[0], agent_position[2])
        
        # Track trajectory
        self.trajectory.append(agent_position.copy())
        
        # Get agent yaw
        yaw = self._get_yaw_from_quaternion(agent_rotation)
        
        # Agent height (Y coordinate)
        agent_height = agent_position[1]
        
        # Step 1: Create local egocentric obstacle/explored maps
        local_size = self.config.local_map_size
        local_obstacle = np.zeros((local_size, local_size), dtype=np.float32)
        local_explored = np.zeros((local_size, local_size), dtype=np.float32)
        
        # Step 2: Project depth to local map
        # Subsample for speed (every 4th pixel)
        for v in range(0, self.img_height, 4):
            for u in range(0, self.img_width, 4):
                d = depth[v, u]
                
                # Skip invalid depth
                if d <= 0.1 or d > 5.0 or not np.isfinite(d):
                    continue
                
                # Back-project to 3D point in camera frame
                # Camera frame: X-right, Y-down, Z-forward
                z_cam = d  # Forward distance
                x_cam = (u - self.cx) * d / self.fx  # Right
                y_cam = (v - self.cy) * d / self.fy  # Down
                
                # Height in world frame (camera Y-down -> world Y-up)
                # Point height = agent_height - y_cam (because y_cam is down)
                point_height = agent_height - y_cam
                
                # Relative height from floor (assuming agent at standard height)
                rel_height = (point_height - (agent_height - self.config.agent_height_cm / 100)) * 100  # in cm
                
                # Convert to local map coordinates (egocentric)
                # In local map: center is agent, +Y is forward, +X is right
                local_x = x_cam * 100 / self.resolution  # cm to pixels
                local_y = z_cam * 100 / self.resolution  # forward = +Y in local map
                
                # Offset to center of local map
                px = int(local_x + local_size // 2)
                py = int(local_size // 2 - local_y)  # Flip Y for image coords
                
                # Check bounds
                if not (0 <= px < local_size and 0 <= py < local_size):
                    continue
                
                # Classify point
                if self.config.min_obstacle_height_cm < rel_height < self.config.max_obstacle_height_cm:
                    # Obstacle
                    local_obstacle[py, px] = 1.0
                
                if rel_height < self.config.floor_threshold_cm:
                    # Floor = explored
                    local_explored[py, px] = 1.0
                else:
                    # Any point = somewhat explored
                    local_explored[py, px] = max(local_explored[py, px], 0.3)
        
        # Step 3: Rotate local maps to align with world frame
        # Rotation angle: from egocentric to allocentric
        # Agent faces -Z in world at yaw=0, so we rotate by -yaw
        rot_angle_deg = np.degrees(yaw)
        
        center = (local_size // 2, local_size // 2)
        M = cv2.getRotationMatrix2D(center, rot_angle_deg, 1.0)
        
        rotated_obstacle = cv2.warpAffine(
            local_obstacle, M, (local_size, local_size),
            flags=cv2.INTER_NEAREST, borderValue=0
        )
        rotated_explored = cv2.warpAffine(
            local_explored, M, (local_size, local_size),
            flags=cv2.INTER_NEAREST, borderValue=0
        )
        
        # Step 4: Place rotated local map into global map
        map_x, map_y = self._world_to_map(agent_position[0], agent_position[2])
        
        half_size = local_size // 2
        
        # Calculate bounds
        g_x1 = max(0, map_x - half_size)
        g_x2 = min(self.map_size, map_x + half_size)
        g_y1 = max(0, map_y - half_size)
        g_y2 = min(self.map_size, map_y + half_size)
        
        l_x1 = half_size - (map_x - g_x1)
        l_x2 = half_size + (g_x2 - map_x)
        l_y1 = half_size - (map_y - g_y1)
        l_y2 = half_size + (g_y2 - map_y)
        
        # Update global maps (max operation to accumulate)
        if g_x2 > g_x1 and g_y2 > g_y1:
            self.obstacle_map[g_y1:g_y2, g_x1:g_x2] = np.maximum(
                self.obstacle_map[g_y1:g_y2, g_x1:g_x2],
                rotated_obstacle[l_y1:l_y2, l_x1:l_x2]
            )
            self.explored_map[g_y1:g_y2, g_x1:g_x2] = np.maximum(
                self.explored_map[g_y1:g_y2, g_x1:g_x2],
                rotated_explored[l_y1:l_y2, l_x1:l_x2]
            )
        
        # Mark agent position as visited
        if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
            cv2.circle(self.visited_map, (map_x, map_y), 2, 1, -1)
            cv2.circle(self.explored_map, (map_x, map_y), 5, 1.0, -1)
        
        return {
            'frame': self.frame_count,
            'agent_map_pos': (map_x, map_y),
            'yaw_deg': np.degrees(yaw)
        }
    
    def get_obstacle_map(self) -> np.ndarray:
        """Get binary obstacle map with cleanup."""
        binary = (self.obstacle_map > 0.5).astype(np.uint8)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def get_explored_map(self) -> np.ndarray:
        """Get binary explored map."""
        binary = (self.explored_map > 0.3).astype(np.uint8)
        
        # Fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def get_navigable_map(self) -> np.ndarray:
        """Get navigable space (explored and not obstacle)."""
        explored = self.get_explored_map()
        obstacle = self.get_obstacle_map()
        return ((explored == 1) & (obstacle == 0)).astype(np.uint8)
    
    def visualize(self) -> np.ndarray:
        """Create RGB visualization."""
        vis = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # Background
        vis[:, :] = [40, 40, 40]
        
        # Explored = light gray
        explored = self.explored_map > 0.3
        vis[explored] = [180, 180, 180]
        
        # Obstacles = dark red
        obstacles = self.obstacle_map > 0.5
        vis[obstacles] = [60, 60, 180]  # BGR -> appears red
        
        # Trajectory = green
        for i, pos in enumerate(self.trajectory):
            mx, my = self._world_to_map(pos[0], pos[2])
            if 0 <= mx < self.map_size and 0 <= my < self.map_size:
                cv2.circle(vis, (mx, my), 1, (0, 200, 0), -1)
        
        # Current position = blue
        if len(self.trajectory) > 0:
            pos = self.trajectory[-1]
            mx, my = self._world_to_map(pos[0], pos[2])
            if 0 <= mx < self.map_size and 0 <= my < self.map_size:
                cv2.circle(vis, (mx, my), 5, (255, 100, 0), -1)
        
        return vis
    
    def get_statistics(self) -> Dict:
        """Get map statistics."""
        explored = self.get_explored_map()
        obstacle = self.get_obstacle_map()
        
        return {
            'map_size': self.map_size,
            'resolution_cm': self.resolution,
            'total_cells': self.map_size ** 2,
            'explored_cells': int(np.sum(explored)),
            'explored_percent': 100.0 * np.sum(explored) / (self.map_size ** 2),
            'obstacle_cells': int(np.sum(obstacle)),
            'total_frames': self.frame_count
        }
    
    def save(self, output_dir: str):
        """Save maps."""
        import json
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'obstacle_map.npy', self.obstacle_map)
        np.save(output_dir / 'explored_map.npy', self.explored_map)
        
        vis = self.visualize()
        cv2.imwrite(str(output_dir / 'map_visualization.png'), vis)
        
        stats = self.get_statistics()
        with open(output_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)