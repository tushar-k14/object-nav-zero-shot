"""
Local Navigation Planner
========================

Low-level navigation for:
1. Following global path waypoints
2. Collision avoidance using costmap
3. Dynamic obstacle handling
4. Stuck detection and recovery

Works with habitat discrete actions: move_forward, turn_left, turn_right
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum


class LocalPlannerState(Enum):
    """Local planner states."""
    IDLE = "idle"
    FOLLOWING_PATH = "following"
    AVOIDING_OBSTACLE = "avoiding"
    RECOVERING = "recovering"
    GOAL_REACHED = "goal_reached"


@dataclass
class Costmap:
    """
    Local costmap for navigation.
    
    Layers:
    - Static obstacles from occupancy map
    - Inflation around obstacles
    - Dynamic obstacles from recent depth
    """
    static_cost: np.ndarray      # From occupancy map
    inflation_cost: np.ndarray   # Distance-based inflation
    dynamic_cost: np.ndarray     # Recent depth observations
    combined: np.ndarray         # Sum of all costs
    
    resolution: float  # meters per pixel
    origin: Tuple[float, float]  # World origin (x, z)
    size: int  # Pixels


class LocalPlanner:
    """
    Local navigation with costmap-based collision avoidance.
    
    Responsibilities:
    1. Convert global waypoints to actions
    2. Avoid collisions using local costmap
    3. Handle dynamic obstacles
    4. Detect and recover from stuck situations
    """
    
    def __init__(
        self,
        turn_angle: float = 10.0,       # Degrees per turn action
        forward_dist: float = 0.25,     # Meters per forward action
        waypoint_tolerance: float = 0.4, # Meters to consider waypoint reached
        obstacle_threshold: float = 0.5, # Meters for obstacle detection
        costmap_size: int = 100,        # Local costmap size in pixels
        costmap_resolution: float = 0.05, # Meters per costmap pixel
        inflation_radius: float = 0.3,  # Meters to inflate obstacles
    ):
        self.turn_angle = turn_angle
        self.forward_dist = forward_dist
        self.waypoint_tolerance = waypoint_tolerance
        self.obstacle_threshold = obstacle_threshold
        self.costmap_size = costmap_size
        self.costmap_resolution = costmap_resolution
        self.inflation_radius = inflation_radius
        
        # State
        self.state = LocalPlannerState.IDLE
        self.current_waypoint_idx = 0
        self.global_path = None
        
        # Stuck detection
        self.position_history = deque(maxlen=20)
        self.stuck_count = 0
        self.recovery_actions = []
        
        # Local costmap
        self.costmap: Optional[Costmap] = None
        
        # Statistics
        self.total_actions = 0
        self.collisions_avoided = 0
    
    def reset(self):
        """Reset planner state."""
        self.state = LocalPlannerState.IDLE
        self.current_waypoint_idx = 0
        self.global_path = None
        self.position_history.clear()
        self.stuck_count = 0
        self.recovery_actions = []
        self.costmap = None
        self.total_actions = 0
        self.collisions_avoided = 0
    
    def set_global_path(self, path):
        """Set new global path to follow."""
        self.global_path = path
        self.current_waypoint_idx = 0
        self.state = LocalPlannerState.FOLLOWING_PATH if path and path.is_valid else LocalPlannerState.IDLE
    
    def update_costmap(
        self,
        depth: np.ndarray,
        agent_position: np.ndarray,
        agent_rotation: np.ndarray,
        occupancy_map
    ):
        """
        Update local costmap from depth and occupancy.
        
        Args:
            depth: Current depth image
            agent_position: Agent world position [x, y, z]
            agent_rotation: Agent rotation quaternion
            occupancy_map: FixedOccupancyMap instance
        """
        size = self.costmap_size
        res = self.costmap_resolution
        
        # Initialize costmap centered on agent
        static_cost = np.zeros((size, size), dtype=np.float32)
        dynamic_cost = np.zeros((size, size), dtype=np.float32)
        
        # Get yaw
        yaw = self._get_yaw(agent_rotation)
        
        # Sample obstacles from occupancy map
        obstacle_map = occupancy_map.get_obstacle_map()
        
        # Convert costmap cells to world and check occupancy
        center = size // 2
        for cy in range(size):
            for cx in range(size):
                # Local offset from agent
                local_x = (cx - center) * res
                local_z = (cy - center) * res
                
                # Rotate to world frame
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                world_x = agent_position[0] + local_x * cos_y - local_z * sin_y
                world_z = agent_position[2] + local_x * sin_y + local_z * cos_y
                
                # Sample from occupancy map
                mx, my = occupancy_map._world_to_map(world_x, world_z)
                if (0 <= mx < obstacle_map.shape[1] and 
                    0 <= my < obstacle_map.shape[0]):
                    static_cost[cy, cx] = obstacle_map[my, mx]
        
        # Add dynamic obstacles from depth
        # Project depth to local costmap
        h, w = depth.shape
        fx = w / 2.0  # Approximate focal length for 90 HFOV
        
        for v in range(0, h, 4):  # Subsample for speed
            for u in range(0, w, 4):
                d = depth[v, u]
                if d <= 0.1 or d > 3.0:  # Only consider nearby obstacles
                    continue
                
                # Project to camera frame
                x_cam = (u - w/2) * d / fx
                z_cam = d
                
                # Convert to local costmap (agent at center, forward = up)
                cx = int(x_cam / res + center)
                cy = int(center - z_cam / res)  # Forward is negative Y in costmap
                
                if 0 <= cx < size and 0 <= cy < size:
                    dynamic_cost[cy, cx] = max(dynamic_cost[cy, cx], 1.0)
        
        # Inflate obstacles
        inflation_px = int(self.inflation_radius / res)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (inflation_px * 2 + 1, inflation_px * 2 + 1)
        )
        
        # Distance-based inflation
        combined = np.maximum(static_cost, dynamic_cost)
        binary = (combined > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        inflation_cost = np.clip(1.0 - dist / (inflation_px + 1), 0, 1)
        inflation_cost[combined > 0.5] = 1.0
        
        # Combine all layers
        combined_cost = np.maximum(static_cost, dynamic_cost)
        combined_cost = np.maximum(combined_cost, inflation_cost * 0.5)
        
        self.costmap = Costmap(
            static_cost=static_cost,
            inflation_cost=inflation_cost,
            dynamic_cost=dynamic_cost,
            combined=combined_cost,
            resolution=res,
            origin=(agent_position[0], agent_position[2]),
            size=size
        )
    
    def get_action(
        self,
        agent_position: np.ndarray,
        agent_rotation: np.ndarray,
        depth: np.ndarray,
        occupancy_map
    ) -> Optional[str]:
        """
        Get next navigation action.
        
        Args:
            agent_position: Current world position
            agent_rotation: Current rotation quaternion
            depth: Current depth image
            occupancy_map: FixedOccupancyMap instance
            
        Returns:
            Action string or None if goal reached/no path
        """
        self.total_actions += 1
        
        # Update position history for stuck detection
        self.position_history.append(agent_position[[0, 2]].copy())
        
        # Update local costmap
        self.update_costmap(depth, agent_position, agent_rotation, occupancy_map)
        
        # Handle recovery actions first
        if self.recovery_actions:
            return self.recovery_actions.pop(0)
        
        # Check if stuck
        if self._is_stuck():
            self._initiate_recovery()
            if self.recovery_actions:
                return self.recovery_actions.pop(0)
        
        # No path to follow
        if self.state == LocalPlannerState.IDLE or self.global_path is None:
            return None
        
        if not self.global_path.is_valid:
            self.state = LocalPlannerState.IDLE
            return None
        
        # Check if goal reached
        if self.current_waypoint_idx >= len(self.global_path.world_waypoints):
            self.state = LocalPlannerState.GOAL_REACHED
            return None
        
        # Get current target waypoint
        target = self.global_path.world_waypoints[self.current_waypoint_idx]
        
        # Check if waypoint reached
        dist_to_waypoint = np.linalg.norm(agent_position[[0, 2]] - target[[0, 2]])
        
        if dist_to_waypoint < self.waypoint_tolerance:
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx >= len(self.global_path.world_waypoints):
                self.state = LocalPlannerState.GOAL_REACHED
                print("Goal reached!")
                return None
            
            target = self.global_path.world_waypoints[self.current_waypoint_idx]
        
        # Compute action towards target with collision avoidance
        return self._compute_action(agent_position, agent_rotation, target, depth)
    
    def _compute_action(
        self,
        agent_position: np.ndarray,
        agent_rotation: np.ndarray,
        target: np.ndarray,
        depth: np.ndarray
    ) -> str:
        """Compute action with collision avoidance."""
        yaw = self._get_yaw(agent_rotation)
        
        # Direction to target
        dx = target[0] - agent_position[0]
        dz = target[2] - agent_position[2]
        target_angle = np.arctan2(dx, -dz)  # Habitat faces -Z at yaw=0
        
        # Angle difference
        angle_diff = target_angle - yaw
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Check if we need to turn
        if abs(angle_diff) > np.radians(15):
            return 'turn_left' if angle_diff > 0 else 'turn_right'
        
        # Check for obstacles ahead using depth
        h, w = depth.shape
        center_region = depth[h//3:2*h//3, w//3:2*w//3]
        min_depth = np.percentile(center_region[center_region > 0.1], 10) if (center_region > 0.1).any() else 10
        
        # Also check costmap
        if self.costmap is not None:
            center = self.costmap.size // 2
            ahead_region = self.costmap.combined[center-5:center, center-3:center+3]
            ahead_cost = np.max(ahead_region) if ahead_region.size > 0 else 0
        else:
            ahead_cost = 0
        
        # Decide action based on obstacles
        if min_depth < self.obstacle_threshold or ahead_cost > 0.7:
            self.collisions_avoided += 1
            self.state = LocalPlannerState.AVOIDING_OBSTACLE
            
            # Check left and right for clear path
            left_depth = np.mean(depth[h//3:2*h//3, :w//4])
            right_depth = np.mean(depth[h//3:2*h//3, 3*w//4:])
            
            if left_depth > right_depth:
                return 'turn_left'
            else:
                return 'turn_right'
        
        self.state = LocalPlannerState.FOLLOWING_PATH
        return 'move_forward'
    
    def _get_yaw(self, quaternion: np.ndarray) -> float:
        """Extract yaw from quaternion."""
        x, y, z, w = quaternion
        return np.arctan2(2 * (w * y + z * x), 1 - 2 * (x * x + y * y))
    
    def _is_stuck(self) -> bool:
        """Check if agent is stuck."""
        if len(self.position_history) < 15:
            return False
        
        positions = list(self.position_history)
        total_movement = sum(
            np.linalg.norm(positions[i] - positions[i-1])
            for i in range(1, len(positions))
        )
        
        if total_movement < 0.15:  # Less than 15cm in 15 steps
            self.stuck_count += 1
            return self.stuck_count > 2
        
        self.stuck_count = 0
        return False
    
    def _initiate_recovery(self):
        """Start recovery behavior."""
        print("Stuck detected - initiating recovery")
        self.state = LocalPlannerState.RECOVERING
        
        # Recovery: turn around, try moving, turn back
        self.recovery_actions = [
            'turn_left', 'turn_left', 'turn_left',  # 30 degrees
            'turn_left', 'turn_left', 'turn_left',  # 60 degrees
            'move_forward', 'move_forward',
            'turn_right', 'turn_right', 'turn_right', 'turn_right',
        ]
        
        self.stuck_count = 0
    
    def visualize_costmap(self) -> np.ndarray:
        """Visualize local costmap."""
        if self.costmap is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Colorize costmap
        cost = self.costmap.combined
        vis = np.zeros((cost.shape[0], cost.shape[1], 3), dtype=np.uint8)
        
        # Free space = green, obstacles = red, inflation = yellow
        vis[:, :, 1] = ((1 - cost) * 255).astype(np.uint8)  # Green for free
        vis[:, :, 2] = (cost * 255).astype(np.uint8)        # Red for obstacle
        
        # Mark agent position
        center = self.costmap.size // 2
        cv2.circle(vis, (center, center), 3, (255, 255, 0), -1)
        
        # Mark forward direction
        cv2.arrowedLine(vis, (center, center), (center, center - 10),
                       (255, 255, 0), 1, tipLength=0.3)
        
        return vis
    
    def get_statistics(self) -> Dict:
        """Get planner statistics."""
        return {
            'state': self.state.value,
            'current_waypoint': self.current_waypoint_idx,
            'total_waypoints': len(self.global_path.waypoints) if self.global_path else 0,
            'total_actions': self.total_actions,
            'collisions_avoided': self.collisions_avoided,
            'stuck_count': self.stuck_count
        }
