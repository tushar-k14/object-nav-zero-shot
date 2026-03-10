"""
Global Navigation Planner
=========================

High-level navigation planning using:
1. Semantic map for goal specification
2. Scene graph for object-based goals
3. A* path planning on occupancy map

This handles "go to the kitchen" or "find the toilet" type goals.
Local navigation (collision avoidance) is handled separately.
"""

import numpy as np
import cv2
import heapq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GoalType(Enum):
    """Types of navigation goals."""
    POSITION = "position"       # Go to specific (x, z) position
    OBJECT = "object"           # Go to specific object class
    ROOM = "room"               # Go to room type (kitchen, bathroom, etc.)
    EXPLORE = "explore"         # Go to nearest unexplored area


@dataclass
class NavigationGoal:
    """A navigation goal specification."""
    goal_type: GoalType
    target: str = None              # Object class or room type
    position: np.ndarray = None     # World position [x, y, z]
    map_position: Tuple[int, int] = None  # Map coordinates
    tolerance: float = 0.5          # Distance tolerance in meters
    floor: int = 0                  # Target floor


@dataclass
class GlobalPath:
    """A planned global path."""
    waypoints: List[Tuple[int, int]]  # Map coordinates
    world_waypoints: List[np.ndarray]  # World coordinates
    goal: NavigationGoal
    total_distance: float
    is_valid: bool = True


class GlobalPlanner:
    """
    Global path planner using semantic map and scene graph.
    
    Responsibilities:
    1. Parse goal specifications (object, room, position)
    2. Find target location from semantic map/scene graph
    3. Plan global path using A* on occupancy map
    4. Provide waypoints for local planner
    """
    
    def __init__(
        self,
        inflation_radius: int = 5,      # Pixels to inflate obstacles
        path_simplification: int = 5,   # Keep every Nth waypoint
        max_planning_iterations: int = 50000
    ):
        self.inflation_radius = inflation_radius
        self.path_simplification = path_simplification
        self.max_iterations = max_planning_iterations
        
        # Cached inflated map
        self._cached_obstacle_map = None
        self._cached_inflated_map = None
    
    def set_goal(
        self,
        goal_type: GoalType,
        semantic_mapper,
        scene_graph,
        target: str = None,
        position: np.ndarray = None
    ) -> Optional[NavigationGoal]:
        """
        Create a navigation goal.
        
        Args:
            goal_type: Type of goal (POSITION, OBJECT, ROOM, EXPLORE)
            semantic_mapper: EnhancedSemanticMapperV2 instance
            scene_graph: SceneGraph instance
            target: Object class or room type (for OBJECT/ROOM goals)
            position: World position (for POSITION goal)
            
        Returns:
            NavigationGoal or None if goal cannot be set
        """
        occ_map = semantic_mapper.occupancy_map
        
        if goal_type == GoalType.POSITION:
            if position is None:
                print("Position goal requires position argument")
                return None
            
            map_pos = occ_map._world_to_map(position[0], position[2])
            return NavigationGoal(
                goal_type=goal_type,
                position=position,
                map_position=map_pos,
                floor=semantic_mapper.current_floor
            )
        
        elif goal_type == GoalType.OBJECT:
            if target is None:
                print("Object goal requires target class")
                return None
            
            # Find object from scene graph
            objects = scene_graph.get_objects_by_class(target.lower())
            
            if not objects:
                # Try semantic mapper directly
                confirmed = semantic_mapper.get_confirmed_objects()
                objects = [o for o in confirmed.values() 
                          if o.class_name.lower() == target.lower()]
            
            if not objects:
                print(f"No '{target}' found in scene")
                return None
            
            # Find nearest on current floor
            current_floor = semantic_mapper.current_floor
            floor_objects = [o for o in objects 
                            if getattr(o, 'floor_level', current_floor) == current_floor]
            
            if not floor_objects:
                floor_objects = objects  # Use any floor if none on current
            
            # Get position of first matching object
            if hasattr(floor_objects[0], 'mean_position'):
                pos = floor_objects[0].mean_position
            else:
                pos = floor_objects[0].position
            
            map_pos = occ_map._world_to_map(pos[0], pos[2])
            
            return NavigationGoal(
                goal_type=goal_type,
                target=target.lower(),
                position=pos,
                map_position=map_pos,
                floor=getattr(floor_objects[0], 'floor_level', current_floor)
            )
        
        elif goal_type == GoalType.ROOM:
            if target is None:
                print("Room goal requires target room type")
                return None
            
            # Find objects in room from scene graph
            room_objects = scene_graph.get_objects_in_room(target.lower())
            
            if not room_objects:
                print(f"No objects found in '{target}'")
                return None
            
            # Compute room center
            positions = [o.position for o in room_objects]
            center = np.mean(positions, axis=0)
            map_pos = occ_map._world_to_map(center[0], center[2])
            
            return NavigationGoal(
                goal_type=goal_type,
                target=target.lower(),
                position=center,
                map_position=map_pos,
                floor=semantic_mapper.current_floor
            )
        
        elif goal_type == GoalType.EXPLORE:
            # Find nearest frontier (unexplored area)
            frontier_pos = self._find_nearest_frontier(semantic_mapper)
            
            if frontier_pos is None:
                print("No unexplored areas found")
                return None
            
            return NavigationGoal(
                goal_type=goal_type,
                map_position=frontier_pos,
                floor=semantic_mapper.current_floor
            )
        
        return None
    
    def _find_nearest_frontier(self, semantic_mapper) -> Optional[Tuple[int, int]]:
        """Find nearest unexplored area."""
        occ_map = semantic_mapper.occupancy_map
        explored = occ_map.get_explored_map()
        obstacle = occ_map.get_obstacle_map()
        
        # Find frontiers
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate((explored > 0).astype(np.uint8), kernel)
        frontier = ((dilated > 0) & (explored == 0) & (obstacle == 0)).astype(np.uint8)
        frontier = cv2.morphologyEx(frontier, cv2.MORPH_OPEN, kernel)
        
        # Get frontier centroids
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier)
        
        if num_labels <= 1:
            return None
        
        # Find nearest to agent
        if occ_map.trajectory:
            agent_pos = occ_map.trajectory[-1]
            agent_map = occ_map._world_to_map(agent_pos[0], agent_pos[2])
        else:
            agent_map = (occ_map.map_size // 2, occ_map.map_size // 2)
        
        best_centroid = None
        best_dist = float('inf')
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:  # Skip small frontiers
                continue
            
            cx, cy = int(centroids[i, 0]), int(centroids[i, 1])
            dist = np.sqrt((cx - agent_map[0])**2 + (cy - agent_map[1])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_centroid = (cx, cy)
        
        return best_centroid
    
    def plan_path(
        self,
        start_position: np.ndarray,
        goal: NavigationGoal,
        semantic_mapper
    ) -> GlobalPath:
        """
        Plan path from start to goal using A*.
        
        Args:
            start_position: Current agent world position [x, y, z]
            goal: NavigationGoal to reach
            semantic_mapper: For occupancy map access
            
        Returns:
            GlobalPath with waypoints
        """
        occ_map = semantic_mapper.occupancy_map
        obstacle = occ_map.get_obstacle_map()
        
        # Inflate obstacles
        if (self._cached_obstacle_map is None or 
            not np.array_equal(self._cached_obstacle_map, obstacle)):
            kernel = np.ones((self.inflation_radius * 2 + 1,) * 2, np.uint8)
            self._cached_inflated_map = cv2.dilate(
                (obstacle > 0).astype(np.uint8), kernel
            )
            self._cached_obstacle_map = obstacle.copy()
        
        inflated = self._cached_inflated_map
        
        # Start and goal in map coordinates
        start = occ_map._world_to_map(start_position[0], start_position[2])
        goal_map = goal.map_position
        
        # Check bounds
        h, w = inflated.shape
        if not (0 <= start[0] < w and 0 <= start[1] < h):
            return GlobalPath([], [], goal, 0, is_valid=False)
        if not (0 <= goal_map[0] < w and 0 <= goal_map[1] < h):
            return GlobalPath([], [], goal, 0, is_valid=False)
        
        # A* search
        path = self._astar(start, goal_map, inflated)
        
        if not path:
            # Try with less inflation
            kernel_small = np.ones((3, 3), np.uint8)
            inflated_small = cv2.dilate((obstacle > 0).astype(np.uint8), kernel_small)
            path = self._astar(start, goal_map, inflated_small)
        
        if not path:
            return GlobalPath([], [], goal, 0, is_valid=False)
        
        # Simplify path
        simplified = path[::self.path_simplification]
        if path[-1] not in simplified:
            simplified.append(path[-1])
        
        # Convert to world coordinates
        world_waypoints = []
        for mx, my in simplified:
            # Approximate world position (inverse of _world_to_map)
            wx = (mx - occ_map.map_size // 2) * occ_map.resolution / 100.0 + occ_map.origin_x
            wz = (my - occ_map.map_size // 2) * occ_map.resolution / 100.0 + occ_map.origin_z
            world_waypoints.append(np.array([wx, start_position[1], wz]))
        
        # Calculate total distance
        total_dist = sum(
            np.linalg.norm(np.array(simplified[i]) - np.array(simplified[i-1]))
            for i in range(1, len(simplified))
        ) * occ_map.resolution / 100.0  # Convert to meters
        
        return GlobalPath(
            waypoints=simplified,
            world_waypoints=world_waypoints,
            goal=goal,
            total_distance=total_dist,
            is_valid=True
        )
    
    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacle_map: np.ndarray
    ) -> List[Tuple[int, int]]:
        """A* pathfinding."""
        h, w = obstacle_map.shape
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        open_set = [(heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}
        counter = 0
        
        while open_set and counter < self.max_iterations:
            counter += 1
            _, _, current = heapq.heappop(open_set)
            
            # Goal reached (with tolerance)
            if heuristic(current, goal) < 5:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                
                # Bounds check
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                
                # Obstacle check
                if obstacle_map[ny, nx] > 0:
                    continue
                
                # Movement cost (diagonal = sqrt(2))
                move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                tentative_g = g_score[current] + move_cost
                
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, counter, (nx, ny)))
        
        return []  # No path found
    
    def visualize_path(
        self,
        map_vis: np.ndarray,
        path: GlobalPath,
        current_waypoint: int = 0
    ) -> np.ndarray:
        """Draw path on map visualization."""
        vis = map_vis.copy()
        
        if not path.is_valid or not path.waypoints:
            return vis
        
        # Draw path line
        for i in range(len(path.waypoints) - 1):
            p1 = path.waypoints[i]
            p2 = path.waypoints[i + 1]
            color = (255, 200, 0) if i >= current_waypoint else (100, 100, 100)
            cv2.line(vis, p1, p2, color, 2)
        
        # Draw waypoints
        for i, wp in enumerate(path.waypoints):
            if i < current_waypoint:
                color = (100, 100, 100)  # Past
            elif i == current_waypoint:
                color = (0, 255, 255)    # Current target
            else:
                color = (255, 200, 0)    # Future
            cv2.circle(vis, wp, 4, color, -1)
        
        # Draw goal
        if path.goal.map_position:
            cv2.circle(vis, path.goal.map_position, 10, (255, 0, 255), 2)
            
            # Label
            if path.goal.target:
                label = path.goal.target
            elif path.goal.goal_type == GoalType.EXPLORE:
                label = "explore"
            else:
                label = "goal"
            
            cv2.putText(vis, label, 
                       (path.goal.map_position[0] + 12, path.goal.map_position[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        return vis


# Convenience functions for common goals
def go_to_object(planner, mapper, scene_graph, object_class: str) -> Optional[GlobalPath]:
    """Plan path to nearest object of given class."""
    goal = planner.set_goal(GoalType.OBJECT, mapper, scene_graph, target=object_class)
    if goal is None:
        return None
    
    if mapper.occupancy_map.trajectory:
        start = mapper.occupancy_map.trajectory[-1]
    else:
        return None
    
    return planner.plan_path(start, goal, mapper)


def go_to_room(planner, mapper, scene_graph, room_type: str) -> Optional[GlobalPath]:
    """Plan path to room of given type."""
    goal = planner.set_goal(GoalType.ROOM, mapper, scene_graph, target=room_type)
    if goal is None:
        return None
    
    if mapper.occupancy_map.trajectory:
        start = mapper.occupancy_map.trajectory[-1]
    else:
        return None
    
    return planner.plan_path(start, goal, mapper)


def go_to_position(planner, mapper, position: np.ndarray) -> Optional[GlobalPath]:
    """Plan path to specific position."""
    goal = planner.set_goal(GoalType.POSITION, mapper, None, position=position)
    if goal is None:
        return None
    
    if mapper.occupancy_map.trajectory:
        start = mapper.occupancy_map.trajectory[-1]
    else:
        return None
    
    return planner.plan_path(start, goal, mapper)


def explore_unknown(planner, mapper) -> Optional[GlobalPath]:
    """Plan path to nearest unexplored area."""
    goal = planner.set_goal(GoalType.EXPLORE, mapper, None)
    if goal is None:
        return None
    
    if mapper.occupancy_map.trajectory:
        start = mapper.occupancy_map.trajectory[-1]
    else:
        return None
    
    return planner.plan_path(start, goal, mapper)
