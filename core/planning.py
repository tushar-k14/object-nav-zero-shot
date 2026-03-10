"""
Path Planning
=============
Single A* implementation used by all modules.

Usage:
    from core.planning import astar, inflate_obstacles

    path = astar(start, goal, obstacle_map, inflation=5)
"""

import numpy as np
import cv2
import heapq
from typing import List, Tuple, Optional


def inflate_obstacles(
    obstacle_map: np.ndarray,
    radius: int = 5,
    clear_points: List[Tuple[int, int]] = None,
    clear_radius: int = 3,
) -> np.ndarray:
    """
    Inflate obstacles for safe path planning.

    Args:
        obstacle_map: HxW array, >0.5 = obstacle
        radius: inflation radius in pixels
        clear_points: list of (x, y) points to force-clear (start/goal)
        clear_radius: radius around clear_points to force free

    Returns:
        Binary blocked map (uint8, 1=blocked)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    blocked = cv2.dilate((obstacle_map > 0.5).astype(np.uint8), kernel)

    if clear_points:
        h, w = blocked.shape
        for px, py in clear_points:
            r = clear_radius
            y0, y1 = max(0, py - r), min(h, py + r + 1)
            x0, x1 = max(0, px - r), min(w, px + r + 1)
            blocked[y0:y1, x0:x1] = 0

    return blocked


def astar(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacle_map: np.ndarray,
    inflation: int = 5,
    max_iterations: int = 80000,
    simplify_step: int = 3,
    allow_diagonal: bool = True,
    cost_map: np.ndarray = None,
) -> List[Tuple[int, int]]:
    """
    A* path planning on 2D grid.

    Args:
        start: (x, y) start position
        goal: (x, y) goal position
        obstacle_map: HxW, >0.5 = obstacle (gets inflated)
        inflation: obstacle inflation radius
        max_iterations: safety cap
        simplify_step: subsample path every N points (0 = no simplify)
        allow_diagonal: allow 8-connected movement
        cost_map: optional HxW additional cost layer (higher = avoid)

    Returns:
        List of (x, y) waypoints from start to goal. Empty if no path.
    """
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    blocked = inflate_obstacles(
        obstacle_map,
        radius=inflation,
        clear_points=[(sx, sy), (gx, gy)],
    )

    h, w = blocked.shape

    # Clamp to bounds
    sx, sy = np.clip(sx, 0, w - 1), np.clip(sy, 0, h - 1)
    gx, gy = np.clip(gx, 0, w - 1), np.clip(gy, 0, h - 1)

    if allow_diagonal:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heur(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    start_node = (sx, sy)
    goal_node = (gx, gy)
    goal_threshold = max(5.0, heur(start_node, goal_node) * 0.02)

    open_set = [(heur(start_node, goal_node), 0, start_node)]
    came_from = {}
    g_score = {start_node: 0.0}
    counter = 0

    while open_set and counter < max_iterations:
        _, _, cur = heapq.heappop(open_set)

        if heur(cur, goal_node) < goal_threshold:
            # Reconstruct
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path = path[::-1]
            if simplify_step > 1:
                path = path[::simplify_step]
                if path[-1] != (gx, gy):
                    path.append((gx, gy))
            return path

        for dx, dy in nbrs:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if blocked[ny, nx]:
                continue

            move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
            if cost_map is not None:
                move_cost += float(cost_map[ny, nx])

            ng = g_score[cur] + move_cost
            if (nx, ny) not in g_score or ng < g_score[(nx, ny)]:
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = cur
                counter += 1
                f = ng + heur((nx, ny), goal_node)
                heapq.heappush(open_set, (f, counter, (nx, ny)))

    return []  # No path found


def path_length_pixels(path: List[Tuple[int, int]]) -> float:
    """Compute total path length in pixels."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        total += np.sqrt(dx * dx + dy * dy)
    return total