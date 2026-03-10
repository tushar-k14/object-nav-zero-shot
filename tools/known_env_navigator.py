#!/usr/bin/env python3
"""
Known-Environment Navigator
=============================

Navigation in a pre-mapped environment using annotations from MapEditorV2.

Pipeline:
  1. Load occupancy map + room/object annotations (from map_editor_v2.py)
  2. Accept natural language instruction ("go to the kitchen")
  3. Parse instruction → NavGoal (via InstructionParser)
  4. Resolve goal to map position (room center or object position)
  5. Plan path using A* on occupancy map
  6. Execute path with collision avoidance

This is the "known environment" mode for thesis:
  - Map is pre-built (from prior exploration or manual creation)
  - Rooms are annotated (via map editor)
  - Agent uses annotations for efficient navigation

Usage:
    # In Habitat simulation
    python known_env_navigator.py \
        --scene data/scene.glb \
        --map-image maps/floor_0.png \
        --annotations map_annotations.json \
        --instruction "go to the kitchen"

    # Interactive mode (type instructions)
    python known_env_navigator.py \
        --scene data/scene.glb \
        --map-image maps/floor_0.png \
        --annotations map_annotations.json \
        --interactive
"""

import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Import from project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.planning import astar
from perception.instruction_parser import InstructionParser, NavGoal, goal_to_action


class KnownEnvNavigator:
    """
    Navigator for pre-mapped environments with room/object annotations.
    """

    def __init__(
        self,
        obstacle_map: np.ndarray,
        annotations: dict,
        resolution_cm: float = 5.0,
        inflation_radius: int = 3,
    ):
        """
        Args:
            obstacle_map: HxW binary map (1=obstacle, 0=free)
            annotations: Dict from map_editor_v2.py export
            resolution_cm: Map resolution
            inflation_radius: A* obstacle inflation
        """
        self.obstacle_map = obstacle_map
        self.annotations = annotations
        self.resolution = resolution_cm
        self.inflation_radius = inflation_radius
        self.h, self.w = obstacle_map.shape

        # Parse annotations
        self.rooms: Dict[str, dict] = annotations.get('rooms', {})
        self.objects: List[dict] = annotations.get('objects', [])

        # Instruction parser (regex mode — lightweight)
        self.parser = InstructionParser(mode='regex')

        # Navigation state
        self.path: List[Tuple[int, int]] = []
        self.path_idx = 0
        self.current_goal: Optional[NavGoal] = None
        self.goal_position: Optional[Tuple[int, int]] = None
        self.status = 'idle'  # idle, navigating, reached, failed

        # Inflate obstacles for planning
        self._inflated = self._inflate_obstacles()

        print(f"Known-env navigator: {len(self.rooms)} rooms, {len(self.objects)} objects")
        for name, room in self.rooms.items():
            print(f"  Room: {name} (type={room['type']}, center={room.get('center')})")

    def _inflate_obstacles(self) -> np.ndarray:
        """Inflate obstacles for safe path planning."""
        if self.inflation_radius <= 0:
            return self.obstacle_map
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * self.inflation_radius + 1, 2 * self.inflation_radius + 1)
        )
        inflated = cv2.dilate(
            (self.obstacle_map > 0.5).astype(np.uint8), kernel
        )
        return inflated.astype(np.float32)

    # ----------------------------------------------------------
    # Instruction handling
    # ----------------------------------------------------------

    def set_instruction(self, instruction: str) -> dict:
        """
        Parse instruction and plan path to goal.

        Returns:
            dict with 'status', 'goal', 'message'
        """
        # Parse instruction
        self.current_goal = self.parser.parse(instruction)
        print(f"  Parsed: type={self.current_goal.type}, target={self.current_goal.target}")

        # Resolve to map position
        action = goal_to_action(
            self.current_goal,
            scene_graph=None,
            annotated_map=self.annotations,
        )

        if action is None:
            self.status = 'failed'
            return {'status': 'failed', 'goal': self.current_goal,
                    'message': f'Cannot understand: {instruction}'}

        if action.get('action') == 'stop':
            self.status = 'reached'
            return {'status': 'stop', 'goal': self.current_goal,
                    'message': 'Stopping as requested'}

        target_pos = action.get('target_pos')
        strategy = action.get('strategy', 'unknown')

        if target_pos is None:
            # Target not in annotations
            self.status = 'failed'
            return {'status': 'not_found', 'goal': self.current_goal,
                    'strategy': strategy,
                    'message': f'{self.current_goal.target} not found in map annotations'}

        self.goal_position = tuple(target_pos)
        return {'status': 'ready', 'goal': self.current_goal,
                'target_pos': self.goal_position, 'strategy': strategy,
                'message': f'Navigate to {self.current_goal.target} at {self.goal_position}'}

    def plan_path(self, agent_pos: Tuple[int, int]) -> bool:
        """
        Plan A* path from agent position to goal.

        Args:
            agent_pos: (mx, my) in map coordinates

        Returns:
            True if path found
        """
        if self.goal_position is None:
            return False

        self.path = astar(
            agent_pos, self.goal_position,
            self.obstacle_map,
            inflation=self.inflation_radius,
        )

        if self.path:
            self.path_idx = 0
            self.status = 'navigating'
            print(f"  Path planned: {len(self.path)} waypoints")
            return True

        self.status = 'failed'
        print(f"  No path found to {self.goal_position}")
        return False

    def get_next_waypoint(self) -> Optional[Tuple[int, int]]:
        """Get the next waypoint to navigate to."""
        if not self.path or self.path_idx >= len(self.path):
            return None
        return self.path[self.path_idx]

    def advance_waypoint(self, agent_pos: Tuple[int, int], tolerance: int = 5) -> str:
        """
        Check if agent reached current waypoint, advance if so.

        Returns:
            'navigating', 'reached', or waypoint coords
        """
        if not self.path or self.path_idx >= len(self.path):
            self.status = 'reached'
            return 'reached'

        wp = self.path[self.path_idx]
        dist = np.sqrt((agent_pos[0] - wp[0])**2 + (agent_pos[1] - wp[1])**2)

        if dist < tolerance:
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                self.status = 'reached'
                return 'reached'

        return 'navigating'

    # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------

    def render_map(self, agent_pos: Tuple[int, int] = None) -> np.ndarray:
        """Render annotated map with path and agent."""
        vis = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Obstacles
        vis[self.obstacle_map < 0.5] = [200, 200, 200]  # Free
        vis[self.obstacle_map >= 0.5] = [60, 60, 60]     # Obstacle

        # Room polygons
        from tools.map_editor_v2 import ROOM_COLORS
        for name, room in self.rooms.items():
            color = ROOM_COLORS.get(room['type'], (136, 136, 136))
            pts = np.array(room['polygon'], dtype=np.int32)
            if len(pts) >= 3:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.polylines(vis, [pts], True, color, 1)

            if room.get('center'):
                cx, cy = room['center']
                cv2.drawMarker(vis, (cx, cy), (255, 255, 255),
                               cv2.MARKER_CROSS, 8, 1)
                cv2.putText(vis, name, (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Objects
        for obj in self.objects:
            pos = tuple(obj['position'])
            cv2.circle(vis, pos, 4, (0, 255, 255), -1)
            cv2.putText(vis, obj['label'], (pos[0] + 6, pos[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # Path
        if self.path and len(self.path) > 1:
            for i in range(1, len(self.path)):
                p0 = self.path[i - 1]
                p1 = self.path[i]
                color = (0, 200, 255) if i > self.path_idx else (100, 100, 100)
                cv2.line(vis, p0, p1, color, 1)

        # Goal
        if self.goal_position:
            cv2.drawMarker(vis, self.goal_position, (0, 255, 0),
                           cv2.MARKER_STAR, 12, 2)

        # Agent
        if agent_pos:
            cv2.circle(vis, agent_pos, 6, (255, 100, 100), -1)
            cv2.circle(vis, agent_pos, 6, (255, 255, 255), 1)

        # Status
        status_text = f"Status: {self.status}"
        if self.current_goal:
            status_text += f" | Goal: {self.current_goal.target}"
        cv2.putText(vis, status_text, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis


# ================================================================
# Demo: Interactive mode with visualization
# ================================================================

def demo_interactive(nav: KnownEnvNavigator):
    """Run interactive demo — type instructions, see planned paths."""
    # Simulated agent position (center of map)
    agent_pos = (nav.w // 2, nav.h // 2)

    # Find a free position for the agent
    free_ys, free_xs = np.where(nav.obstacle_map < 0.5)
    if len(free_xs) > 0:
        idx = len(free_xs) // 2
        agent_pos = (int(free_xs[idx]), int(free_ys[idx]))

    print("\n" + "=" * 50)
    print("  KNOWN-ENVIRONMENT NAVIGATOR — Interactive Demo")
    print("=" * 50)
    print("  Type instructions like:")
    print("    'go to the kitchen'")
    print("    'find a toilet'")
    print("    'navigate to bedroom'")
    print("    'go to room 3'")
    print("  Type 'quit' to exit")
    print("=" * 50)

    window = "Known-Env Navigator"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    # Click to set agent position
    def mouse_cb(event, x, y, flags, param):
        nonlocal agent_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            if nav.obstacle_map[y, x] < 0.5:
                agent_pos = (x, y)
                print(f"  Agent position: {agent_pos}")
                if nav.goal_position:
                    nav.plan_path(agent_pos)

    cv2.setMouseCallback(window, mouse_cb)

    while True:
        vis = nav.render_map(agent_pos)

        # Instructions
        cv2.putText(vis, "Click: set agent pos | Type instruction in terminal",
                    (5, nav.h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        cv2.imshow(window, vis)
        key = cv2.waitKey(100)

        if key == ord('q'):
            break

        # Check for terminal input (non-blocking)
        try:
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                instruction = input().strip()
                if instruction.lower() == 'quit':
                    break
                result = nav.set_instruction(instruction)
                print(f"  → {result['message']}")
                if result['status'] == 'ready':
                    nav.plan_path(agent_pos)
        except (ImportError, OSError):
            # select doesn't work on Windows, use simple input
            pass

    cv2.destroyAllWindows()


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Known-Environment Navigator')
    parser.add_argument('--map-image', type=str, help='Occupancy map image (PNG)')
    parser.add_argument('--map-pt', type=str, help='Navigable map tensor (.pt) — preferred for planning')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Room/object annotations JSON (from map_editor_v2)')
    parser.add_argument('--instruction', type=str, help='Navigation instruction')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode with visualization')
    parser.add_argument('--resolution', type=float, default=5.0)
    args = parser.parse_args()

    # Load annotations
    with open(args.annotations) as f:
        annotations = json.load(f)

    # Load obstacle map (priority: .pt > .png > empty)
    obstacle_map = None

    if args.map_pt:
        import torch
        tensor = torch.load(args.map_pt, map_location='cpu')
        obstacle_map = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
        if obstacle_map.ndim > 2:
            obstacle_map = obstacle_map[0]  # Take first channel if multi-channel
        obstacle_map = obstacle_map.astype(np.float32)
        print(f"Loaded obstacle map from .pt: {obstacle_map.shape}")

    elif args.map_image:
        img = cv2.imread(args.map_image, cv2.IMREAD_GRAYSCALE)
        # Dark pixels = obstacles, light = free
        obstacle_map = (img < 100).astype(np.float32)
        print(f"Loaded obstacle map from image: {obstacle_map.shape}")

    else:
        # Create from annotation map_size
        ms = annotations.get('map_size', [480, 480])
        obstacle_map = np.zeros((ms[1], ms[0]), dtype=np.float32)
        print(f"Created empty obstacle map: {obstacle_map.shape}")

    nav = KnownEnvNavigator(obstacle_map, annotations, args.resolution)

    if args.interactive:
        demo_interactive(nav)
    elif args.instruction:
        result = nav.set_instruction(args.instruction)
        print(f"Result: {result}")
        if result['status'] == 'ready':
            # Demo: plan from center
            free_ys, free_xs = np.where(obstacle_map < 0.5)
            if len(free_xs) > 0:
                agent = (int(free_xs[len(free_xs)//2]), int(free_ys[len(free_ys)//2]))
                nav.plan_path(agent)
                vis = nav.render_map(agent)
                out_path = 'nav_plan.png'
                cv2.imwrite(out_path, vis)
                print(f"Plan saved to {out_path}")
    else:
        print("Use --instruction or --interactive")


if __name__ == '__main__':
    main()