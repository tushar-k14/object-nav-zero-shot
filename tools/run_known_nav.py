#!/usr/bin/env python3
"""
Known-Environment Navigation in Habitat-Sim
=============================================

Runs the known_env_navigator inside Habitat simulator.
Uses pre-built maps + annotations from map_editor_v2 + YOLO-detected objects.

Pipeline:
  1. Load scene in Habitat
  2. Load pre-built navigable_map.pt + map_annotations.json
  3. Optionally load YOLO-detected objects (objects.json from exploration)
  4. Accept natural language instruction
  5. Plan A* path and execute in simulation
  6. Visualize RGB + map with path overlay

Usage:
    python tools/run_known_nav.py \
        --scene data/scene_datasets/hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb \
        --map-pt my_map/floor_-1/navigable_map.pt \
        --annotations my_map/floor_-1/map_annotations.json \
        --objects my_map/floor_-1/objects.json \
        --instruction "go to the kitchen"

    # Interactive mode
    python tools/run_known_nav.py \
        --scene ... --map-pt ... --annotations ... --interactive
"""

import argparse
import numpy as np
import cv2
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.planning import astar
from perception.instruction_parser import InstructionParser, goal_to_action


# ================================================================
# Habitat Environment (lightweight wrapper)
# ================================================================

class HabitatEnv:
    """Minimal Habitat-sim wrapper for known-env navigation."""

    def __init__(self, scene_path: str, scene_dataset_config: str = None):
        import habitat_sim

        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = scene_path
        if scene_dataset_config:
            cfg.scene_dataset_config_file = scene_dataset_config

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        # RGB sensor
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [480, 640]
        rgb_spec.position = [0.0, 1.5, 0.0]
        rgb_spec.hfov = 90

        # Depth sensor
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [480, 640]
        depth_spec.position = [0.0, 1.5, 0.0]
        depth_spec.hfov = 90

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(0.25)),
            "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(10.0)),
            "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(10.0)),
        }

        sim_cfg = habitat_sim.Configuration(cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(sim_cfg)
        self.agent = self.sim.get_agent(0)

    def reset(self, position=None, rotation=None):
        obs = self.sim.reset()
        if position is not None:
            state = self.agent.get_state()
            state.position = position
            if rotation is not None:
                state.rotation = rotation
            self.agent.set_state(state)
            obs = self.sim.get_sensor_observations()
        return self._extract(obs)

    def step(self, action: str):
        obs = self.sim.step(action)
        return self._extract(obs)

    def get_position(self):
        return np.array(self.agent.get_state().position, dtype=np.float32)

    def get_rotation(self):
        r = self.agent.get_state().rotation
        return np.array([r.x, r.y, r.z, r.w], dtype=np.float32)

    def _extract(self, obs):
        rgb = obs['rgb'][:, :, :3]
        depth = obs['depth']
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return rgb, depth

    def close(self):
        self.sim.close()


# ================================================================
# Map coordinate transforms
# ================================================================

class MapTransform:
    """Convert between world coordinates and map pixels."""

    def __init__(self, map_size: int, resolution_cm: float = 5.0,
                 origin_world: np.ndarray = None):
        self.map_size = map_size
        self.resolution = resolution_cm / 100.0  # Convert to meters
        self.origin_world = origin_world  # Will be set from first agent position

    def set_origin(self, world_pos: np.ndarray):
        """Set map origin from agent's starting world position."""
        self.origin_world = world_pos.copy()

    def world_to_map(self, wx: float, wz: float) -> tuple:
        """Convert world (x, z) to map (mx, my)."""
        if self.origin_world is None:
            return (self.map_size // 2, self.map_size // 2)
        dx = wx - self.origin_world[0]
        dz = wz - self.origin_world[2]
        mx = int(self.map_size // 2 + dx / self.resolution)
        my = int(self.map_size // 2 + dz / self.resolution)
        mx = np.clip(mx, 0, self.map_size - 1)
        my = np.clip(my, 0, self.map_size - 1)
        return (mx, my)

    def map_to_world(self, mx: int, my: int, y_height: float = 0.0) -> np.ndarray:
        """Convert map (mx, my) to world (x, y, z)."""
        if self.origin_world is None:
            return np.array([0, y_height, 0], dtype=np.float32)
        dx = (mx - self.map_size // 2) * self.resolution
        dz = (my - self.map_size // 2) * self.resolution
        wx = self.origin_world[0] + dx
        wz = self.origin_world[2] + dz
        return np.array([wx, y_height, wz], dtype=np.float32)


# ================================================================
# Known-Nav Runner
# ================================================================

def get_yaw(rotation):
    """Extract yaw from quaternion [x, y, z, w]."""
    x, y, z, w = rotation
    siny = 2.0 * (w * y - z * x)
    return float(np.arctan2(siny, 1.0 - 2.0 * (y * y + z * z)))


def angle_to_target(ax, az, yaw, tx, tz):
    """Compute signed angle from agent heading to target."""
    dx = tx - ax
    dz = tz - az
    target_angle = np.arctan2(-dx, -dz)
    diff = target_angle - yaw
    return (diff + np.pi) % (2 * np.pi) - np.pi


class KnownNavRunner:
    """Run known-environment navigation in Habitat."""

    def __init__(self, env: HabitatEnv, obstacle_map: np.ndarray,
                 annotations: dict, detected_objects: list = None,
                 resolution_cm: float = 5.0):
        self.env = env
        self.obstacle_map = obstacle_map
        self.annotations = annotations
        self.resolution = resolution_cm

        # Merge detected objects into annotations
        if detected_objects:
            existing_labels = {o['label'] for o in self.annotations.get('objects', [])}
            for obj in detected_objects:
                if isinstance(obj, dict):
                    label = obj.get('class_name', obj.get('label', 'unknown')).lower()
                    pos = obj.get('map_position', obj.get('position'))
                else:
                    label = str(obj).lower()
                    pos = None

                if label not in existing_labels and pos is not None:
                    self.annotations.setdefault('objects', []).append({
                        'label': label,
                        'position': pos[:2] if len(pos) > 2 else pos,
                        'room': None,
                    })
            print(f"  Merged {len(detected_objects)} detected objects into annotations")
            print(f"  Total objects: {len(self.annotations.get('objects', []))}")

        self.map_size = obstacle_map.shape[0]
        self.transform = MapTransform(self.map_size, resolution_cm)
        self.parser = InstructionParser(mode='regex')

        # Navigation state
        self.path_world = []
        self.path_map = []
        self.path_idx = 0
        self.goal_map = None
        self.status = 'idle'
        self.step_count = 0

    def set_origin_from_agent(self):
        """Set map origin from current agent position."""
        pos = self.env.get_position()
        self.transform.set_origin(pos)
        print(f"  Map origin set to world pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    def navigate_to(self, instruction: str) -> dict:
        """Parse instruction and plan path."""
        goal = self.parser.parse(instruction)
        action = goal_to_action(goal, annotated_map=self.annotations)

        if action is None or action.get('target_pos') is None:
            self.status = 'failed'
            msg = action.get('message', 'Target not found') if action else 'Parse failed'
            print(f"  Navigation failed: {msg}")
            return {'status': 'failed', 'message': msg}

        if action.get('action') == 'stop':
            self.status = 'stopped'
            return {'status': 'stopped'}

        self.goal_map = tuple(action['target_pos'])
        agent_pos = self.env.get_position()
        agent_map = self.transform.world_to_map(agent_pos[0], agent_pos[2])

        print(f"  Goal: {goal.target} at map pos {self.goal_map}")
        print(f"  Agent at map pos {agent_map}")

        # Plan A* path
        self.path_map = astar(
            agent_map, self.goal_map,
            self.obstacle_map, inflation=3,
        )

        if not self.path_map:
            self.status = 'no_path'
            print(f"  No path found!")
            return {'status': 'no_path'}

        # Convert path to world coords
        y_height = agent_pos[1]
        self.path_world = [
            self.transform.map_to_world(mx, my, y_height)
            for mx, my in self.path_map
        ]
        self.path_idx = 0
        self.status = 'navigating'
        self.step_count = 0
        print(f"  Path planned: {len(self.path_map)} waypoints")
        return {'status': 'navigating', 'path_length': len(self.path_map)}

    def step_navigation(self, rgb, depth) -> str:
        """Execute one step of navigation. Returns action string."""
        if self.status != 'navigating' or self.path_idx >= len(self.path_world):
            self.status = 'reached'
            return 'stop'

        self.step_count += 1
        pos = self.env.get_position()
        rot = self.env.get_rotation()
        yaw = get_yaw(rot)

        target_wp = self.path_world[self.path_idx]
        dist = np.sqrt((pos[0] - target_wp[0])**2 + (pos[2] - target_wp[2])**2)

        # Advance waypoint if close enough
        if dist < 0.3:
            self.path_idx += 1
            if self.path_idx >= len(self.path_world):
                self.status = 'reached'
                print(f"  Reached goal in {self.step_count} steps!")
                return 'stop'
            target_wp = self.path_world[self.path_idx]

        # Check if at final goal
        if self.goal_map:
            goal_world = self.transform.map_to_world(self.goal_map[0], self.goal_map[1], pos[1])
            goal_dist = np.sqrt((pos[0] - goal_world[0])**2 + (pos[2] - goal_world[2])**2)
            if goal_dist < 0.5:
                self.status = 'reached'
                print(f"  Reached goal (dist={goal_dist:.2f}m) in {self.step_count} steps!")
                return 'stop'

        # Turn toward waypoint
        diff = angle_to_target(pos[0], pos[2], yaw, target_wp[0], target_wp[2])

        # Check for obstacle ahead
        h, w = depth.shape[:2]
        center = depth[h//3:2*h//3, w//3:2*w//3]
        valid = center[(center > 0.1) & (center < 10)]
        front_dist = float(np.percentile(valid, 5)) if len(valid) > 100 else 10.0

        if front_dist < 0.3:
            return 'turn_left' if diff >= 0 else 'turn_right'

        if abs(diff) > np.radians(15):
            return 'turn_left' if diff > 0 else 'turn_right'

        return 'move_forward'

    def render_map_view(self, vis_size: int = 400) -> np.ndarray:
        """Render top-down map with path and agent."""
        from tools.map_editor_v2 import ROOM_COLORS

        h, w = self.obstacle_map.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[self.obstacle_map < 0.5] = [200, 200, 200]
        vis[self.obstacle_map >= 0.5] = [60, 60, 60]

        # Rooms
        for name, room in self.annotations.get('rooms', {}).items():
            color = ROOM_COLORS.get(room['type'], (136, 136, 136))
            pts = np.array(room['polygon'], dtype=np.int32)
            if len(pts) >= 3:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            if room.get('center'):
                cx, cy = room['center']
                cv2.putText(vis, name.split('_')[0], (cx - 15, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Objects
        for obj in self.annotations.get('objects', []):
            pos = tuple(obj['position'][:2])
            cv2.circle(vis, pos, 3, (0, 255, 255), -1)

        # Path
        if self.path_map:
            for i in range(1, len(self.path_map)):
                color = (0, 200, 255) if i > self.path_idx else (80, 80, 80)
                cv2.line(vis, self.path_map[i-1], self.path_map[i], color, 1)

        # Goal
        if self.goal_map:
            cv2.drawMarker(vis, self.goal_map, (0, 255, 0), cv2.MARKER_STAR, 10, 2)

        # Agent
        pos = self.env.get_position()
        agent_map = self.transform.world_to_map(pos[0], pos[2])
        cv2.circle(vis, agent_map, 5, (255, 100, 100), -1)
        cv2.circle(vis, agent_map, 5, (255, 255, 255), 1)

        # Status
        cv2.putText(vis, f"Status: {self.status} | Steps: {self.step_count}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Crop to relevant area and resize
        ys, xs = np.where(vis[:, :, 0] > 30)
        if len(xs) > 0:
            margin = 30
            x1 = max(0, xs.min() - margin)
            x2 = min(w, xs.max() + margin)
            y1 = max(0, ys.min() - margin)
            y2 = min(h, ys.max() + margin)
            vis = vis[y1:y2, x1:x2]

        # Resize to fit
        vh, vw = vis.shape[:2]
        scale = min(vis_size / vw, vis_size / vh)
        vis = cv2.resize(vis, (int(vw * scale), int(vh * scale)))

        return vis


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Known-Env Navigation in Habitat')
    parser.add_argument('--scene', type=str, required=True, help='Habitat scene GLB file')
    parser.add_argument('--scene-dataset-config', type=str, default=None)
    parser.add_argument('--map-pt', type=str, required=True, help='navigable_map.pt')
    parser.add_argument('--annotations', type=str, required=True, help='map_annotations.json')
    parser.add_argument('--objects', type=str, default=None, help='objects.json from exploration')
    parser.add_argument('--instruction', type=str, default=None)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--start-pos', type=float, nargs=3, default=None)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--resolution', type=float, default=5.0)
    args = parser.parse_args()

    # Load obstacle map
    import torch
    obstacle_map = torch.load(args.map_pt, map_location='cpu')
    if hasattr(obstacle_map, 'numpy'):
        obstacle_map = obstacle_map.numpy()
    obstacle_map = np.array(obstacle_map, dtype=np.float32)
    if obstacle_map.ndim > 2:
        obstacle_map = obstacle_map[0]
    print(f"Obstacle map: {obstacle_map.shape}")

    # Load annotations
    with open(args.annotations) as f:
        annotations = json.load(f)

    # Load detected objects (from exploration phase)
    detected_objects = None
    if args.objects and Path(args.objects).exists():
        with open(args.objects) as f:
            detected_objects = json.load(f)
        print(f"Loaded {len(detected_objects)} detected objects from exploration")

    # Setup Habitat
    print("Loading Habitat scene...")
    env = HabitatEnv(args.scene, args.scene_dataset_config)

    # Set start position
    if args.start_pos:
        rgb, depth = env.reset(position=args.start_pos)
    else:
        rgb, depth = env.reset()
        # Try random navigable point
        if env.sim.pathfinder.is_loaded:
            nav_point = env.sim.pathfinder.get_random_navigable_point()
            state = env.agent.get_state()
            state.position = nav_point
            env.agent.set_state(state)
            rgb, depth = env.step("move_forward")  # Get fresh obs

    runner = KnownNavRunner(
        env, obstacle_map, annotations,
        detected_objects=detected_objects,
        resolution_cm=args.resolution,
    )
    runner.set_origin_from_agent()

    if args.interactive:
        _run_interactive(runner, env, args.max_steps)
    elif args.instruction:
        _run_single(runner, env, args.instruction, args.max_steps)
    else:
        print("Use --instruction or --interactive")

    env.close()


def _run_single(runner, env, instruction, max_steps):
    """Run a single instruction to completion."""
    print(f"\nInstruction: '{instruction}'")
    result = runner.navigate_to(instruction)
    if result['status'] != 'navigating':
        print(f"Cannot navigate: {result}")
        return

    window = "Known-Env Navigation"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    for step in range(max_steps):
        rgb, depth = env.step("move_forward")  # Dummy to get obs
        # Undo the dummy step - get actual action
        rgb_obs = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        map_vis = runner.render_map_view(400)

        # Combine RGB and map
        rgb_resized = cv2.resize(rgb_obs, (400, 300))
        display = np.zeros((300 + map_vis.shape[0], 400, 3), dtype=np.uint8)
        display[:300, :] = rgb_resized
        display[300:300+map_vis.shape[0], :map_vis.shape[1]] = map_vis

        cv2.imshow(window, display)
        if cv2.waitKey(30) == ord('q'):
            break

        action = runner.step_navigation(rgb, depth)
        if action == 'stop':
            print(f"Navigation complete: {runner.status}")
            cv2.waitKey(0)
            break

        rgb, depth = env.step(action)

    cv2.destroyAllWindows()


def _run_interactive(runner, env, max_steps):
    """Interactive mode — type instructions in terminal."""
    window = "Known-Env Navigation"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 50)
    print("  HABITAT KNOWN-ENV NAVIGATION")
    print("=" * 50)
    print("  Type instruction in terminal, watch agent navigate")
    print("  Press 'q' in window to quit")
    print("  Press 'n' for new instruction")
    print("=" * 50)

    rgb, depth = env.step("move_forward")
    runner.set_origin_from_agent()

    navigating = False

    while True:
        # Render
        rgb_obs = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        map_vis = runner.render_map_view(400)
        rgb_resized = cv2.resize(rgb_obs, (400, 300))

        h_map = map_vis.shape[0]
        display = np.zeros((max(300, h_map), 800, 3), dtype=np.uint8)
        display[:300, :400] = rgb_resized
        display[:h_map, 400:400+map_vis.shape[1]] = map_vis

        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n') or (not navigating):
            if not navigating:
                print("\nType instruction: ", end='', flush=True)
                try:
                    instruction = input().strip()
                    if instruction.lower() == 'quit':
                        break
                    result = runner.navigate_to(instruction)
                    print(f"  → {result.get('message', result['status'])}")
                    navigating = result['status'] == 'navigating'
                except EOFError:
                    break

        if navigating:
            action = runner.step_navigation(rgb, depth)
            if action == 'stop':
                print(f"  Reached! Status: {runner.status}")
                navigating = False
            else:
                rgb, depth = env.step(action)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()