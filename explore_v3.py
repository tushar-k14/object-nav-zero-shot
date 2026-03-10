#!/usr/bin/env python3
"""
Interactive Explorer V3
========================

Explore scenes in Habitat-sim with YOLO detection and semantic mapping.

Improvements over V2:
  - Any YOLO model via --yolo-path (yolov8, yolo11, yolo26, etc.)
  - Simplified visualization (2-panel: RGB+detections | Map)
  - Auto-save every N steps
  - Natural language instruction support (press N)
  - Compatible with scene_graph from perception/ module

Controls:
  WASD / Arrow keys  - Manual movement
  Click on map       - Navigate to position
  N                  - Type natural language instruction
  R                  - 360° scan
  S                  - Save maps + scene graph
  Space              - Toggle auto-navigation
  Q / Esc            - Quit (auto-saves)

Usage:
    python explore_v3.py --scene path/to/scene.glb --yolo-path yolo26x-seg.pt
    python explore_v3.py --scene path/to/scene.glb --yolo-path yolov8n-seg.pt --output-dir my_map
"""

import argparse
import numpy as np
import cv2
import sys
import time
import json
import heapq
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def setup_habitat(scene_path):
    import habitat_sim
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [480, 640]
    rgb_spec.position = [0.0, 1.25, 0.0]
    rgb_spec.hfov = 90

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [480, 640]
    depth_spec.position = [0.0, 1.25, 0.0]
    depth_spec.hfov = 90

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)),
    }

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


class Explorer:
    def __init__(self, sim, mapper, scene_graph, output_dir,
                 auto_save_interval=200):
        self.sim = sim
        self.mapper = mapper
        self.sg = scene_graph
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.step_count = 0
        self.auto_mode = False
        self.nav_path = []
        self.nav_index = 0
        self.auto_save_interval = auto_save_interval
        self._last_save_step = 0

        # Frontier explorer (proper scoring + blacklisting)
        try:
            from core.frontier import FrontierExplorer
            self.frontier_explorer = FrontierExplorer(
                min_size=5, distance_weight=10.0,
                semantic_strength=1.5, semantic_radius=60,
            )
            print("  Using FrontierExplorer from core/frontier.py")
        except ImportError:
            self.frontier_explorer = None
            print("  FrontierExplorer not available — using simple frontier")

        # Stuck detection
        self._prev_positions = []
        self._stuck_counter = 0
        self._frontier_blacklist = set()
        self._current_frontier_id = None

        # A* planner
        try:
            from core.planning import astar
            self._astar = astar
            print("  Using A* from core/planning.py")
        except ImportError:
            self._astar = None

    def get_state(self):
        state = self.sim.get_agent(0).get_state()
        pos = np.array(state.position)
        rot = np.array([state.rotation.x, state.rotation.y,
                        state.rotation.z, state.rotation.w])
        return pos, rot

    def get_yaw(self, q):
        return np.arctan2(2 * (q[3] * q[1] + q[2] * q[0]),
                          1 - 2 * (q[0]**2 + q[1]**2))

    def step(self, action):
        if action:
            self.sim.step(action)
            self.step_count += 1
        obs = self.sim.get_sensor_observations()
        rgb = obs['color_sensor'][:, :, :3]
        depth = obs['depth_sensor']
        pos, rot = self.get_state()
        stats = self.mapper.update(rgb, depth, pos, rot)

        # Update scene graph from mapper's tracked objects
        if self.sg is not None:
            try:
                self.sg.update_from_tracker(self.mapper.tracked_objects)
            except AttributeError:
                # New-style scene graph uses update_objects
                try:
                    occ = self.mapper.occupancy_map
                    self.sg.update_objects(self.mapper.get_confirmed_objects(), occ)
                except Exception:
                    pass

        # Auto-save
        if (self.auto_save_interval > 0 and
                self.step_count - self._last_save_step >= self.auto_save_interval):
            self.save_state(quiet=True)
            self._last_save_step = self.step_count

        return rgb, depth, pos, rot, stats

    def plan_path(self, gx, gy):
        """Plan path using core A* or fallback inline A*."""
        pos, _ = self.get_state()
        occ = self.mapper.occupancy_map
        start = occ._world_to_map(pos[0], pos[2])

        if self._astar:
            return self._astar(start, (gx, gy), occ.get_obstacle_map(), inflation=3)

        # Fallback inline A*
        obs_map = occ.get_obstacle_map()
        inflated = cv2.dilate((obs_map > 0.5).astype(np.uint8),
                              np.ones((5, 5), np.uint8))

        def h(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        open_set = [(h(start, (gx, gy)), 0, start)]
        came_from = {}
        g = {start: 0}
        cnt = 0

        for _ in range(8000):
            if not open_set:
                break
            _, _, cur = heapq.heappop(open_set)
            if h(cur, (gx, gy)) < 5:
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                return path[::-1][::3]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = cur[0] + dx, cur[1] + dy
                hh, ww = inflated.shape
                if 0 <= nx < ww and 0 <= ny < hh and inflated[ny, nx] == 0:
                    tg = g[cur] + (1.414 if abs(dx) + abs(dy) == 2 else 1)
                    if (nx, ny) not in g or tg < g[(nx, ny)]:
                        came_from[(nx, ny)] = cur
                        g[(nx, ny)] = tg
                        cnt += 1
                        heapq.heappush(open_set,
                                       (tg + h((nx, ny), (gx, gy)), cnt, (nx, ny)))
        return []

    def get_auto_action(self, pos, rot):
        """Get next action for auto navigation or exploration."""
        occ = self.mapper.occupancy_map
        am = occ._world_to_map(pos[0], pos[2])

        # Stuck detection (same position for 15+ steps)
        self._prev_positions.append(am)
        if len(self._prev_positions) > 20:
            self._prev_positions.pop(0)

        if len(self._prev_positions) >= 15:
            recent = self._prev_positions[-15:]
            spread = max(
                max(p[0] for p in recent) - min(p[0] for p in recent),
                max(p[1] for p in recent) - min(p[1] for p in recent),
            )
            if spread < 3:
                self._stuck_counter += 1
                if self._stuck_counter == 1:
                    print("  STUCK — recovering")
                    # Blacklist current frontier
                    if self._current_frontier_id is not None:
                        self._frontier_blacklist.add(self._current_frontier_id)
                    self.nav_path = []
                    self.nav_index = 0
                # Recovery: turn for a bit then try new frontier
                if self._stuck_counter < 8:
                    return 'turn_left'
                elif self._stuck_counter < 12:
                    return 'move_forward'
                else:
                    self._stuck_counter = 0
                    return None  # Will trigger new frontier selection
            else:
                self._stuck_counter = 0

        # If we have a nav path, follow it
        if self.nav_path and self.nav_index < len(self.nav_path):
            tgt = self.nav_path[self.nav_index]

            if np.sqrt((tgt[0] - am[0])**2 + (tgt[1] - am[1])**2) < 3:
                self.nav_index += 1
                if self.nav_index >= len(self.nav_path):
                    # Reached frontier — scan surroundings
                    self.nav_path = []
                    self._current_frontier_id = None
                    return 'turn_left'  # Quick look around
                tgt = self.nav_path[self.nav_index]

            yaw = self.get_yaw(rot)
            ta = np.arctan2(-(tgt[1] - am[1]), tgt[0] - am[0])
            diff = ta - yaw
            while diff > np.pi: diff -= 2 * np.pi
            while diff < -np.pi: diff += 2 * np.pi

            if abs(diff) < np.radians(15):
                return 'move_forward'
            return 'turn_left' if diff > 0 else 'turn_right'

        # No nav path — select new frontier
        return self._select_frontier_and_plan(pos, rot)

    def _select_frontier_and_plan(self, pos, rot):
        """Select best frontier using FrontierExplorer and plan path to it."""
        occ = self.mapper.occupancy_map
        explored = occ.get_explored_map()
        obstacle = occ.get_obstacle_map()
        am = occ._world_to_map(pos[0], pos[2])

        # Get detected objects for semantic scoring
        det_objects = []
        for obj in self.mapper.get_confirmed_objects().values():
            mx, my = occ._world_to_map(obj.mean_position[0], obj.mean_position[2])
            det_objects.append((obj.class_name.lower(), (mx, my)))

        if self.frontier_explorer:
            # Use proper FrontierExplorer
            frontier = self.frontier_explorer.select(
                agent_pos=am,
                detected_objects=det_objects,
                blacklist=self._frontier_blacklist,
                explored_map=explored,
                obstacle_map=obstacle,
            )

            if frontier is None:
                exp_pct = float(explored.sum()) / explored.size * 100
                print(f"  No more frontiers — exploration complete! ({exp_pct:.1f}% explored)")
                self.auto_mode = False
                return None

            target = frontier.centroid
            self._current_frontier_id = frontier.id

            if self.step_count % 50 == 0:
                print(f"  Frontier: id={frontier.id} size={frontier.size} "
                      f"score={frontier.score:.3f} at ({target[0]},{target[1]})")
        else:
            # Fallback: simple largest frontier
            target = self._find_simple_frontier(explored, obstacle)
            if target is None:
                self.auto_mode = False
                return None

        # Plan path
        self.nav_path = self.plan_path(target[0], target[1])
        self.nav_index = 0

        if not self.nav_path:
            # Can't reach — blacklist and try again
            if self._current_frontier_id is not None:
                self._frontier_blacklist.add(self._current_frontier_id)
            return 'turn_left'

        return self.get_auto_action(pos, rot)

    def _find_simple_frontier(self, explored, obstacle):
        """Fallback: find largest frontier cluster (no scoring)."""
        explored_bin = (explored > 0.5).astype(np.uint8)
        free_bin = (obstacle < 0.5).astype(np.uint8)
        unexplored = (explored < 0.5).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        near_unexplored = cv2.dilate(unexplored, kernel) > 0
        frontier_mask = (explored_bin & free_bin & near_unexplored.astype(np.uint8))

        n_labels, labels = cv2.connectedComponents(frontier_mask)
        if n_labels <= 1:
            return None

        best_target = None
        best_size = 0
        for lbl in range(1, n_labels):
            ys, xs = np.where(labels == lbl)
            if len(xs) > best_size:
                best_size = len(xs)
                best_target = (int(xs.mean()), int(ys.mean()))
        return best_target

    # ----------------------------------------------------------
    # Visualization (simplified 2-panel)
    # ----------------------------------------------------------

    def create_vis(self, rgb, depth, stats):
        """Create simplified 2-panel visualization: [RGB+YOLO | Map]"""
        PH, PW = 400, 500

        # Left panel: RGB with YOLO detections
        det_vis = self.mapper.visualize_detections(rgb)
        left = cv2.resize(cv2.cvtColor(det_vis, cv2.COLOR_RGB2BGR), (PW, PH))

        # Right panel: Semantic map
        map_vis = self.mapper.visualize(show_legend=True)
        mh, mw = map_vis.shape[:2]
        scale = min(PW / mw, PH / mh)
        map_resized = cv2.resize(map_vis, (int(mw * scale), int(mh * scale)))

        # Center map in panel
        right = np.zeros((PH, PW, 3), dtype=np.uint8)
        right[:] = [30, 30, 30]
        y_off = (PH - map_resized.shape[0]) // 2
        x_off = (PW - map_resized.shape[1]) // 2
        right[y_off:y_off + map_resized.shape[0],
              x_off:x_off + map_resized.shape[1]] = map_resized

        # Draw nav path on map
        if self.nav_path:
            for i in range(len(self.nav_path) - 1):
                p1 = self.nav_path[i]
                p2 = self.nav_path[i + 1]
                sp1 = (int(p1[0] * scale) + x_off, int(p1[1] * scale) + y_off)
                sp2 = (int(p2[0] * scale) + x_off, int(p2[1] * scale) + y_off)
                color = (0, 255, 0) if i >= self.nav_index else (0, 100, 0)
                cv2.line(right, sp1, sp2, color, 1)

        vis = np.hstack([left, right])

        # Status bar
        floor = stats.get('current_floor', 0)
        n_obj = stats.get('confirmed_objects', 0)
        expl = stats.get('explored_percent', 0)
        mode = "AUTO" if self.auto_mode else "MANUAL"

        bar_h = 28
        bar = np.zeros((bar_h, vis.shape[1], 3), dtype=np.uint8)
        bar[:] = [25, 25, 35]
        info = (f"Step: {self.step_count}  |  Floor: {floor}  |  "
                f"Objects: {n_obj}  |  Explored: {expl:.1f}%  |  {mode}")
        cv2.putText(bar, info, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 200), 1)
        controls = "WASD:Move | Click:Navigate | N:Instruction | R:Scan | S:Save | Q:Quit"
        cv2.putText(bar, controls, (vis.shape[1] - 520, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1)

        vis = np.vstack([vis, bar])

        return vis, PH, PW, scale, x_off, y_off

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    def save_state(self, quiet=False):
        if not quiet:
            print("\nSaving...")
        self.mapper.save(str(self.output_dir))
        if self.sg is not None:
            try:
                self.sg.save(str(self.output_dir / 'scene_graph.json'))
            except Exception:
                pass
        if not quiet:
            print(f"  Saved to {self.output_dir}")
        else:
            print(f"  [auto-save at step {self.step_count}]")

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------

    def run(self):
        print("\n" + "=" * 60)
        print("  Explorer V3 — YOLO + Semantic Mapping")
        print("=" * 60)
        print("  WASD=Move | Click map=Navigate | N=Instruction")
        print("  R=360° Scan | S=Save | Space=Toggle auto | Q=Quit")
        print("=" * 60)

        # Initial 360° scan
        print("\n  Initial scan...")
        for _ in range(36):
            self.step('turn_left')
        stats = self.mapper.get_statistics()
        print(f"  Found {stats.get('confirmed_objects', 0)} objects\n")

        window = 'Explorer V3'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1000, 430)

        vis_info = {'PH': 400, 'PW': 500, 'scale': 1.0, 'x_off': 0, 'y_off': 0}

        def mouse_cb(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            PH, PW = vis_info['PH'], vis_info['PW']
            scale, x_off, y_off = vis_info['scale'], vis_info['x_off'], vis_info['y_off']
            # Click on right panel (map)
            if x >= PW:
                mx = int((x - PW - x_off) / scale)
                my = int((y - y_off) / scale)
                print(f"  Navigate to map({mx}, {my})")
                self.nav_path = self.plan_path(mx, my)
                self.nav_index = 0
                self.auto_mode = bool(self.nav_path)
                if self.nav_path:
                    print(f"  Path: {len(self.nav_path)} waypoints")
                else:
                    print(f"  No path found!")

        cv2.setMouseCallback(window, mouse_cb)

        rgb, depth, pos, rot, stats = self.step(None)

        while True:
            vis, PH, PW, scale, x_off, y_off = self.create_vis(rgb, depth, stats)
            vis_info.update({'PH': PH, 'PW': PW, 'scale': scale,
                             'x_off': x_off, 'y_off': y_off})
            cv2.imshow(window, vis)

            key = cv2.waitKey(30) & 0xFF
            action = None

            if key == ord('q') or key == 27:
                break
            elif key == ord('w') or key == 82:
                action = 'move_forward'; self.auto_mode = False
            elif key == ord('a') or key == 81:
                action = 'turn_left'; self.auto_mode = False
            elif key == ord('d') or key == 83:
                action = 'turn_right'; self.auto_mode = False
            elif key == ord('s'):
                self.save_state()
            elif key == ord('r'):
                print("  Scanning...")
                for _ in range(36):
                    rgb, depth, pos, rot, stats = self.step('turn_left')
                    cv2.imshow(window, self.create_vis(rgb, depth, stats)[0])
                    cv2.waitKey(1)
                print(f"  Done: {stats.get('confirmed_objects', 0)} objects")
            elif key == ord(' '):
                self.auto_mode = not self.auto_mode
                print(f"  Auto: {'ON' if self.auto_mode else 'OFF'}")
            elif key == ord('n'):
                # Natural language instruction
                print("\n  Type instruction: ", end='', flush=True)
                try:
                    instruction = input().strip()
                    if instruction:
                        self._handle_instruction(instruction, pos)
                except EOFError:
                    pass

            if self.auto_mode:
                action = self.get_auto_action(pos, rot)

            if action:
                rgb, depth, pos, rot, stats = self.step(action)

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        self.save_state()

    def _handle_instruction(self, instruction, pos):
        """Handle natural language instruction during exploration."""
        try:
            from perception.instruction_parser import InstructionParser, goal_to_action
            parser = InstructionParser(mode='regex')
            goal = parser.parse(instruction)
            print(f"  Parsed: {goal.type}:{goal.target}")

            if goal.type == 'stop':
                self.auto_mode = False
                self.nav_path = []
                return

            if goal.type == 'object':
                # Find in mapped objects
                occ = self.mapper.occupancy_map
                for obj in self.mapper.get_confirmed_objects().values():
                    name = obj.class_name.lower()
                    if goal.target in name or name in goal.target:
                        mx, my = occ._world_to_map(
                            obj.mean_position[0], obj.mean_position[2])
                        print(f"  Found {name} at map({mx},{my})")
                        self.nav_path = self.plan_path(mx, my)
                        self.nav_index = 0
                        self.auto_mode = bool(self.nav_path)
                        return
                print(f"  Object '{goal.target}' not found in map")

            elif goal.type == 'room':
                # Check if we have room annotations
                print(f"  Room navigation requires map annotations (run map_editor_v2)")

        except ImportError:
            print("  Instruction parser not available")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Explorer V3')
    parser.add_argument('--scene', required=True, help='Habitat scene GLB file')
    parser.add_argument('--yolo-path', type=str, default=None,
                        help='Path to YOLO model (yolov8n-seg.pt, yolo26x-seg.pt, etc.)')
    parser.add_argument('--model-size', default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size (ignored if --yolo-path is set)')
    parser.add_argument('--confidence', type=float, default=0.4)
    parser.add_argument('--output-dir', default='./output_v3')
    parser.add_argument('--auto-save', type=int, default=200,
                        help='Auto-save every N steps (0 to disable)')
    parser.add_argument('--map-size', type=int, default=4800,
                        help='Map size in cm')
    args = parser.parse_args()

    print("Loading Habitat scene...")
    sim = setup_habitat(args.scene)

    # Place agent at random navigable point
    agent = sim.get_agent(0)
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)

    print(f"Agent start: {agent.get_state().position}")

    # Setup mapper with custom YOLO model
    from mapping.fixed_occupancy_map import MapConfig
    from mapping.enhanced_semantic_mapper_v2 import EnhancedSemanticMapperV2

    map_config = MapConfig()
    map_config.map_size_cm = args.map_size

    # Load YOLO model
    yolo_model = None
    if args.yolo_path:
        print(f"Loading YOLO from: {args.yolo_path}")
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_path)
        print(f"YOLO loaded: {args.yolo_path} ({len(yolo_model.names)} classes)")

    print("Creating mapper...")
    mapper = EnhancedSemanticMapperV2(
        map_config=map_config,
        model_size=args.model_size,
        confidence_threshold=args.confidence,
        yolo_model=yolo_model,
    )
    print("Mapper ready")

    # Scene graph — try both old and new versions
    scene_graph = None
    try:
        from navigation.scene_graph import SceneGraph
        scene_graph = SceneGraph()
        print("Using navigation.scene_graph")
    except ImportError:
        try:
            from perception.scene_graph import SceneGraph
            scene_graph = SceneGraph()
            print("Using perception.scene_graph")
        except ImportError:
            print("No scene graph module found — running without")

    explorer = Explorer(sim, mapper, scene_graph, args.output_dir,
                        auto_save_interval=args.auto_save)
    explorer.run()
    sim.close()


if __name__ == '__main__':
    main()