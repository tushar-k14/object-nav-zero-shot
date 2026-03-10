#!/usr/bin/env python3
"""
Navigation Test V2 - Full Visualization
========================================

Shows EVERYTHING:
1. What the planner sees (obstacle map, inflation, planned path)
2. Real-time RGBD while navigating
3. Agent's view + map view synchronized

Layout:
┌────────────────────────────────────────────────────┐
│  RGB View        │  YOLO Detections               │
├──────────────────┼─────────────────────────────────┤
│  Depth View      │  Planner View (obstacles+path) │
├──────────────────┴─────────────────────────────────┤
│              Semantic Map + Navigation             │
└────────────────────────────────────────────────────┘

Controls:
- 1-9: Navigate to object
- F: Explore frontier  
- Click on map: Navigate to position
- P: Show planner debug view
- WASD: Manual
- R: Scan | C: Cancel | S: Save | Q: Quit
"""

import argparse
import numpy as np
import cv2
import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))

from mapping.fixed_occupancy_map import MapConfig
from mapping.enhanced_semantic_mapper_v2 import EnhancedSemanticMapperV2
from navigation.scene_graph import SceneGraph
from navigation.nav_controller import NavigationController, NavState, GoalType


OBJECT_HOTKEYS = {
    ord('1'): 'toilet', ord('2'): 'bed', ord('3'): 'chair',
    ord('4'): 'couch', ord('5'): 'tv', ord('6'): 'refrigerator',
    ord('7'): 'sink', ord('8'): 'dining table', ord('9'): 'potted plant',
}


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
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)),
    }
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


class NavigationVisualizer:
    """
    Visualizes what the navigation planner sees and uses.
    """
    
    def __init__(self, nav_controller, semantic_mapper):
        self.nav = nav_controller
        self.mapper = semantic_mapper
    
    def get_planner_view(self) -> np.ndarray:
        """
        Visualize what the global planner sees:
        - Raw obstacle map
        - Inflated obstacles
        - Planned path
        - Start and goal
        """
        occ_map = self.mapper.occupancy_map
        map_size = occ_map.map_size
        
        # Get raw obstacle map
        obstacle = occ_map.get_obstacle_map()
        
        # Get inflated map (what A* actually uses)
        inflation = self.nav.path_inflation
        kernel = np.ones((inflation * 2 + 1,) * 2, np.uint8)
        inflated = cv2.dilate((obstacle > 0).astype(np.uint8), kernel)
        
        # Create RGB visualization
        vis = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        # Background (unexplored = dark gray)
        vis[:, :] = [40, 40, 40]
        
        # Explored = light gray
        explored = occ_map.get_explored_map()
        vis[explored > 0] = [100, 100, 100]
        
        # Raw obstacles = blue
        vis[obstacle > 0] = [255, 100, 100]  # Blue (BGR)
        
        # Inflated obstacles = lighter blue (semi-transparent effect)
        inflation_only = (inflated > 0) & (obstacle == 0)
        vis[inflation_only] = [255, 180, 150]  # Light blue
        
        # Draw planned path
        if self.nav.path and self.nav.path.valid:
            waypoints = self.nav.path.waypoints_map
            
            # Draw path line
            for i in range(len(waypoints) - 1):
                p1, p2 = waypoints[i], waypoints[i + 1]
                color = (0, 255, 0) if i >= self.nav.current_waypoint_idx else (0, 100, 0)
                cv2.line(vis, p1, p2, color, 2)
            
            # Draw waypoints
            for i, wp in enumerate(waypoints):
                if i < self.nav.current_waypoint_idx:
                    color = (0, 100, 0)  # Past = dark green
                elif i == self.nav.current_waypoint_idx:
                    color = (0, 255, 255)  # Current = yellow
                else:
                    color = (0, 255, 0)  # Future = green
                cv2.circle(vis, wp, 4, color, -1)
        
        # Draw goal
        if self.nav.goal and self.nav.goal.target_map:
            gx, gy = self.nav.goal.target_map
            cv2.circle(vis, (gx, gy), 12, (255, 0, 255), 2)  # Magenta
            cv2.circle(vis, (gx, gy), 6, (255, 0, 255), -1)
            
            # Goal label
            label = self.nav.goal.target_name or self.nav.goal.goal_type.value
            cv2.putText(vis, label, (gx + 15, gy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw agent position and orientation
        if occ_map.trajectory:
            agent_pos = occ_map.trajectory[-1]
            ax, ay = occ_map._world_to_map(agent_pos[0], agent_pos[2])
            
            # Agent marker
            cv2.circle(vis, (ax, ay), 8, (0, 165, 255), -1)  # Orange
            cv2.circle(vis, (ax, ay), 8, (0, 0, 0), 2)
        
        # Add legend
        vis = self._add_planner_legend(vis)
        
        return vis
    
    def _add_planner_legend(self, vis: np.ndarray) -> np.ndarray:
        """Add legend explaining planner visualization."""
        legend_h = 120
        legend_w = 150
        legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
        legend[:, :] = [30, 30, 30]
        
        items = [
            ("PLANNER VIEW", None),
            ("Obstacle (raw)", (255, 100, 100)),
            ("Inflation zone", (255, 180, 150)),
            ("Planned path", (0, 255, 0)),
            ("Current waypoint", (0, 255, 255)),
            ("Goal", (255, 0, 255)),
            ("Agent", (0, 165, 255)),
        ]
        
        y = 12
        for text, color in items:
            if color is None:
                cv2.putText(legend, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            else:
                cv2.rectangle(legend, (5, y - 8), (15, y + 2), color, -1)
                cv2.putText(legend, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            y += 15
        
        # Position legend in corner
        if legend_h <= vis.shape[0] and legend_w <= vis.shape[1]:
            vis[5:5+legend_h, vis.shape[1]-legend_w-5:vis.shape[1]-5] = legend
        
        return vis
    
    def get_navigation_status_overlay(self, vis: np.ndarray) -> np.ndarray:
        """Add navigation status info."""
        h, w = vis.shape[:2]
        
        status = self.nav.get_status()
        goal = self.nav.goal
        
        # Status text
        lines = [
            f"State: {status['state']}",
            f"Goal: {status['goal_target'] or 'none'} ({status['goal_type'] or '-'})",
            f"Waypoint: {status['waypoint']}",
            f"Collisions avoided: {status['collisions_avoided']}",
        ]
        
        if self.nav.path and self.nav.path.valid:
            lines.append(f"Path distance: {self.nav.path.distance:.1f}m")
        
        y = h - 10 - len(lines) * 18
        for line in lines:
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 18
        
        return vis


class NavigationTesterV2:
    def __init__(self, sim, mapper, scene_graph, nav, output_dir):
        self.sim = sim
        self.mapper = mapper
        self.scene_graph = scene_graph
        self.nav = nav
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer = NavigationVisualizer(nav, mapper)
        
        self.step_count = 0
        self.start_time = time.time()
        self.auto_nav = False
        self.show_planner_view = True  # Show planner debug by default
        
        # For smooth visualization
        self.last_rgb = None
        self.last_depth = None
    
    def get_agent_state(self):
        state = self.sim.get_agent(0).get_state()
        return np.array(state.position), np.array([state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w])
    
    def step(self, action):
        if action:
            self.sim.step(action)
            self.step_count += 1
        
        obs = self.sim.get_sensor_observations()
        rgb, depth = obs['color_sensor'][:,:,:3], obs['depth_sensor']
        self.last_rgb = rgb
        self.last_depth = depth
        
        pos, rot = self.get_agent_state()
        stats = self.mapper.update(rgb, depth, pos, rot)
        self.scene_graph.update_from_tracker(self.mapper.tracked_objects)
        
        return rgb, depth, pos, rot, stats
    
    def create_full_visualization(self, rgb, depth, stats):
        """
        Create comprehensive visualization showing everything.
        """
        panel_h, panel_w = 240, 320
        
        # === Top Row: RGB + Detections ===
        rgb_bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (panel_w, panel_h))
        det_vis = cv2.resize(cv2.cvtColor(self.mapper.visualize_detections(rgb), cv2.COLOR_RGB2BGR), (panel_w, panel_h))
        
        # === Middle Row: Depth + Planner View ===
        depth_norm = np.clip(depth / 5.0, 0, 1)
        depth_vis = cv2.resize(cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), (panel_w, panel_h))
        
        if self.show_planner_view:
            planner_vis = self.visualizer.get_planner_view()
            planner_vis = cv2.resize(planner_vis, (panel_w, panel_h))
        else:
            # Show semantic map instead
            sem_vis = self.mapper.visualize(show_legend=False)
            planner_vis = cv2.resize(sem_vis, (panel_w, panel_h))
        
        # === Bottom Row: Full Semantic Map with Navigation ===
        map_vis = self.mapper.visualize(show_legend=True)
        map_vis = self.nav.visualize_on_map(map_vis)
        
        # Scale map to fit width
        map_w = panel_w * 2
        map_scale = map_w / map_vis.shape[1]
        map_h = int(map_vis.shape[0] * map_scale)
        map_vis = cv2.resize(map_vis, (map_w, map_h))
        
        # Combine panels
        top_row = np.hstack([rgb_bgr, det_vis])
        mid_row = np.hstack([depth_vis, planner_vis])
        
        # Ensure rows have same width
        total_w = max(top_row.shape[1], mid_row.shape[1], map_vis.shape[1])
        
        def pad_to_width(img, target_w):
            if img.shape[1] < target_w:
                pad = np.zeros((img.shape[0], target_w - img.shape[1], 3), np.uint8)
                return np.hstack([img, pad])
            return img
        
        top_row = pad_to_width(top_row, total_w)
        mid_row = pad_to_width(mid_row, total_w)
        map_vis = pad_to_width(map_vis, total_w)
        
        vis = np.vstack([top_row, mid_row, map_vis])
        
        # === Add Info Overlay ===
        elapsed = time.time() - self.start_time
        nav_status = self.nav.get_status()
        
        # Top info bar
        info1 = f"Step:{self.step_count} | Obj:{stats.get('confirmed_objects',0)} | Floor:{stats.get('current_floor',0)} | Time:{elapsed:.0f}s"
        cv2.putText(vis, info1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Navigation info
        mode = "AUTO-NAV" if self.auto_nav else "MANUAL"
        state_color = {
            'navigating': (0, 255, 0),
            'reached': (0, 255, 0),
            'avoiding': (0, 255, 255),
            'recovering': (0, 0, 255),
            'failed': (0, 0, 255),
        }.get(nav_status['state'], (200, 200, 200))
        
        info2 = f"[{mode}] Nav:{nav_status['state']} | Goal:{nav_status['goal_target'] or 'none'} | WP:{nav_status['waypoint']}"
        cv2.putText(vis, info2, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1)
        
        # Panel labels
        cv2.putText(vis, "RGB", (5, panel_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(vis, "YOLO Detections", (panel_w + 5, panel_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(vis, "Depth", (5, 2*panel_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        planner_label = "Planner View (P to toggle)" if self.show_planner_view else "Semantic Map (P to toggle)"
        cv2.putText(vis, planner_label, (panel_w + 5, 2*panel_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Controls hint
        cv2.putText(vis, "1-9:Object F:Explore Click:Position | WASD:Manual C:Cancel R:Scan P:Planner S:Save Q:Quit",
                   (10, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
        
        return vis, panel_h, panel_w
    
    def save_state(self):
        print("\nSaving...")
        self.mapper.save(str(self.output_dir))
        self.scene_graph.save(str(self.output_dir / 'scene_graph.json'))
        
        # Save planner visualization
        planner_vis = self.visualizer.get_planner_view()
        cv2.imwrite(str(self.output_dir / 'planner_view.png'), planner_vis)
        
        # Save nav stats
        with open(self.output_dir / 'nav_stats.json', 'w') as f:
            json.dump(self.nav.get_status(), f, indent=2)
        
        print(f"Saved to {self.output_dir}")
    
    def run(self):
        print("\n" + "="*70)
        print("Navigation Test V2 - Full Visualization")
        print("="*70)
        print("Goal Setting:")
        print("  1-9: Navigate to object (1=toilet, 2=bed, 3=chair, ...)")
        print("  F: Navigate to frontier (explore)")
        print("  Click on bottom map: Navigate to position")
        print("Controls:")
        print("  WASD/Arrows: Manual movement (cancels auto-nav)")
        print("  P: Toggle planner view / semantic map")
        print("  C: Cancel navigation | R: 360° scan")
        print("  S: Save state | Q: Quit")
        print("="*70)
        
        # Initial scan
        print("\nPerforming initial 360° scan...")
        for _ in range(36):
            self.step('turn_left')
        
        stats = self.mapper.get_statistics()
        print(f"\nDiscovered objects: {list(stats['class_counts'].keys())}")
        print(f"Total confirmed: {stats['confirmed_objects']}")
        
        # Setup window
        window = 'Navigation Test V2'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 700, 800)
        
        panel_info = {'h': 240, 'w': 320}
        
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                h, w = panel_info['h'], panel_info['w']
                map_top = 2 * h  # Map starts after 2 rows of panels
                
                if y >= map_top:
                    # Click on map area
                    map_y = y - map_top
                    map_size = self.mapper.occupancy_map.map_size
                    
                    # Get map dimensions (accounting for legend)
                    map_vis = self.mapper.visualize(show_legend=True)
                    vis_w, vis_h = 2 * w, int(map_vis.shape[0] * (2 * w) / map_vis.shape[1])
                    
                    if x < vis_w and map_y < vis_h:
                        # Scale click to map coordinates
                        scale_x = map_vis.shape[1] / vis_w
                        scale_y = map_vis.shape[0] / vis_h
                        
                        mx = int(x * scale_x)
                        my = int(map_y * scale_y)
                        
                        # Check if click is within actual map (not legend)
                        if mx < map_size and my < map_size:
                            occ = self.mapper.occupancy_map
                            wx = (mx - occ.map_size//2) * occ.resolution / 100.0 + occ.origin_x
                            wz = (my - occ.map_size//2) * occ.resolution / 100.0 + occ.origin_z
                            
                            print(f"\nNavigate to position ({wx:.1f}, {wz:.1f})")
                            if self.nav.set_goal_position(np.array([wx, 0, wz])):
                                self.auto_nav = True
        
        cv2.setMouseCallback(window, mouse_cb)
        
        # Initial observation
        rgb, depth, pos, rot, stats = self.step(None)
        
        # Main loop
        while True:
            vis, h, w = self.create_full_visualization(rgb, depth, stats)
            panel_info['h'], panel_info['w'] = h, w
            cv2.imshow(window, vis)
            
            key = cv2.waitKey(30) & 0xFF
            action = None
            
            # Quit
            if key == ord('q') or key == 27:
                break
            
            # Manual movement (cancels auto-nav)
            elif key == ord('w') or key == 82:
                action = 'move_forward'
                self.auto_nav = False
                self.nav.cancel_goal()
            elif key == ord('a') or key == 81:
                action = 'turn_left'
                self.auto_nav = False
                self.nav.cancel_goal()
            elif key == ord('d') or key == 83:
                action = 'turn_right'
                self.auto_nav = False
                self.nav.cancel_goal()
            elif key == ord('s') and not (key == ord('s') and self.auto_nav):
                # 's' for manual back movement only when not navigating
                action = 'move_forward'  # No back action in habitat
                self.auto_nav = False
                self.nav.cancel_goal()
            
            # Object navigation hotkeys
            elif key in OBJECT_HOTKEYS:
                obj_class = OBJECT_HOTKEYS[key]
                print(f"\nNavigate to: {obj_class}")
                if self.nav.set_goal_object(obj_class):
                    self.auto_nav = True
                    print(f"  Path planned: {len(self.nav.path.waypoints_map)} waypoints, {self.nav.path.distance:.1f}m")
                else:
                    print(f"  '{obj_class}' not found in scene!")
            
            # Frontier exploration
            elif key == ord('f'):
                print("\nNavigate to frontier (explore)")
                if self.nav.set_goal_frontier():
                    self.auto_nav = True
                    print(f"  Path planned: {len(self.nav.path.waypoints_map)} waypoints")
                else:
                    print("  No unexplored areas found!")
            
            # Cancel navigation
            elif key == ord('c'):
                print("Navigation cancelled")
                self.nav.cancel_goal()
                self.auto_nav = False
            
            # Toggle planner view
            elif key == ord('p'):
                self.show_planner_view = not self.show_planner_view
                print(f"Planner view: {'ON' if self.show_planner_view else 'OFF'}")
            
            # 360° scan
            elif key == ord('r'):
                print("360° scan...")
                self.auto_nav = False
                self.nav.cancel_goal()
                for _ in range(36):
                    rgb, depth, pos, rot, stats = self.step('turn_left')
                    vis, _, _ = self.create_full_visualization(rgb, depth, stats)
                    cv2.imshow(window, vis)
                    cv2.waitKey(1)
                print(f"Scan complete. Objects: {self.mapper.get_statistics()['confirmed_objects']}")
            
            # Save
            elif key == ord('s'):
                self.save_state()
            
            # Auto navigation
            if self.auto_nav:
                action = self.nav.get_action(depth, pos, rot)
                
                if action is None:
                    if self.nav.state == NavState.REACHED:
                        print("\n*** GOAL REACHED! ***")
                    elif self.nav.state == NavState.FAILED:
                        print("\n*** NAVIGATION FAILED ***")
                    self.auto_nav = False
            
            # Execute action
            if action:
                rgb, depth, pos, rot, stats = self.step(action)
            
            # Check window
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
        self.save_state()
        
        # Final summary
        print("\n" + "="*70)
        print("Session Summary")
        print("="*70)
        nav_stats = self.nav.get_status()
        print(f"Total steps: {self.step_count}")
        print(f"Objects found: {self.mapper.get_statistics()['confirmed_objects']}")
        print(f"Navigation actions: {nav_stats['total_actions']}")
        print(f"Path replans: {nav_stats['replans']}")
        print(f"Collisions avoided: {nav_stats['collisions_avoided']}")


def main():
    parser = argparse.ArgumentParser(description='Navigation Test V2')
    parser.add_argument('--scene', required=True, help='Scene GLB file')
    parser.add_argument('--output-dir', default='./output_nav_v2', help='Output directory')
    parser.add_argument('--model-size', default='n', choices=['n','s','m','l','x'], help='YOLO model size')
    parser.add_argument('--confidence', type=float, default=0.4, help='Detection confidence')
    args = parser.parse_args()
    
    print("Loading scene...")
    sim = setup_habitat(args.scene)
    
    # Initialize agent
    agent = sim.get_agent(0)
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
        print(f"Agent start position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    
    # Setup mapping
    map_config = MapConfig()
    map_config.map_size_cm = 4800
    
    mapper = EnhancedSemanticMapperV2(
        map_config=map_config,
        model_size=args.model_size,
        confidence_threshold=args.confidence
    )
    
    scene_graph = SceneGraph()
    nav = NavigationController(mapper, scene_graph)
    
    # Run test
    tester = NavigationTesterV2(sim, mapper, scene_graph, nav, args.output_dir)
    tester.run()
    
    sim.close()


if __name__ == '__main__':
    main()
