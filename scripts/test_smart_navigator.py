#!/usr/bin/env python3
"""
Smart Navigator Test
====================

Interactive test for the integrated navigation system.

Usage:
    python test_smart_navigator.py --scene /path/to/scene.glb --map-dir ./maps/scene

Controls:
    W/↑ = Forward      A/← = Turn left    D/→ = Turn right
    G = Enter goal (VLM command)
    N = Navigate to nearest object type
    C = Correct pose to GT
    R = Reset
    L = List available objects
    Q = Quit
"""

import argparse
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from navigation.smart_navigator import SmartNavigator, NavState


def setup_habitat(scene_path: str):
    """Setup Habitat simulator."""
    import habitat_sim
    
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [480, 640]
    rgb_spec.position = [0.0, 1.25, 0.0]
    rgb_spec.hfov = 90
    
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
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
    
    return habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))


def get_gt_pose(sim):
    """Get ground truth pose."""
    state = sim.get_agent(0).get_state()
    pos = np.array(state.position)
    
    x, y, z, w = state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return pos, yaw


def main():
    parser = argparse.ArgumentParser(description='Smart Navigator Test')
    parser.add_argument('--scene', required=True, help='Path to scene')
    parser.add_argument('--map-dir', help='Path to pre-built maps')
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO')
    parser.add_argument('--no-vlm', action='store_true', help='Disable VLM')
    args = parser.parse_args()
    
    print("="*60)
    print("Smart Navigator Test")
    print("="*60)
    print("Controls:")
    print("  W/↑ = Forward    A/← = Left    D/→ = Right")
    print("  G = VLM command  N = Navigate  L = List objects")
    print("  C = Correct pose R = Reset     Q = Quit")
    print("  SPACE = Auto-navigate (follow path)")
    print("="*60)
    
    # Setup
    sim = setup_habitat(args.scene)
    
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
    
    gt_pos, gt_yaw = get_gt_pose(sim)
    
    # Navigator
    nav = SmartNavigator(
        map_dir=args.map_dir,
        use_yolo=not args.no_yolo,
        use_vlm=not args.no_vlm,
        use_ollama=True
    )
    nav.initialize(gt_pos, gt_yaw)
    
    print(f"\nYOLO: {'ON' if nav.detector and nav.detector.enabled else 'OFF'}")
    print(f"VLM: {'ON' if nav.vlm and nav.vlm.available else 'OFF'}")
    print("="*60)
    
    # Window
    window = 'Smart Navigator'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 900, 600)
    
    auto_navigate = False
    last_action = None
    
    while True:
        # Get observations
        obs = sim.get_sensor_observations()
        rgb = obs['rgb'][:, :, :3]
        depth = obs['depth']
        
        gt_pos, gt_yaw = get_gt_pose(sim)
        
        # Update navigator
        state = nav.update(last_action, rgb, depth, (gt_pos, gt_yaw))
        last_action = None
        
        # Auto-navigate
        action = None
        if auto_navigate and nav.state == NavState.NAVIGATING:
            action = nav.get_action(depth)
            if action is None:
                auto_navigate = False
                print("Navigation complete or failed")
        
        # Visualization
        vis = create_visualization(nav, rgb, depth, state, gt_pos, gt_yaw)
        
        cv2.imshow(window, vis)
        
        wait_time = 50 if auto_navigate else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        # Handle input
        if key == ord('q') or key == 27:
            break
        
        elif key == ord('w') or key == 82:  # Forward
            action = 'move_forward'
            auto_navigate = False
            
        elif key == ord('a') or key == 81:  # Left
            action = 'turn_left'
            auto_navigate = False
            
        elif key == ord('d') or key == 83:  # Right
            action = 'turn_right'
            auto_navigate = False
            
        elif key == ord(' '):  # Space - toggle auto
            auto_navigate = not auto_navigate
            print(f"Auto-navigate: {'ON' if auto_navigate else 'OFF'}")
            
        elif key == ord('g'):  # VLM command
            auto_navigate = False
            print("\nEnter command (e.g., 'go to bathroom', 'find a bed'):")
            command = input("> ").strip()
            if command:
                if nav.navigate_with_command(command, rgb):
                    auto_navigate = True
                    print("Navigation started. Press SPACE to pause.")
            
        elif key == ord('n'):  # Navigate to object
            auto_navigate = False
            objects = nav.get_available_objects()
            if objects:
                print(f"\nAvailable: {', '.join(objects)}")
                print("Enter object name:")
                target = input("> ").strip()
                if target:
                    if nav.navigate_to(target):
                        auto_navigate = True
            else:
                print("No objects detected yet. Explore more.")
            
        elif key == ord('l'):  # List objects
            objects = nav.get_available_objects()
            print(f"\nAvailable objects: {', '.join(objects) if objects else 'None'}")
            
        elif key == ord('c'):  # Correct pose
            nav.correct_pose(gt_pos, gt_yaw)
            
        elif key == ord('r'):  # Reset
            nav.initialize(gt_pos, gt_yaw)
            auto_navigate = False
        
        # Execute action
        if action:
            sim.step(action)
            last_action = action
        
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()
    sim.close()
    
    # Stats
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)
    stats = nav.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def create_visualization(nav, rgb, depth, state, gt_pos, gt_yaw):
    """Create visualization."""
    # RGB with detections
    if nav.detector and nav.detector.enabled:
        rgb_vis = nav.detector.visualize(rgb, depth)
    else:
        rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rgb_vis = cv2.resize(rgb_vis, (400, 300))
    
    # Depth
    depth_vis = cv2.applyColorMap(
        (np.clip(depth / 5.0, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS
    )
    depth_vis = cv2.resize(depth_vis, (400, 300))
    
    top = np.hstack([rgb_vis, depth_vis])
    
    # Map
    map_vis = nav.visualize(rgb, depth)
    if map_vis.shape[0] < 300:
        pad = np.zeros((300 - map_vis.shape[0], map_vis.shape[1], 3), dtype=np.uint8)
        map_vis = np.vstack([map_vis, pad])
    map_vis = map_vis[:300, :400]
    
    # Pad width if needed
    if map_vis.shape[1] < 400:
        pad = np.zeros((300, 400 - map_vis.shape[1], 3), dtype=np.uint8)
        map_vis = np.hstack([map_vis, pad])
    
    # Info panel
    info = np.zeros((300, 400, 3), dtype=np.uint8)
    info[:] = [30, 30, 30]
    
    stats = nav.get_stats()
    
    # State color
    state_colors = {
        'idle': (150, 150, 150),
        'navigating': (0, 255, 0),
        'avoiding': (0, 255, 255),
        'reached': (0, 255, 0),
        'failed': (0, 0, 255),
        'exploring': (255, 200, 0),
    }
    
    err_color = (0, 255, 0) if state.get('error_2d', 1) < 0.1 else \
               (0, 255, 255) if state.get('error_2d', 1) < 0.5 else (0, 0, 255)
    
    lines = [
        (f"State: {state['state']}", state_colors.get(state['state'], (200, 200, 200))),
        (f"Goal: {state['goal'] or 'None'}", (200, 200, 200)),
        (f"Floor: {state['floor']}", (200, 200, 200)),
        ("", (0, 0, 0)),
        (f"SLAM: ({state['position'][0]:+.2f}, {state['position'][2]:+.2f})", (255, 200, 100)),
        (f"GT:   ({gt_pos[0]:+.2f}, {gt_pos[2]:+.2f})", (100, 255, 100)),
        (f"Error: {state.get('error_2d', 0):.4f}m", err_color),
        ("", (0, 0, 0)),
        (f"Distance: {stats['distance']:.1f}m", (200, 200, 200)),
        (f"Collisions: {stats['collisions']}", (200, 200, 200)),
        (f"Landmarks: {stats['landmarks']}", (200, 200, 200)),
        (f"Drift: {stats.get('drift', 0):.2f}%", (200, 200, 200)),
        ("", (0, 0, 0)),
        ("G=VLM cmd  N=Navigate  L=List", (100, 100, 100)),
        ("SPACE=Auto  C=Correct  R=Reset", (100, 100, 100)),
    ]
    
    for i, (text, color) in enumerate(lines):
        cv2.putText(info, text, (10, 20 + i*19),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    # Collision indicator
    if state.get('collision'):
        cv2.rectangle(info, (350, 10), (390, 30), (0, 0, 255), -1)
        cv2.putText(info, "COL", (355, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    bottom = np.hstack([map_vis, info])
    
    return np.vstack([top, bottom])


if __name__ == '__main__':
    main()
