#!/usr/bin/env python3
"""
SLAM Validation Test V2
=======================

Fixed issues:
1. Proper action timing (apply action, THEN compare GT)
2. Floor change detection
3. Better error tracking
4. Uses RobustSLAMv2 with floor handling

Usage:
    python test_slam_v2.py --scene /path/to/scene.glb
"""

import argparse
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from slam.robust_slam_v2 import RobustSLAMv2, LocalizationState


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
    
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def get_gt_pose(sim):
    """Get ground truth pose from Habitat."""
    state = sim.get_agent(0).get_state()
    pos = np.array(state.position)
    
    x, y, z, w = state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return pos, yaw


def main():
    parser = argparse.ArgumentParser(description='SLAM Validation V2')
    parser.add_argument('--scene', required=True, help='Path to scene')
    parser.add_argument('--steps', type=int, default=500, help='Max steps')
    parser.add_argument('--no-lc', action='store_true', help='Disable loop closure')
    args = parser.parse_args()
    
    print("="*60)
    print("SLAM Validation V2 - With Floor Handling")
    print("="*60)
    
    sim = setup_habitat(args.scene)
    
    # Start at navigable point
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
    
    # Get initial GT
    gt_pos, gt_yaw = get_gt_pose(sim)
    
    # Initialize SLAM
    slam = RobustSLAMv2(enable_loop_closure=not args.no_lc)
    slam.initialize(gt_pos, gt_yaw)
    
    print(f"Loop closure: {'OFF' if args.no_lc else 'ON'}")
    print("="*60)
    
    window = 'SLAM V2 Validation'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 800, 500)
    
    for step in range(args.steps):
        # Get observations
        obs = sim.get_sensor_observations()
        rgb = obs['rgb'][:, :, :3]
        depth = obs['depth']
        
        # Decide action
        r = np.random.random()
        if r < 0.6:
            action = 'move_forward'
        elif r < 0.8:
            action = 'turn_left'
        else:
            action = 'turn_right'
        
        # Execute action FIRST
        sim.step(action)
        
        # Get GT AFTER action
        gt_pos, gt_yaw = get_gt_pose(sim)
        
        # Update SLAM
        result = slam.update(
            action=action,
            gt_pose=(gt_pos, gt_yaw),
            depth=depth,
        )
        
        # Visualization
        vis = create_vis(rgb, depth, result, gt_pos, gt_yaw, slam.get_stats(), step)
        
        cv2.imshow(window, vis)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            slam.initialize(gt_pos, gt_yaw)
            print(f"Reset at step {step}")
        elif key == ord('c'):
            slam.correct_pose(gt_pos, gt_yaw)
            print(f"Corrected at step {step}")
        
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()
    sim.close()
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    stats = slam.get_stats()
    
    print(f"Frames: {stats['frames']}")
    print(f"Distance: {stats['distance']:.2f} m")
    print(f"Floor changes: {stats['floor_changes']}")
    print(f"Current floor: {stats['current_floor']}")
    
    if 'error_2d_mean' in stats:
        print()
        print(f"2D Error (X-Z):")
        print(f"  Mean:    {stats['error_2d_mean']:.4f} m")
        print(f"  Max:     {stats['error_2d_max']:.4f} m")
        print(f"  Current: {stats['error_2d_current']:.4f} m")
        print()
        print(f"Drift rate: {stats['drift_rate']:.2f}%")
        
        if stats['drift_rate'] < 1.0:
            print("\n✓ EXCELLENT - Near-zero drift!")
        elif stats['drift_rate'] < 5.0:
            print("\n✓ GOOD - Acceptable drift")
        else:
            print("\n⚠ High drift - check for issues")


def create_vis(rgb, depth, result, gt_pos, gt_yaw, stats, step):
    """Create visualization."""
    h, w = 250, 400
    
    # RGB panel
    rgb_small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (200, 150))
    
    # Depth panel  
    depth_vis = cv2.applyColorMap(
        (np.clip(depth / 5.0, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS
    )
    depth_small = cv2.resize(depth_vis, (200, 150))
    
    top = np.hstack([rgb_small, depth_small])
    
    # Info panel
    info = np.zeros((h, w, 3), dtype=np.uint8)
    info[:] = [30, 30, 30]
    
    state_colors = {
        LocalizationState.EXCELLENT: (0, 255, 0),
        LocalizationState.GOOD: (0, 200, 100),
        LocalizationState.UNCERTAIN: (0, 255, 255),
        LocalizationState.LOST: (0, 0, 255),
    }
    
    lines = [
        (f"Step: {step}  Floor: {result.current_floor}", (200, 200, 200)),
        (f"State: {result.state.value}", state_colors.get(result.state, (200, 200, 200))),
        ("", (0, 0, 0)),
        (f"SLAM: ({result.position[0]:+.2f}, {result.position[2]:+.2f})", (255, 200, 100)),
        (f"GT:   ({gt_pos[0]:+.2f}, {gt_pos[2]:+.2f})", (100, 255, 100)),
        ("", (0, 0, 0)),
    ]
    
    if result.error_2d is not None:
        err_color = (0, 255, 0) if result.error_2d < 0.1 else (0, 255, 255) if result.error_2d < 0.5 else (0, 0, 255)
        lines.append((f"Error 2D: {result.error_2d:.4f} m", err_color))
    
    if result.error_yaw is not None:
        lines.append((f"Error Yaw: {result.error_yaw:.1f} deg", (200, 200, 200)))
    
    lines.append(("", (0, 0, 0)))
    lines.append((f"Distance: {stats['distance']:.2f} m", (200, 200, 200)))
    
    if 'drift_rate' in stats:
        drift_color = (0, 255, 0) if stats['drift_rate'] < 1 else (0, 255, 255) if stats['drift_rate'] < 5 else (0, 0, 255)
        lines.append((f"Drift: {stats['drift_rate']:.2f}%", drift_color))
    
    for i, (text, color) in enumerate(lines):
        cv2.putText(info, text, (10, 22 + i*20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    # Error bar
    if result.error_2d is not None:
        bar_len = min(int(result.error_2d * 300), w - 20)
        bar_color = (0, 255, 0) if result.error_2d < 0.1 else (0, 255, 255) if result.error_2d < 0.5 else (0, 0, 255)
        cv2.rectangle(info, (10, h - 25), (10 + bar_len, h - 10), bar_color, -1)
    
    vis = np.vstack([top, info])
    
    return vis


if __name__ == '__main__':
    main()
