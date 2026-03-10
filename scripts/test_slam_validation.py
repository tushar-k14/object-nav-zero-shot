#!/usr/bin/env python3
"""
SLAM Validation Test
====================

Test the SLAM system in Habitat simulator with ground truth comparison.

Usage:
    python test_slam_validation.py --scene /path/to/scene.glb

Tests:
1. Dead reckoning accuracy
2. Loop closure detection
3. Visual odometry (if enabled)
4. User correction
5. Depth comparison (hardware vs monocular)
"""

import argparse
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from slam.robust_slam import RobustSLAM, LocalizationState


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


def get_camera_K():
    """Get camera intrinsic matrix."""
    width, height = 640, 480
    hfov = np.radians(90)
    fx = width / (2.0 * np.tan(hfov / 2))
    fy = fx
    cx, cy = width / 2, height / 2
    
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='SLAM Validation Test')
    parser.add_argument('--scene', required=True, help='Path to scene (.glb)')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps')
    parser.add_argument('--use-vo', action='store_true', help='Enable visual odometry')
    parser.add_argument('--use-pf', action='store_true', help='Enable particle filter')
    parser.add_argument('--test-depth', action='store_true', help='Test monocular depth')
    args = parser.parse_args()
    
    print("="*60)
    print("SLAM Validation Test")
    print("="*60)
    
    # Setup
    sim = setup_habitat(args.scene)
    
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
    
    camera_K = get_camera_K()
    
    # Create SLAM system
    slam = RobustSLAM(
        use_vo=args.use_vo,
        use_loop_closure=True,
        use_particle_filter=args.use_pf,
        camera_K=camera_K if args.use_vo else None,
    )
    
    # Initialize
    gt_pos, gt_yaw = get_gt_pose(sim)
    slam.initialize(gt_pos, gt_yaw)
    
    print(f"Start: ({gt_pos[0]:.2f}, {gt_pos[2]:.2f})")
    print(f"VO: {'ON' if args.use_vo else 'OFF'}")
    print(f"PF: {'ON' if args.use_pf else 'OFF'}")
    print("="*60)
    
    # Optional: Depth comparison
    depth_comparator = None
    if args.test_depth:
        try:
            from slam.depth_comparison import DepthComparator
            depth_comparator = DepthComparator(model='midas_small')
            print(f"Depth comparison: {depth_comparator.model_name}")
        except Exception as e:
            print(f"Depth comparison not available: {e}")
    
    # Test loop
    window = 'SLAM Validation'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1000, 600)
    
    errors = []
    actions_taken = []
    
    for step in range(args.steps):
        # Get observations
        obs = sim.get_sensor_observations()
        rgb = obs['rgb'][:, :, :3]
        depth = obs['depth']
        
        # Get ground truth
        gt_pos, gt_yaw = get_gt_pose(sim)
        
        # Random action (or could use keyboard)
        r = np.random.random()
        if r < 0.6:
            action = 'move_forward'
        elif r < 0.8:
            action = 'turn_left'
        else:
            action = 'turn_right'
        
        # Update SLAM
        result = slam.update(
            action=actions_taken[-1] if actions_taken else None,
            rgb=rgb,
            depth=depth,
            gt_pose=(gt_pos, gt_yaw),
        )
        
        # Execute action
        sim.step(action)
        actions_taken.append(action)
        
        if result.gt_error is not None:
            errors.append(result.gt_error)
        
        # Visualization
        vis = create_visualization(
            rgb, depth, result, gt_pos, gt_yaw, slam.get_stats()
        )
        
        # Depth comparison overlay
        if depth_comparator is not None and step % 10 == 0:
            metrics = depth_comparator.compare(rgb, depth)
            if metrics:
                cv2.putText(vis, f"Depth MAE: {metrics.mae:.3f}m", (10, vis.shape[0]-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(vis, f"Depth Time: {metrics.inference_time_ms:.1f}ms", (10, vis.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow(window, vis)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            # User correction to GT
            slam.correct_pose(gt_pos, gt_yaw)
            print(f"User correction applied at step {step}")
        elif key == ord('r'):
            # Reset to GT
            slam.initialize(gt_pos, gt_yaw)
            errors.clear()
            print(f"Reset at step {step}")
        
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()
    sim.close()
    
    # Final report
    print("\n" + "="*60)
    print("SLAM Validation Results")
    print("="*60)
    
    stats = slam.get_stats()
    
    print(f"Total frames: {stats['frame_count']}")
    print(f"Total distance: {stats['total_distance']:.2f}m")
    print(f"Final confidence: {stats['confidence']:.2f}")
    
    if 'gt_error_mean' in stats:
        print(f"\nGround Truth Comparison:")
        print(f"  Mean error: {stats['gt_error_mean']:.4f}m")
        print(f"  Max error: {stats['gt_error_max']:.4f}m")
        print(f"  Drift rate: {stats['drift_rate']:.2f}%")
    
    if 'time_total_ms' in stats:
        print(f"\nTiming:")
        print(f"  DR: {stats.get('time_dr_ms', 0):.2f}ms")
        print(f"  VO: {stats.get('time_vo_ms', 0):.2f}ms")
        print(f"  LC: {stats.get('time_lc_ms', 0):.2f}ms")
        print(f"  Total: {stats['time_total_ms']:.2f}ms")
    
    print(f"\nLoop closure candidates: {stats['lc_candidates']}")
    print(f"User corrections: {stats['user_corrections']}")
    
    if depth_comparator is not None:
        depth_summary = depth_comparator.get_summary()
        if depth_summary:
            print(f"\nMonocular Depth ({depth_comparator.model_name}):")
            print(f"  MAE: {depth_summary['mae_mean']:.3f}m ± {depth_summary['mae_std']:.3f}m")
            print(f"  Inference: {depth_summary['inference_time_mean_ms']:.1f}ms")
    
    # Verdict
    print("\n" + "="*60)
    if 'drift_rate' in stats:
        if stats['drift_rate'] < 1.0:
            print("✓ EXCELLENT - Drift < 1%")
        elif stats['drift_rate'] < 5.0:
            print("✓ GOOD - Drift < 5%")
        elif stats['drift_rate'] < 10.0:
            print("⚠ ACCEPTABLE - Drift < 10%")
        else:
            print("✗ NEEDS IMPROVEMENT - Drift > 10%")
    print("="*60)


def create_visualization(rgb, depth, result, gt_pos, gt_yaw, stats):
    """Create visualization panel."""
    h, w = 300, 400
    
    # RGB
    rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rgb_vis = cv2.resize(rgb_vis, (w, h))
    
    # Depth
    depth_vis = cv2.applyColorMap(
        (np.clip(depth / 5.0, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS
    )
    depth_vis = cv2.resize(depth_vis, (w, h))
    
    # Info panel
    info = np.zeros((h, w, 3), dtype=np.uint8)
    info[:] = [30, 30, 30]
    
    # State colors
    state_colors = {
        LocalizationState.EXCELLENT: (0, 255, 0),
        LocalizationState.GOOD: (0, 200, 100),
        LocalizationState.UNCERTAIN: (0, 255, 255),
        LocalizationState.LOST: (0, 0, 255),
    }
    
    lines = [
        (f"State: {result.state.value}", state_colors.get(result.state, (200, 200, 200))),
        (f"Confidence: {result.confidence:.2f}", (200, 200, 200)),
        ("", (0, 0, 0)),
        (f"SLAM: ({result.position[0]:.2f}, {result.position[2]:.2f})", (255, 200, 100)),
        (f"GT:   ({gt_pos[0]:.2f}, {gt_pos[2]:.2f})", (100, 255, 100)),
        (f"Error: {result.gt_error:.4f}m" if result.gt_error else "Error: N/A", 
         (0, 255, 0) if result.gt_error and result.gt_error < 0.1 else (0, 255, 255)),
        ("", (0, 0, 0)),
        (f"Distance: {stats['total_distance']:.2f}m", (200, 200, 200)),
        (f"Drift: {stats.get('drift_rate', 0):.2f}%", (200, 200, 200)),
        ("", (0, 0, 0)),
        ("C = Correct to GT", (150, 150, 150)),
        ("R = Reset", (150, 150, 150)),
        ("Q = Quit", (150, 150, 150)),
    ]
    
    for i, (text, color) in enumerate(lines):
        cv2.putText(info, text, (10, 25 + i*22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    # Error plot (simple bar)
    if result.gt_error is not None:
        bar_width = int(min(result.gt_error * 100, w - 20))
        bar_color = (0, 255, 0) if result.gt_error < 0.1 else (0, 255, 255) if result.gt_error < 0.5 else (0, 0, 255)
        cv2.rectangle(info, (10, h - 30), (10 + bar_width, h - 15), bar_color, -1)
        cv2.putText(info, f"{result.gt_error:.3f}m", (15 + bar_width, h - 17),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    # Combine
    top = np.hstack([rgb_vis, depth_vis])
    
    # Trajectory panel (placeholder)
    traj = np.zeros((h, w, 3), dtype=np.uint8)
    traj[:] = [40, 40, 40]
    cv2.putText(traj, "Trajectory", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    bottom = np.hstack([traj, info])
    
    vis = np.vstack([top, bottom])
    
    return vis


if __name__ == '__main__':
    main()
