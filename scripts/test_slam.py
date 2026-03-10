#!/usr/bin/env python3
"""
SLAM Evaluation Script
======================

Test Visual SLAM system with Habitat simulator.
Compares estimated trajectory with ground truth.

Usage:
    python test_slam.py --scene /path/to/scene.glb --output ./slam_results
"""

import argparse
import numpy as np
import cv2
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.types import Pose, TrackingState
from core.transforms import CameraIntrinsics
from slam import VisualSLAM, SLAMConfig, VOConfig, LoopClosureConfig


def setup_habitat(scene_path: str):
    """Setup Habitat simulator."""
    import habitat_sim
    
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    
    # RGB sensor
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [480, 640]
    rgb_spec.position = [0.0, 1.25, 0.0]
    rgb_spec.hfov = 90
    
    # Depth sensor
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [480, 640]
    depth_spec.position = [0.0, 1.25, 0.0]
    depth_spec.hfov = 90
    
    # Agent config
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


def get_ground_truth_pose(sim) -> Pose:
    """Get ground truth pose from Habitat."""
    state = sim.get_agent(0).get_state()
    
    position = np.array(state.position)
    quat = np.array([state.rotation.x, state.rotation.y, 
                     state.rotation.z, state.rotation.w])
    
    # Convert quaternion to rotation matrix
    from core.types import quat_to_rotation_matrix
    rotation = quat_to_rotation_matrix(quat)
    
    return Pose(position=position, rotation=rotation)


def create_visualization(rgb, depth, slam_result, slam, gt_pose):
    """Create visualization panel."""
    h, w = 480, 640
    panel_h, panel_w = 240, 320
    
    # Resize RGB
    rgb_small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (panel_w, panel_h))
    
    # Depth visualization
    depth_vis = cv2.applyColorMap(
        (np.clip(depth / 5.0, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS
    )
    depth_small = cv2.resize(depth_vis, (panel_w, panel_h))
    
    # Trajectory visualization
    traj_panel = np.zeros((panel_h * 2, panel_w, 3), dtype=np.uint8)
    traj_panel[:, :] = [30, 30, 30]
    
    # Draw trajectories
    est_traj = slam.get_trajectory()
    gt_traj = slam.get_ground_truth_trajectory()
    
    if len(est_traj) > 1:
        # Scale to fit panel
        all_points = est_traj[:, [0, 2]]  # X, Z
        if gt_traj is not None and len(gt_traj) > 0:
            all_points = np.vstack([all_points, gt_traj[:, [0, 2]]])
        
        min_pt = np.min(all_points, axis=0)
        max_pt = np.max(all_points, axis=0)
        range_pt = max_pt - min_pt
        range_pt[range_pt < 0.1] = 0.1
        
        def to_pixel(pt):
            x = int((pt[0] - min_pt[0]) / range_pt[0] * (panel_w - 40) + 20)
            y = int((pt[1] - min_pt[1]) / range_pt[1] * (panel_h * 2 - 40) + 20)
            return (x, y)
        
        # Draw ground truth (green)
        if gt_traj is not None:
            for i in range(len(gt_traj) - 1):
                p1 = to_pixel(gt_traj[i, [0, 2]])
                p2 = to_pixel(gt_traj[i + 1, [0, 2]])
                cv2.line(traj_panel, p1, p2, (0, 255, 0), 1)
        
        # Draw estimated (blue)
        for i in range(len(est_traj) - 1):
            p1 = to_pixel(est_traj[i, [0, 2]])
            p2 = to_pixel(est_traj[i + 1, [0, 2]])
            cv2.line(traj_panel, p1, p2, (255, 100, 0), 2)
        
        # Current position
        curr_px = to_pixel(est_traj[-1, [0, 2]])
        cv2.circle(traj_panel, curr_px, 5, (0, 255, 255), -1)
    
    cv2.putText(traj_panel, "Green: GT, Blue: Est", (10, panel_h * 2 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Info panel
    info_panel = np.zeros((panel_h * 2, panel_w, 3), dtype=np.uint8)
    info_panel[:, :] = [30, 30, 30]
    
    stats = slam.get_statistics()
    
    # State color
    state_colors = {
        'ok': (0, 255, 0),
        'lost': (0, 0, 255),
        'not_initialized': (200, 200, 200),
    }
    state_color = state_colors.get(slam_result.tracking_state.value, (200, 200, 200))
    
    lines = [
        (f"State: {slam_result.tracking_state.value}", state_color),
        (f"Frame: {stats['frame_count']}", (200, 200, 200)),
        (f"Keyframes: {stats['keyframe_count']}", (200, 200, 200)),
        (f"Loop Closures: {stats['loop_closure_count']}", (255, 200, 0) if stats['loop_closure_count'] > 0 else (200, 200, 200)),
        ("", (0, 0, 0)),
        (f"Processing: {slam_result.processing_time_ms:.1f}ms", (200, 200, 200)),
        (f"Distance: {stats['total_distance']:.2f}m", (200, 200, 200)),
        ("", (0, 0, 0)),
    ]
    
    # Add ATE if available
    if 'ate_rmse' in stats:
        ate = stats['ate_rmse']
        color = (0, 255, 0) if ate < 0.1 else (0, 255, 255) if ate < 0.3 else (0, 0, 255)
        lines.append((f"ATE RMSE: {ate:.4f}m", color))
    
    if 'rpe_translation' in stats:
        lines.append((f"RPE Trans: {stats['rpe_translation']:.4f}m", (200, 200, 200)))
    
    # Estimated pose
    est_pose = slam_result.estimated_pose
    lines.append(("", (0, 0, 0)))
    lines.append((f"Est Pos: ({est_pose.position[0]:.2f}, {est_pose.position[2]:.2f})", (200, 200, 200)))
    lines.append((f"GT Pos: ({gt_pose.position[0]:.2f}, {gt_pose.position[2]:.2f})", (0, 255, 0)))
    
    for i, (line, color) in enumerate(lines):
        cv2.putText(info_panel, line, (10, 25 + i * 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    
    # Combine panels
    left_col = np.vstack([rgb_small, depth_small])
    right_col = np.vstack([traj_panel[:panel_h], info_panel[:panel_h]])
    bottom = np.hstack([traj_panel[panel_h:], info_panel[panel_h:]])
    
    top = np.hstack([left_col, right_col])
    vis = np.vstack([top, bottom])
    
    # Labels
    cv2.putText(vis, "RGB", (5, panel_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(vis, "Depth", (5, 2 * panel_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(vis, "Trajectory", (panel_w + 5, panel_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(vis, "Statistics", (panel_w + 5, 2 * panel_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    # Controls
    cv2.putText(vis, "WASD: Move | R: Reset | Space: Pause | Q: Quit",
               (10, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)
    
    return vis


def main():
    parser = argparse.ArgumentParser(description='Test Visual SLAM')
    parser.add_argument('--scene', required=True, help='Habitat scene file')
    parser.add_argument('--output', default='./slam_results', help='Output directory')
    parser.add_argument('--max-frames', type=int, default=1000, help='Maximum frames to process')
    parser.add_argument('--auto', action='store_true', help='Automatic exploration')
    args = parser.parse_args()
    
    print("="*60)
    print("Visual SLAM Test")
    print("="*60)
    
    # Setup Habitat
    print("Loading simulator...")
    sim = setup_habitat(args.scene)
    
    # Initialize at navigable point
    agent = sim.get_agent(0)
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
    
    print(f"Start position: {agent.get_state().position}")
    
    # Setup camera
    camera = CameraIntrinsics(
        width=640,
        height=480,
        hfov=90.0
    )
    
    # Setup SLAM
    config = SLAMConfig(
        vo_config=VOConfig(
            n_features=1000,
            use_icp=True,
            min_inliers=20
        ),
        lc_config=LoopClosureConfig(
            min_keyframe_gap=20,
            similarity_threshold=0.7,
            min_inliers=30
        ),
        keyframe_translation=0.2,
        keyframe_rotation=0.15,
        optimize_every_n_keyframes=10,
        record_ground_truth=True
    )
    
    slam = VisualSLAM(camera, config)
    
    # Window
    cv2.namedWindow('SLAM Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SLAM Test', 640, 480)
    
    frame_count = 0
    paused = False
    auto_explore = args.auto
    
    # Simple exploration pattern for auto mode
    explore_actions = ['move_forward'] * 5 + ['turn_left'] * 2
    explore_idx = 0
    
    print("Starting SLAM...")
    print("Controls: WASD=Move, R=Reset, Space=Pause, Q=Quit")
    
    while frame_count < args.max_frames:
        # Get observations
        obs = sim.get_sensor_observations()
        rgb = obs['color_sensor'][:, :, :3]
        depth = obs['depth_sensor']
        
        # Get ground truth pose
        gt_pose = get_ground_truth_pose(sim)
        
        # Process frame through SLAM
        slam_result = slam.process_frame(
            rgb, depth,
            timestamp=frame_count * 0.033,
            ground_truth_pose=gt_pose
        )
        
        # Visualization
        vis = create_visualization(rgb, depth, slam_result, slam, gt_pose)
        cv2.imshow('SLAM Test', vis)
        
        # Handle input
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        action = None
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            slam.reset()
            print("SLAM reset")
        elif key == ord('w') or key == 82:
            action = 'move_forward'
            auto_explore = False
        elif key == ord('a') or key == 81:
            action = 'turn_left'
            auto_explore = False
        elif key == ord('d') or key == 83:
            action = 'turn_right'
            auto_explore = False
        elif key == ord('e'):
            auto_explore = True
        
        # Auto exploration
        if auto_explore and not paused:
            action = explore_actions[explore_idx % len(explore_actions)]
            explore_idx += 1
        
        # Execute action
        if action and not paused:
            sim.step(action)
            frame_count += 1
        
        # Print periodic stats
        if frame_count % 100 == 0 and frame_count > 0:
            stats = slam.get_statistics()
            print(f"\nFrame {frame_count}:")
            print(f"  Keyframes: {stats['keyframe_count']}")
            print(f"  Loop closures: {stats['loop_closure_count']}")
            if 'ate_rmse' in stats:
                print(f"  ATE RMSE: {stats['ate_rmse']:.4f}m")
            print(f"  Avg time: {stats['avg_processing_time_ms']:.1f}ms")
        
        if cv2.getWindowProperty('SLAM Test', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Final statistics
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    
    stats = slam.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    output_dir = Path(args.output)
    slam.save_results(str(output_dir))
    
    # Cleanup
    cv2.destroyAllWindows()
    sim.close()
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
