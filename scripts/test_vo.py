#!/usr/bin/env python3
"""
Test Lightweight Visual Odometry
================================

Compare VO estimates with Habitat ground truth.
Run navigation using VO instead of GT to validate accuracy.

Usage:
    python test_vo.py --scene /path/to/scene.glb
    python test_vo.py --scene /path/to/scene.glb --use-vo  # Navigate with VO
"""

import argparse
import numpy as np
import cv2
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from slam.lightweight_vo import LightweightVO, PoseTracker, TrackingState


def setup_habitat(scene_path: str):
    """Setup Habitat simulator."""
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


def get_gt_pose(sim):
    """Get ground truth pose from Habitat."""
    state = sim.get_agent(0).get_state()
    pos = np.array(state.position)
    
    # Quaternion to rotation matrix
    q = state.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return pos, R


def main():
    parser = argparse.ArgumentParser(description='Test Visual Odometry')
    parser.add_argument('--scene', required=True, help='Habitat scene file')
    parser.add_argument('--use-vo', action='store_true', help='Use VO for navigation')
    parser.add_argument('--max-frames', type=int, default=500)
    args = parser.parse_args()
    
    print("="*60)
    print("Lightweight Visual Odometry Test")
    print("="*60)
    
    # Setup
    sim = setup_habitat(args.scene)
    vo = LightweightVO()
    
    # Initialize at navigable point
    agent = sim.get_agent(0)
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
    
    print(f"Start: {agent.get_state().position}")
    
    # Storage for trajectories
    gt_trajectory = []
    vo_trajectory = []
    
    # Window
    cv2.namedWindow('VO Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('VO Test', 800, 400)
    
    frame = 0
    total_time = 0
    
    print("\nControls: WASD=Move, R=Reset, Q=Quit")
    print("Blue=VO estimate, Green=Ground Truth\n")
    
    while frame < args.max_frames:
        # Get observations
        obs = sim.get_sensor_observations()
        rgb = obs['color_sensor'][:, :, :3]
        depth = obs['depth_sensor']
        
        # Get GT pose
        gt_pos, gt_rot = get_gt_pose(sim)
        gt_trajectory.append(gt_pos.copy())
        
        # Run VO
        start = time.time()
        result = vo.process_frame(rgb, depth)
        vo_time = (time.time() - start) * 1000
        total_time += vo_time
        
        vo_trajectory.append(result.position.copy())
        
        # Compute error
        error = np.linalg.norm(result.position - gt_pos)
        
        # Visualization
        vis_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        vis_rgb = cv2.resize(vis_rgb, (400, 300))
        
        # Trajectory panel
        traj_panel = np.zeros((300, 400, 3), dtype=np.uint8)
        traj_panel[:, :] = [30, 30, 30]
        
        if len(gt_trajectory) > 1:
            # Scale trajectories to panel
            all_pts = np.array(gt_trajectory + vo_trajectory)
            all_xz = all_pts[:, [0, 2]]
            
            min_pt = np.min(all_xz, axis=0)
            max_pt = np.max(all_xz, axis=0)
            range_pt = max_pt - min_pt
            range_pt[range_pt < 0.1] = 0.1
            
            def to_px(p):
                x = int((p[0] - min_pt[0]) / range_pt[0] * 360 + 20)
                y = int((p[1] - min_pt[1]) / range_pt[1] * 260 + 20)
                return (x, y)
            
            # Draw GT (green)
            for i in range(len(gt_trajectory) - 1):
                p1 = to_px(gt_trajectory[i][[0, 2]])
                p2 = to_px(gt_trajectory[i + 1][[0, 2]])
                cv2.line(traj_panel, p1, p2, (0, 255, 0), 1)
            
            # Draw VO (blue)
            for i in range(len(vo_trajectory) - 1):
                p1 = to_px(vo_trajectory[i][[0, 2]])
                p2 = to_px(vo_trajectory[i + 1][[0, 2]])
                cv2.line(traj_panel, p1, p2, (255, 100, 0), 2)
            
            # Current positions
            cv2.circle(traj_panel, to_px(gt_trajectory[-1][[0, 2]]), 4, (0, 255, 0), -1)
            cv2.circle(traj_panel, to_px(vo_trajectory[-1][[0, 2]]), 4, (255, 100, 0), -1)
        
        # Info overlay
        state_color = (0, 255, 0) if result.tracking_state == TrackingState.OK else (0, 0, 255)
        
        cv2.putText(vis_rgb, f"Frame: {frame}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_rgb, f"State: {result.tracking_state.value}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        cv2.putText(vis_rgb, f"Inliers: {result.inliers}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_rgb, f"Error: {error:.3f}m", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_rgb, f"VO Time: {vo_time:.1f}ms", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(traj_panel, "Green=GT, Blue=VO", (10, 290),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Combine
        vis = np.hstack([vis_rgb, traj_panel])
        cv2.imshow('VO Test', vis)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        action = None
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            vo.reset()
            gt_trajectory.clear()
            vo_trajectory.clear()
            print("Reset")
        elif key == ord('w') or key == 82:
            action = 'move_forward'
        elif key == ord('a') or key == 81:
            action = 'turn_left'
        elif key == ord('d') or key == 83:
            action = 'turn_right'
        
        if action:
            sim.step(action)
            frame += 1
        
        if cv2.getWindowProperty('VO Test', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Final stats
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    if len(gt_trajectory) > 0 and len(vo_trajectory) > 0:
        n = min(len(gt_trajectory), len(vo_trajectory))
        gt_arr = np.array(gt_trajectory[:n])
        vo_arr = np.array(vo_trajectory[:n])
        
        errors = np.linalg.norm(gt_arr - vo_arr, axis=1)
        
        print(f"Frames processed: {frame}")
        print(f"Avg VO time: {total_time / max(1, frame):.1f}ms")
        print(f"Mean error: {np.mean(errors):.4f}m")
        print(f"Max error: {np.max(errors):.4f}m")
        print(f"Final error: {errors[-1]:.4f}m")
        
        # Drift rate
        gt_dist = np.sum(np.linalg.norm(np.diff(gt_arr, axis=0), axis=1))
        if gt_dist > 0:
            drift_rate = errors[-1] / gt_dist * 100
            print(f"Drift rate: {drift_rate:.2f}% of distance")
    
    cv2.destroyAllWindows()
    sim.close()


if __name__ == '__main__':
    main()
