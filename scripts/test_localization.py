#!/usr/bin/env python3
"""
Test Robust Localization
========================

Compare localization approaches:
1. Ground Truth (Habitat)
2. Pure Visual Odometry (drifts badly)
3. Action-based Dead Reckoning
4. Particle Filter with Map Matching

Usage:
    python test_localization.py --scene /path/to/scene.glb --map-dir ./my_map
"""

import argparse
import numpy as np
import cv2
import time
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))


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
    
    # Get yaw from quaternion
    q = state.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return pos, yaw


def main():
    parser = argparse.ArgumentParser(description='Test Robust Localization')
    parser.add_argument('--scene', required=True, help='Habitat scene file')
    parser.add_argument('--map-dir', required=True, help='Directory with pre-built map')
    parser.add_argument('--max-frames', type=int, default=300)
    args = parser.parse_args()
    
    print("="*60)
    print("Robust Localization Test")
    print("="*60)
    
    # Load map
    map_dir = Path(args.map_dir)
    
    # Load occupancy map
    occ_path = map_dir / 'floor_0' / 'navigable_map.pt'
    if not occ_path.exists():
        occ_path = map_dir / 'floor_0' / 'occupancy_map.pt'
    
    occupancy_map = torch.load(occ_path).numpy()
    print(f"Loaded map: {occupancy_map.shape}")
    
    # Load map parameters
    import json
    stats_path = map_dir / 'statistics.json'
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        origin_x = stats.get('origin_x', 0.0)
        origin_z = stats.get('origin_z', 0.0)
        resolution = stats.get('resolution_cm', 5.0)
        map_size = stats.get('map_size', occupancy_map.shape[0])
    else:
        origin_x, origin_z = 0.0, 0.0
        resolution = 5.0
        map_size = occupancy_map.shape[0]
    
    print(f"Map params: origin=({origin_x:.2f}, {origin_z:.2f}), res={resolution}cm")
    
    # Setup localizers
    from slam.robust_localizer import HybridLocalizer, ActionModel
    
    localizer = HybridLocalizer(
        occupancy_map=occupancy_map,
        resolution_cm=resolution,
        origin_x=origin_x,
        origin_z=origin_z,
        map_size=map_size,
        correction_interval=3,  # Correct every 3 frames
        n_particles=100
    )
    
    # Dead reckoning only (for comparison)
    dr_position = None
    dr_yaw = None
    
    # Setup simulator
    sim = setup_habitat(args.scene)
    
    # Initialize at navigable point
    agent = sim.get_agent(0)
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
    
    # Get initial pose
    gt_pos, gt_yaw = get_gt_pose(sim)
    print(f"Start: pos=({gt_pos[0]:.2f}, {gt_pos[2]:.2f}), yaw={np.degrees(gt_yaw):.1f}°")
    
    # Initialize localizers
    localizer.initialize(gt_pos, gt_yaw)
    dr_position = gt_pos.copy()
    dr_yaw = gt_yaw
    
    # Storage
    gt_trajectory = [gt_pos.copy()]
    pf_trajectory = [gt_pos.copy()]
    dr_trajectory = [gt_pos.copy()]
    
    # Window
    cv2.namedWindow('Localization Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Localization Test', 900, 450)
    
    frame = 0
    total_time = 0
    last_action = None
    
    print("\nControls: WASD=Move, R=Reset, Q=Quit")
    print("Green=GT, Blue=Particle Filter, Red=Dead Reckoning\n")
    
    while frame < args.max_frames:
        # Get observations
        obs = sim.get_sensor_observations()
        rgb = obs['color_sensor'][:, :, :3]
        depth = obs['depth_sensor']
        
        # Get GT pose (this is AFTER the previous action was applied)
        gt_pos, gt_yaw = get_gt_pose(sim)
        gt_trajectory.append(gt_pos.copy())
        
        # Update localizers with the action that was just applied
        start = time.time()
        result = localizer.update(last_action, depth)
        pf_time = (time.time() - start) * 1000
        total_time += pf_time
        
        pf_trajectory.append(result.position.copy())
        
        # Dead reckoning update - apply the action that Habitat just executed
        if last_action is not None:
            dr_position, dr_yaw = ActionModel.apply_action(dr_position, dr_yaw, last_action)
        dr_trajectory.append(dr_position.copy())
        
        # Compute errors
        pf_error = np.linalg.norm(result.position[[0, 2]] - gt_pos[[0, 2]])
        dr_error = np.linalg.norm(dr_position[[0, 2]] - gt_pos[[0, 2]])
        
        # Visualization
        vis_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        vis_rgb = cv2.resize(vis_rgb, (300, 225))
        
        # Trajectory panel
        traj_panel = np.zeros((225, 300, 3), dtype=np.uint8)
        traj_panel[:, :] = [30, 30, 30]
        
        if len(gt_trajectory) > 1:
            all_pts = np.array(gt_trajectory + pf_trajectory + dr_trajectory)
            all_xz = all_pts[:, [0, 2]]
            
            min_pt = np.min(all_xz, axis=0) - 0.5
            max_pt = np.max(all_xz, axis=0) + 0.5
            range_pt = max_pt - min_pt
            range_pt[range_pt < 1.0] = 1.0
            
            def to_px(p):
                x = int((p[0] - min_pt[0]) / range_pt[0] * 260 + 20)
                y = int((p[1] - min_pt[1]) / range_pt[1] * 185 + 20)
                return (x, y)
            
            # Draw trajectories
            for i in range(len(gt_trajectory) - 1):
                p1 = to_px(gt_trajectory[i][[0, 2]])
                p2 = to_px(gt_trajectory[i + 1][[0, 2]])
                cv2.line(traj_panel, p1, p2, (0, 255, 0), 1)  # Green = GT
            
            for i in range(len(pf_trajectory) - 1):
                p1 = to_px(pf_trajectory[i][[0, 2]])
                p2 = to_px(pf_trajectory[i + 1][[0, 2]])
                cv2.line(traj_panel, p1, p2, (255, 100, 0), 2)  # Blue = PF
            
            for i in range(len(dr_trajectory) - 1):
                p1 = to_px(dr_trajectory[i][[0, 2]])
                p2 = to_px(dr_trajectory[i + 1][[0, 2]])
                cv2.line(traj_panel, p1, p2, (0, 0, 255), 1)  # Red = DR
            
            # Current positions
            cv2.circle(traj_panel, to_px(gt_trajectory[-1][[0, 2]]), 4, (0, 255, 0), -1)
            cv2.circle(traj_panel, to_px(pf_trajectory[-1][[0, 2]]), 4, (255, 100, 0), -1)
            cv2.circle(traj_panel, to_px(dr_trajectory[-1][[0, 2]]), 3, (0, 0, 255), -1)
        
        cv2.putText(traj_panel, "G=GT B=PF R=DR", (10, 215),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Map panel (show occupancy and particle positions)
        map_panel = np.zeros((225, 300, 3), dtype=np.uint8)
        
        # Resize map to fit
        map_vis = (occupancy_map * 255).astype(np.uint8)
        map_vis = cv2.resize(map_vis, (200, 200))
        map_vis = cv2.cvtColor(map_vis, cv2.COLOR_GRAY2BGR)
        map_vis[map_vis[:,:,0] > 128] = [100, 100, 100]
        
        # Draw particles
        if hasattr(localizer.pf, 'particles') and localizer.pf.particles is not None:
            for i in range(min(30, localizer.pf.n_particles)):
                px = localizer.pf.particles[i, 0]
                pz = localizer.pf.particles[i, 2]
                mx = int((px - origin_x) * 100 / resolution + map_size // 2)
                my = int((pz - origin_z) * 100 / resolution + map_size // 2)
                # Scale to visualization
                vx = int(mx * 200 / map_size)
                vy = int(my * 200 / map_size)
                if 0 <= vx < 200 and 0 <= vy < 200:
                    cv2.circle(map_vis, (vx, vy), 2, (0, 255, 255), -1)
        
        # Draw GT position on map
        mx = int((gt_pos[0] - origin_x) * 100 / resolution + map_size // 2)
        my = int((gt_pos[2] - origin_z) * 100 / resolution + map_size // 2)
        vx = int(mx * 200 / map_size)
        vy = int(my * 200 / map_size)
        if 0 <= vx < 200 and 0 <= vy < 200:
            cv2.circle(map_vis, (vx, vy), 5, (0, 255, 0), -1)
        
        map_panel[12:212, 50:250] = map_vis
        cv2.putText(map_panel, "Particles", (100, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Info panel
        info_panel = np.zeros((225, 300, 3), dtype=np.uint8)
        info_panel[:, :] = [30, 30, 30]
        
        pf_color = (0, 255, 0) if pf_error < 0.3 else (0, 255, 255) if pf_error < 1.0 else (0, 0, 255)
        dr_color = (0, 255, 0) if dr_error < 0.3 else (0, 255, 255) if dr_error < 1.0 else (0, 0, 255)
        conf_color = (0, 255, 0) if result.confidence > 0.5 else (0, 255, 255) if result.confidence > 0.2 else (0, 0, 255)
        
        lines = [
            (f"Frame: {frame}", (200, 200, 200)),
            (f"PF Error: {pf_error:.3f}m", pf_color),
            (f"DR Error: {dr_error:.3f}m", dr_color),
            (f"Confidence: {result.confidence:.2f}", conf_color),
            (f"State: {result.state.value}", conf_color),
            (f"Time: {pf_time:.1f}ms", (200, 200, 200)),
            ("", (0, 0, 0)),
            (f"GT: ({gt_pos[0]:.2f}, {gt_pos[2]:.2f})", (0, 255, 0)),
            (f"PF: ({result.position[0]:.2f}, {result.position[2]:.2f})", (255, 100, 0)),
        ]
        
        for i, (line, color) in enumerate(lines):
            cv2.putText(info_panel, line, (10, 22 + i*22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Combine
        top = np.hstack([vis_rgb, traj_panel, map_panel])
        
        # Depth panel
        depth_vis = cv2.applyColorMap(
            (np.clip(depth / 5.0, 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS
        )
        depth_small = cv2.resize(depth_vis, (300, 225))
        
        bottom = np.hstack([depth_small, info_panel, np.zeros((225, 300, 3), np.uint8)])
        
        vis = np.vstack([top, bottom])
        cv2.imshow('Localization Test', vis)
        
        # Handle input - get next action
        key = cv2.waitKey(30) & 0xFF
        next_action = None  # Don't reset last_action here!
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            # Reset
            gt_pos, gt_yaw = get_gt_pose(sim)
            localizer.initialize(gt_pos, gt_yaw)
            dr_position = gt_pos.copy()
            dr_yaw = gt_yaw
            gt_trajectory = [gt_pos.copy()]
            pf_trajectory = [gt_pos.copy()]
            dr_trajectory = [gt_pos.copy()]
            last_action = None
            print("Reset")
        elif key == ord('w') or key == 82:
            next_action = 'move_forward'
        elif key == ord('a') or key == 81:
            next_action = 'turn_left'
        elif key == ord('d') or key == 83:
            next_action = 'turn_right'
        elif key == ord('e'):
            # Auto explore
            next_action = np.random.choice(['move_forward', 'move_forward', 'move_forward', 'turn_left', 'turn_right'])
        
        if next_action:
            sim.step(next_action)
            last_action = next_action  # Store for next iteration's DR update
            frame += 1
        else:
            last_action = None  # No action this frame
        
        if cv2.getWindowProperty('Localization Test', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Final stats
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    if len(gt_trajectory) > 1:
        gt_arr = np.array(gt_trajectory)
        pf_arr = np.array(pf_trajectory)
        dr_arr = np.array(dr_trajectory)
        
        n = min(len(gt_arr), len(pf_arr), len(dr_arr))
        
        pf_errors = np.linalg.norm(gt_arr[:n, [0, 2]] - pf_arr[:n, [0, 2]], axis=1)
        dr_errors = np.linalg.norm(gt_arr[:n, [0, 2]] - dr_arr[:n, [0, 2]], axis=1)
        
        gt_dist = np.sum(np.linalg.norm(np.diff(gt_arr[:n], axis=0), axis=1))
        
        print(f"Frames: {frame}")
        print(f"Distance traveled: {gt_dist:.2f}m")
        print(f"Avg PF time: {total_time / max(1, frame):.1f}ms")
        print()
        print("Particle Filter:")
        print(f"  Mean error: {np.mean(pf_errors):.4f}m")
        print(f"  Max error: {np.max(pf_errors):.4f}m")
        print(f"  Final error: {pf_errors[-1]:.4f}m")
        print(f"  Drift rate: {pf_errors[-1] / max(gt_dist, 0.01) * 100:.2f}%")
        print()
        print("Dead Reckoning:")
        print(f"  Mean error: {np.mean(dr_errors):.4f}m")
        print(f"  Max error: {np.max(dr_errors):.4f}m")
        print(f"  Final error: {dr_errors[-1]:.4f}m")
        print(f"  Drift rate: {dr_errors[-1] / max(gt_dist, 0.01) * 100:.2f}%")
    
    cv2.destroyAllWindows()
    sim.close()


if __name__ == '__main__':
    main()