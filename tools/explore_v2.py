#!/usr/bin/env python3
"""
Interactive Explorer V2 - With Real-time YOLO Visualization
"""

import argparse
import numpy as np
import cv2
import sys
from pathlib import Path
import time
import json
import heapq

sys.path.insert(0, str(Path(__file__).parent))

from mapping.fixed_occupancy_map import MapConfig
from mapping.enhanced_semantic_mapper_v2 import EnhancedSemanticMapperV2
from navigation.scene_graph import SceneGraph


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


class InteractiveExplorerV2:
    def __init__(self, sim, mapper, scene_graph, output_dir):
        self.sim = sim
        self.mapper = mapper
        self.scene_graph = scene_graph
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.step_count = 0
        self.auto_mode = False
        self.nav_path = []
        self.nav_index = 0
        self.start_time = time.time()
    
    def get_agent_state(self):
        state = self.sim.get_agent(0).get_state()
        return np.array(state.position), np.array([state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w])
    
    def get_yaw(self, q):
        return np.arctan2(2*(q[3]*q[1] + q[2]*q[0]), 1 - 2*(q[0]**2 + q[1]**2))
    
    def step(self, action):
        if action:
            self.sim.step(action)
            self.step_count += 1
        obs = self.sim.get_sensor_observations()
        rgb, depth = obs['color_sensor'][:,:,:3], obs['depth_sensor']
        pos, rot = self.get_agent_state()
        stats = self.mapper.update(rgb, depth, pos, rot)
        self.scene_graph.update_from_tracker(self.mapper.tracked_objects)
        return rgb, depth, pos, rot, stats
    
    def plan_path(self, gx, gy):
        pos, _ = self.get_agent_state()
        occ = self.mapper.occupancy_map
        start = occ._world_to_map(pos[0], pos[2])
        obs = cv2.dilate(occ.get_obstacle_map(), np.ones((5,5), np.uint8))
        
        def h(a, b): return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
        cnt, open_set, came_from, g = 0, [(h(start, (gx,gy)), 0, start)], {}, {start: 0}
        for _ in range(5000):
            if not open_set: break
            _, _, cur = heapq.heappop(open_set)
            if h(cur, (gx,gy)) < 5:
                path = [cur]
                while cur in came_from: cur = came_from[cur]; path.append(cur)
                return path[::-1][::3]
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = cur[0]+dx, cur[1]+dy
                if 0 <= nx < obs.shape[1] and 0 <= ny < obs.shape[0] and obs[ny,nx] == 0:
                    tg = g[cur] + (1.414 if abs(dx)+abs(dy)==2 else 1)
                    if (nx,ny) not in g or tg < g[(nx,ny)]:
                        came_from[(nx,ny)] = cur; g[(nx,ny)] = tg; cnt += 1
                        heapq.heappush(open_set, (tg + h((nx,ny), (gx,gy)), cnt, (nx,ny)))
        return []
    
    def get_auto_action(self, pos, rot):
        if not self.nav_path or self.nav_index >= len(self.nav_path):
            self.auto_mode = False
            return None
        occ = self.mapper.occupancy_map
        am = occ._world_to_map(pos[0], pos[2])
        tgt = self.nav_path[self.nav_index]
        if np.sqrt((tgt[0]-am[0])**2 + (tgt[1]-am[1])**2) < 3:
            self.nav_index += 1
            if self.nav_index >= len(self.nav_path):
                self.auto_mode = False
                return None
            tgt = self.nav_path[self.nav_index]
        yaw = self.get_yaw(rot)
        ta = np.arctan2(-(tgt[1]-am[1]), tgt[0]-am[0])
        diff = ta - yaw
        while diff > np.pi: diff -= 2*np.pi
        while diff < -np.pi: diff += 2*np.pi
        return 'move_forward' if abs(diff) < np.radians(15) else ('turn_left' if diff > 0 else 'turn_right')
    
    def create_vis(self, rgb, depth, stats):
        h, w = 300, 400
        rgb_bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (w, h))
        det = cv2.resize(cv2.cvtColor(self.mapper.visualize_detections(rgb), cv2.COLOR_RGB2BGR), (w, h))
        dep = cv2.resize(cv2.applyColorMap((np.clip(depth/5,0,1)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), (w, h))
        map_vis = self.mapper.visualize(show_legend=True)
        map_h = h
        map_w = int(map_vis.shape[1] * map_h / map_vis.shape[0])
        map_r = cv2.resize(map_vis, (map_w, map_h))
        
        top = np.hstack([rgb_bgr, det])
        bot_base = np.hstack([dep, map_r])
        if bot_base.shape[1] < top.shape[1]:
            bot_base = np.hstack([bot_base, np.zeros((h, top.shape[1]-bot_base.shape[1], 3), np.uint8)])
        elif bot_base.shape[1] > top.shape[1]:
            top = np.hstack([top, np.zeros((h, bot_base.shape[1]-top.shape[1], 3), np.uint8)])
        
        vis = np.vstack([top, bot_base])
        info = f"Step:{self.step_count} | Floor:{stats.get('current_floor',0)} | Det:{stats.get('detections',0)} | Obj:{stats.get('confirmed_objects',0)} | Expl:{stats.get('explored_percent',0):.1f}%"
        cv2.putText(vis, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(vis, "RGB", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(vis, "YOLO Detections", (w+10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(vis, "Depth", (10, 2*h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(vis, "Semantic Map", (w+10, 2*h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        return vis, h, w, map_w
    
    def save_state(self):
        print("\nSaving...")
        self.mapper.save(str(self.output_dir))
        self.scene_graph.save(str(self.output_dir / 'scene_graph.json'))
        print(f"Saved to {self.output_dir}")
    
    def run(self):
        print("\n" + "="*60)
        print("Interactive Explorer V2 - Real-time YOLO")
        print("="*60)
        print("WASD=Move | Click map=Navigate | R=Scan | S=Save | Q=Quit")
        print("="*60)
        
        print("\nInitial scan...")
        for _ in range(36):
            self.step('turn_left')
        print(f"Done: {self.mapper.get_statistics()['confirmed_objects']} objects")
        
        window = 'ObjectNav V2'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1100, 650)
        
        panel_info = {'h': 300, 'w': 400}
        
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and y >= panel_info['h']:
                if x >= panel_info['w']:
                    map_size = self.mapper.occupancy_map.map_size
                    scale = map_size / panel_info['h']
                    mx, my = int((x - panel_info['w']) * scale), int((y - panel_info['h']) * scale)
                    print(f"Navigate to ({mx}, {my})")
                    self.nav_path = self.plan_path(mx, my)
                    self.nav_index = 0
                    self.auto_mode = bool(self.nav_path)
                    if self.nav_path: print(f"Path: {len(self.nav_path)} pts")
        
        cv2.setMouseCallback(window, mouse_cb)
        
        rgb, depth, pos, rot, stats = self.step(None)
        
        while True:
            vis, h, w, _ = self.create_vis(rgb, depth, stats)
            panel_info['h'], panel_info['w'] = h, w
            cv2.imshow(window, vis)
            key = cv2.waitKey(30) & 0xFF
            
            action = None
            if key == ord('q') or key == 27: break
            elif key == ord('w') or key == 82: action = 'move_forward'; self.auto_mode = False
            elif key == ord('a') or key == 81: action = 'turn_left'; self.auto_mode = False
            elif key == ord('d') or key == 83: action = 'turn_right'; self.auto_mode = False
            elif key == ord('s'): self.save_state()
            elif key == ord('r'):
                for _ in range(36):
                    rgb, depth, pos, rot, stats = self.step('turn_left')
                    cv2.imshow(window, self.create_vis(rgb, depth, stats)[0])
                    cv2.waitKey(1)
            elif key == ord(' '): self.auto_mode = not self.auto_mode
            
            if self.auto_mode:
                action = self.get_auto_action(pos, rot)
            
            if action:
                rgb, depth, pos, rot, stats = self.step(action)
            
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1: break
        
        cv2.destroyAllWindows()
        self.save_state()
        print("\n" + self.scene_graph.visualize_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output-dir', default='./output_v2')
    parser.add_argument('--model-size', default='n', choices=['n','s','m','l','x'])
    parser.add_argument('--confidence', type=float, default=0.4)
    args = parser.parse_args()
    
    print("Loading...")
    sim = setup_habitat(args.scene)
    
    agent = sim.get_agent(0)
    if sim.pathfinder.is_loaded:
        pos = sim.pathfinder.get_random_navigable_point()
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)
    
    map_config = MapConfig()
    map_config.map_size_cm = 4800
    
    mapper = EnhancedSemanticMapperV2(
        map_config=map_config,
        model_size=args.model_size,
        confidence_threshold=args.confidence
    )
    scene_graph = SceneGraph()
    
    explorer = InteractiveExplorerV2(sim, mapper, scene_graph, args.output_dir)
    explorer.run()
    sim.close()


if __name__ == '__main__':
    main()