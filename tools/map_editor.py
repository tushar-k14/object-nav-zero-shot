#!/usr/bin/env python3
"""
Map Editor - Mark Stairs and Traversable Zones
===============================================

Interactive tool to fix navigable maps by:
1. Marking stair regions as traversable
2. Creating floor transition zones
3. Clearing obstacles that shouldn't be there

Controls:
- Left click + drag: Mark area as TRAVERSABLE (clear obstacles)
- Right click + drag: Mark area as OBSTACLE (add obstacles)  
- S: Mark current selection as STAIRS (floor transition)
- 1/2/3...: Switch between floors
- +/-: Adjust brush size
- Z: Undo last action
- Ctrl+S: Save corrected maps
- Q: Quit

Usage:
    python map_editor.py --map-dir ./output_enhanced
"""

import argparse
import numpy as np
import cv2
import torch
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import copy


@dataclass
class StairZone:
    """A stair/floor transition zone."""
    map_pos: Tuple[int, int]  # Center position
    radius: int               # Size in pixels
    floor_from: int
    floor_to: int
    

@dataclass  
class EditAction:
    """For undo functionality."""
    floor_id: int
    old_navigable: np.ndarray
    old_obstacle: np.ndarray


class MapEditor:
    def __init__(self, map_dir: str):
        self.map_dir = Path(map_dir)
        
        # Load floor info
        floor_info_path = self.map_dir / 'floor_info.json'
        if floor_info_path.exists():
            with open(floor_info_path) as f:
                self.floor_info = json.load(f)
            self.floor_ids = self.floor_info.get('floor_ids', [0])
        else:
            # Auto-detect floors
            self.floor_ids = []
            for d in self.map_dir.iterdir():
                if d.is_dir() and d.name.startswith('floor_'):
                    try:
                        fid = int(d.name.split('_')[1])
                        self.floor_ids.append(fid)
                    except:
                        pass
            self.floor_ids = sorted(self.floor_ids) if self.floor_ids else [0]
            self.floor_info = {'floor_ids': self.floor_ids}
        
        print(f"Found floors: {self.floor_ids}")
        
        # Load maps for each floor
        self.navigable_maps = {}
        self.obstacle_maps = {}
        self.explored_maps = {}
        self.semantic_maps = {}
        
        for fid in self.floor_ids:
            floor_dir = self.map_dir / f'floor_{fid}'
            if floor_dir.exists():
                # Load navigable map
                nav_path = floor_dir / 'navigable_map.pt'
                if nav_path.exists():
                    self.navigable_maps[fid] = torch.load(nav_path).numpy()
                else:
                    # Fall back to occupancy map
                    occ_path = floor_dir / 'occupancy_map.pt'
                    if occ_path.exists():
                        self.navigable_maps[fid] = torch.load(occ_path).numpy()
                
                # Load obstacle map
                occ_path = floor_dir / 'occupancy_map.pt'
                if occ_path.exists():
                    self.obstacle_maps[fid] = torch.load(occ_path).numpy()
                
                # Load explored map
                exp_path = floor_dir / 'explored_map.pt'
                if exp_path.exists():
                    self.explored_maps[fid] = torch.load(exp_path).numpy()
                    
                print(f"  Floor {fid}: loaded maps")
        
        if not self.navigable_maps:
            raise ValueError(f"No maps found in {map_dir}")
        
        # Current state
        self.current_floor = self.floor_ids[0]
        self.brush_size = 15
        self.drawing = False
        self.draw_mode = 'clear'  # 'clear' or 'obstacle'
        self.last_pos = None
        
        # Stair zones
        self.stair_zones: List[StairZone] = []
        self._load_stairs()
        
        # Undo stack
        self.undo_stack: List[EditAction] = []
        self.max_undo = 20
        
        # Selection for stair marking
        self.selection_start = None
        self.selection_end = None
        
    def _load_stairs(self):
        """Load existing stair data."""
        stairs_path = self.map_dir / 'stairs.json'
        if stairs_path.exists():
            with open(stairs_path) as f:
                stairs_data = json.load(f)
            for s in stairs_data:
                self.stair_zones.append(StairZone(
                    map_pos=tuple(s.get('map_pos', (0, 0))),
                    radius=s.get('radius', 15),
                    floor_from=s.get('floor', 0),
                    floor_to=s.get('floor_to', s.get('floor', 0) + 1)
                ))
            print(f"Loaded {len(self.stair_zones)} stair zones")
    
    def _save_undo(self):
        """Save current state for undo."""
        fid = self.current_floor
        if fid in self.navigable_maps and fid in self.obstacle_maps:
            action = EditAction(
                floor_id=fid,
                old_navigable=self.navigable_maps[fid].copy(),
                old_obstacle=self.obstacle_maps[fid].copy()
            )
            self.undo_stack.append(action)
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)
    
    def undo(self):
        """Undo last action."""
        if not self.undo_stack:
            print("Nothing to undo")
            return
        
        action = self.undo_stack.pop()
        self.navigable_maps[action.floor_id] = action.old_navigable
        self.obstacle_maps[action.floor_id] = action.old_obstacle
        print(f"Undone action on floor {action.floor_id}")
    
    def clear_area(self, x: int, y: int):
        """Clear obstacles in area (make traversable)."""
        fid = self.current_floor
        if fid not in self.navigable_maps:
            return
        
        nav_map = self.navigable_maps[fid]
        obs_map = self.obstacle_maps.get(fid)
        
        h, w = nav_map.shape
        r = self.brush_size
        
        y1, y2 = max(0, y - r), min(h, y + r)
        x1, x2 = max(0, x - r), min(w, x + r)
        
        # Create circular mask
        yy, xx = np.ogrid[y1-y:y2-y, x1-x:x2-x]
        mask = xx*xx + yy*yy <= r*r
        
        # Clear in navigable map
        region = nav_map[y1:y2, x1:x2]
        region[mask] = 0
        
        # Also clear in obstacle map
        if obs_map is not None:
            region_obs = obs_map[y1:y2, x1:x2]
            region_obs[mask] = 0
    
    def add_obstacle(self, x: int, y: int):
        """Add obstacles in area."""
        fid = self.current_floor
        if fid not in self.navigable_maps:
            return
        
        nav_map = self.navigable_maps[fid]
        obs_map = self.obstacle_maps.get(fid)
        
        h, w = nav_map.shape
        r = self.brush_size
        
        y1, y2 = max(0, y - r), min(h, y + r)
        x1, x2 = max(0, x - r), min(w, x + r)
        
        yy, xx = np.ogrid[y1-y:y2-y, x1-x:x2-x]
        mask = xx*xx + yy*yy <= r*r
        
        region = nav_map[y1:y2, x1:x2]
        region[mask] = 1
        
        if obs_map is not None:
            region_obs = obs_map[y1:y2, x1:x2]
            region_obs[mask] = 1
    
    def mark_stairs(self, x1: int, y1: int, x2: int, y2: int):
        """Mark a rectangular region as stairs."""
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max(abs(x2 - x1), abs(y2 - y1)) // 2
        
        # Determine floor connection
        fid = self.current_floor
        floor_to = fid + 1 if fid + 1 in self.floor_ids else fid - 1
        
        stair = StairZone(
            map_pos=(cx, cy),
            radius=radius,
            floor_from=fid,
            floor_to=floor_to
        )
        self.stair_zones.append(stair)
        
        # Clear the stair region on BOTH connected floors
        self._save_undo()
        
        for floor_id in [fid, floor_to]:
            if floor_id in self.navigable_maps:
                nav_map = self.navigable_maps[floor_id]
                obs_map = self.obstacle_maps.get(floor_id)
                
                h, w = nav_map.shape
                cy_c = min(max(cy, 0), h-1)
                cx_c = min(max(cx, 0), w-1)
                
                y1_c, y2_c = max(0, cy - radius), min(h, cy + radius)
                x1_c, x2_c = max(0, cx - radius), min(w, cx + radius)
                
                nav_map[y1_c:y2_c, x1_c:x2_c] = 0
                if obs_map is not None:
                    obs_map[y1_c:y2_c, x1_c:x2_c] = 0
        
        print(f"Marked stairs at ({cx}, {cy}) connecting floor {fid} <-> {floor_to}")
    
    def visualize(self) -> np.ndarray:
        """Create visualization of current floor."""
        fid = self.current_floor
        
        if fid not in self.navigable_maps:
            return np.zeros((500, 500, 3), dtype=np.uint8)
        
        nav_map = self.navigable_maps[fid]
        exp_map = self.explored_maps.get(fid, np.ones_like(nav_map))
        
        h, w = nav_map.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background
        vis[:, :] = [40, 40, 40]
        
        # Explored areas
        vis[exp_map > 0] = [150, 150, 150]
        
        # Obstacles (from navigable map)
        vis[nav_map > 0.5] = [80, 80, 80]
        
        # Stair zones (green)
        for stair in self.stair_zones:
            if stair.floor_from == fid or stair.floor_to == fid:
                sx, sy = stair.map_pos
                r = stair.radius
                cv2.rectangle(vis, (sx - r, sy - r), (sx + r, sy + r), (0, 255, 0), 2)
                cv2.putText(vis, f"S{stair.floor_from}<>{stair.floor_to}", 
                           (sx - r, sy - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Current selection
        if self.selection_start and self.selection_end:
            cv2.rectangle(vis, self.selection_start, self.selection_end, (255, 255, 0), 1)
        
        # Brush preview
        if self.last_pos:
            color = (0, 255, 255) if self.draw_mode == 'clear' else (255, 0, 0)
            cv2.circle(vis, self.last_pos, self.brush_size, color, 1)
        
        # Add info panel
        vis = self._add_info_panel(vis)
        
        return vis
    
    def _add_info_panel(self, vis: np.ndarray) -> np.ndarray:
        """Add info panel on the side."""
        panel_w = 200
        panel = np.zeros((vis.shape[0], panel_w, 3), dtype=np.uint8)
        panel[:, :] = [30, 30, 30]
        
        lines = [
            "MAP EDITOR",
            "",
            f"Floor: {self.current_floor}",
            f"Brush: {self.brush_size}px",
            f"Mode: {self.draw_mode.upper()}",
            f"Stairs: {len(self.stair_zones)}",
            "",
            "CONTROLS:",
            "Left drag: Clear",
            "Right drag: Add obs",
            "Shift+drag: Select",
            "S: Mark as stairs",
            "+/-: Brush size",
            "1-9: Switch floor",
            "Z: Undo",
            "Ctrl+S: Save",
            "Q: Quit",
        ]
        
        y = 20
        for line in lines:
            color = (255, 255, 255) if line in ["MAP EDITOR", "CONTROLS:"] else (200, 200, 200)
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y += 18
        
        # Legend
        y += 10
        cv2.rectangle(panel, (10, y), (25, y + 12), [150, 150, 150], -1)
        cv2.putText(panel, "Explored", (30, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        y += 18
        
        cv2.rectangle(panel, (10, y), (25, y + 12), [80, 80, 80], -1)
        cv2.putText(panel, "Obstacle", (30, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        y += 18
        
        cv2.rectangle(panel, (10, y), (25, y + 12), [0, 255, 0], -1)
        cv2.putText(panel, "Stairs", (30, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        return np.hstack([vis, panel])
    
    def save(self):
        """Save corrected maps."""
        print("\nSaving corrected maps...")
        
        for fid in self.floor_ids:
            floor_dir = self.map_dir / f'floor_{fid}'
            if not floor_dir.exists():
                continue
            
            if fid in self.navigable_maps:
                # Save navigable map
                nav_tensor = torch.from_numpy(self.navigable_maps[fid]).float()
                torch.save(nav_tensor, floor_dir / 'navigable_map.pt')
                
                # Save PNG
                nav_vis = (self.navigable_maps[fid] * 255).astype(np.uint8)
                cv2.imwrite(str(floor_dir / 'navigable_map.png'), nav_vis)
                
                print(f"  Floor {fid}: saved navigable_map")
            
            if fid in self.obstacle_maps:
                # Save obstacle map
                obs_tensor = torch.from_numpy(self.obstacle_maps[fid]).float()
                torch.save(obs_tensor, floor_dir / 'occupancy_map.pt')
                
                obs_vis = (self.obstacle_maps[fid] * 255).astype(np.uint8)
                cv2.imwrite(str(floor_dir / 'occupancy_map.png'), obs_vis)
        
        # Save stair zones
        stairs_data = []
        for stair in self.stair_zones:
            stairs_data.append({
                'map_pos': list(stair.map_pos),
                'radius': stair.radius,
                'floor': stair.floor_from,
                'floor_to': stair.floor_to,
                'position': [0, 0, 0]  # World pos placeholder
            })
        
        with open(self.map_dir / 'stairs.json', 'w') as f:
            json.dump(stairs_data, f, indent=2)
        
        print(f"Saved {len(self.stair_zones)} stair zones")
        print(f"Done! Maps saved to {self.map_dir}")
    
    def run(self):
        """Run interactive editor."""
        print("\n" + "="*50)
        print("Map Editor - Mark Stairs and Traversable Zones")
        print("="*50)
        print("Left-drag: Clear obstacles (make traversable)")
        print("Right-drag: Add obstacles")
        print("Shift+drag: Select region, then S to mark as stairs")
        print("+/-: Adjust brush size")
        print("1-9: Switch floors")
        print("Z: Undo | Ctrl+S: Save | Q: Quit")
        print("="*50)
        
        window = 'Map Editor'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 900, 700)
        
        shift_pressed = False
        ctrl_pressed = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal shift_pressed
            
            # Adjust for panel
            fid = self.current_floor
            if fid not in self.navigable_maps:
                return
            
            h, w = self.navigable_maps[fid].shape
            if x >= w:  # Click on panel
                return
            
            self.last_pos = (x, y)
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    # Start selection for stairs
                    self.selection_start = (x, y)
                    self.selection_end = (x, y)
                else:
                    self._save_undo()
                    self.drawing = True
                    self.draw_mode = 'clear'
                    self.clear_area(x, y)
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._save_undo()
                self.drawing = True
                self.draw_mode = 'obstacle'
                self.add_obstacle(x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.selection_start and (flags & cv2.EVENT_FLAG_SHIFTKEY):
                    self.selection_end = (x, y)
                elif self.drawing:
                    if self.draw_mode == 'clear':
                        self.clear_area(x, y)
                    else:
                        self.add_obstacle(x, y)
            
            elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                self.drawing = False
        
        cv2.setMouseCallback(window, mouse_callback)
        
        while True:
            vis = self.visualize()
            cv2.imshow(window, vis)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            
            elif key == ord('s'):
                if self.selection_start and self.selection_end:
                    self.mark_stairs(
                        self.selection_start[0], self.selection_start[1],
                        self.selection_end[0], self.selection_end[1]
                    )
                    self.selection_start = None
                    self.selection_end = None
            
            elif key == ord('z'):
                self.undo()
            
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(50, self.brush_size + 2)
                print(f"Brush size: {self.brush_size}")
            
            elif key == ord('-'):
                self.brush_size = max(3, self.brush_size - 2)
                print(f"Brush size: {self.brush_size}")
            
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(self.floor_ids):
                    self.current_floor = self.floor_ids[idx]
                    print(f"Switched to floor {self.current_floor}")
            
            elif key == 19:  # Ctrl+S
                self.save()
            
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
        
        # Ask to save
        print("\nSave changes before exit? (y/n): ", end='')
        try:
            if input().strip().lower() == 'y':
                self.save()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description='Map Editor')
    parser.add_argument('--map-dir', required=True, help='Directory with saved maps')
    args = parser.parse_args()
    
    editor = MapEditor(args.map_dir)
    editor.run()


if __name__ == '__main__':
    main()
