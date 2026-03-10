#!/usr/bin/env python3
"""
Map Editor with Room & Object Annotation
==========================================

Extends the original map editor with room boundary drawing and labeling
for known-environment navigation.

Features:
  - Load occupancy map (from exploration or saved maps)
  - Draw room boundaries (polygon mode)
  - Label rooms (bedroom, kitchen, etc.)
  - Mark objects with labels
  - Set room centers (navigation targets for "go to room X")
  - Export/Import annotation JSON
  - Compatible with InstructionParser + goal_to_action()

Controls:
  General:
    Q           - Quit
    Ctrl+S      - Save annotations
    Ctrl+Z      - Undo last action
    Tab         - Cycle through modes
    +/-         - Zoom in/out
    Arrow keys  - Pan map

  Mode: ROOM (default)
    Left click  - Add polygon vertex
    Enter       - Finish polygon (close room boundary)
    Escape      - Cancel current polygon
    R           - Set room type (cycles through types)
    C           - Click to set room center

  Mode: OBJECT
    Left click  - Place object marker
    O           - Set object type (cycles through types)

  Mode: EDIT
    Left click  - Select nearest room/object
    Delete      - Remove selected item
    M           - Move selected item (click new position)

  Mode: OBSTACLE
    Left click + drag  - Clear obstacles (brush)
    Right click + drag - Add obstacles (brush)

Usage:
    # Load from occupancy map image
    python map_editor_v2.py --image maps/floor_0.png

    # Load from saved map directory
    python map_editor_v2.py --map-dir output_enhanced/

    # Load existing annotations
    python map_editor_v2.py --image maps/floor_0.png --annotations rooms.json
"""

import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


# ================================================================
# Constants
# ================================================================

class Mode(Enum):
    ROOM = "ROOM"
    OBJECT = "OBJECT"
    EDIT = "EDIT"
    OBSTACLE = "OBSTACLE"

ROOM_TYPES = [
    'bedroom', 'bathroom', 'kitchen', 'living_room',
    'dining_room', 'hallway', 'office', 'garage',
    'laundry', 'closet', 'other',
]

ROOM_COLORS = {
    'bedroom':     (107, 107, 255),   # Red
    'bathroom':    (196, 205, 78),    # Teal
    'kitchen':     (109, 230, 255),   # Yellow
    'living_room': (211, 225, 149),   # Green
    'dining_room': (129, 129, 243),   # Pink
    'hallway':     (218, 150, 170),   # Purple
    'office':      (231, 92, 108),    # Blue
    'garage':      (234, 216, 168),   # Light blue
    'laundry':     (211, 186, 252),   # Pink
    'closet':      (221, 160, 221),   # Plum
    'other':       (136, 136, 136),   # Gray
}

OBJECT_TYPES = [
    'bed', 'chair', 'toilet', 'sofa', 'tv_monitor',
    'plant', 'table', 'refrigerator', 'sink', 'desk',
    'lamp', 'microwave', 'oven', 'custom',
]


# ================================================================
# Data Classes
# ================================================================

@dataclass
class RoomAnnotation:
    label: str                          # e.g. "kitchen_1"
    room_type: str                      # e.g. "kitchen"
    polygon: List[Tuple[int, int]]      # Boundary vertices
    center: Tuple[int, int] = None      # Navigation target (auto or manual)

    def compute_center(self):
        if self.polygon:
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            self.center = (int(np.mean(xs)), int(np.mean(ys)))

    def contains(self, x: int, y: int) -> bool:
        """Check if point is inside room polygon."""
        if len(self.polygon) < 3:
            return False
        pts = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


@dataclass
class ObjectAnnotation:
    label: str                          # e.g. "chair"
    position: Tuple[int, int]           # Map coordinates
    room: str = None                    # Parent room label


# ================================================================
# Map Editor
# ================================================================

class MapEditorV2:
    """Interactive map editor with room and object annotation."""

    WINDOW = "Map Editor - Room & Object Annotation"

    def __init__(self, map_image: np.ndarray, resolution_cm: float = 5.0):
        """
        Args:
            map_image: HxW or HxWx3 occupancy map image
            resolution_cm: Map resolution in cm/pixel
        """
        if map_image.ndim == 2:
            self.base_map = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
        else:
            self.base_map = map_image.copy()

        self.obstacle_map = None  # Will be set if editing obstacles
        self.resolution = resolution_cm
        self.h, self.w = self.base_map.shape[:2]

        # Annotations
        self.rooms: List[RoomAnnotation] = []
        self.objects: List[ObjectAnnotation] = []

        # UI state
        self.mode = Mode.ROOM
        self.room_type_idx = 0
        self.object_type_idx = 0
        self.current_polygon: List[Tuple[int, int]] = []
        self.selected_idx = -1          # Index of selected room/object
        self.selected_type = None       # 'room' or 'object'

        # Viewport
        self.zoom = 1.0
        self.pan_x, self.pan_y = 0, 0
        self.display_w, self.display_h = min(1200, self.w), min(800, self.h)

        # Brush (for obstacle mode)
        self.brush_size = 5
        self.drawing = False

        # Undo stack
        self._undo_stack: List = []

        # Mouse state
        self._mouse_pos = (0, 0)
        self._setting_center = False

    # ----------------------------------------------------------
    # Coordinate transforms
    # ----------------------------------------------------------

    def _screen_to_map(self, sx: int, sy: int) -> Tuple[int, int]:
        """Convert screen coords to map coords."""
        mx = int((sx - self.pan_x) / self.zoom)
        my = int((sy - self.pan_y) / self.zoom)
        return (max(0, min(mx, self.w - 1)), max(0, min(my, self.h - 1)))

    def _map_to_screen(self, mx: int, my: int) -> Tuple[int, int]:
        sx = int(mx * self.zoom + self.pan_x)
        sy = int(my * self.zoom + self.pan_y)
        return (sx, sy)

    # ----------------------------------------------------------
    # Drawing
    # ----------------------------------------------------------

    def render(self) -> np.ndarray:
        """Render the annotated map."""
        vis = self.base_map.copy()

        # Draw room polygons
        for i, room in enumerate(self.rooms):
            color = ROOM_COLORS.get(room.room_type, (136, 136, 136))
            pts = np.array(room.polygon, dtype=np.int32)

            if len(pts) >= 3:
                # Semi-transparent fill
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

                # Border
                is_selected = (self.selected_type == 'room' and self.selected_idx == i)
                border_color = (255, 255, 255) if is_selected else color
                thickness = 3 if is_selected else 1
                cv2.polylines(vis, [pts], True, border_color, thickness)

            # Room center
            if room.center:
                cx, cy = room.center
                cv2.drawMarker(vis, (cx, cy), (255, 255, 255),
                               cv2.MARKER_CROSS, 12, 2)
                # Label
                cv2.putText(vis, room.label, (cx - 30, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw current polygon in progress
        if self.current_polygon:
            color = ROOM_COLORS.get(ROOM_TYPES[self.room_type_idx], (255, 255, 255))
            for j in range(len(self.current_polygon)):
                cv2.circle(vis, self.current_polygon[j], 4, color, -1)
                if j > 0:
                    cv2.line(vis, self.current_polygon[j-1],
                             self.current_polygon[j], color, 2, cv2.LINE_AA)
            # Dashed line to mouse
            if len(self.current_polygon) > 0:
                last = self.current_polygon[-1]
                mx, my = self._mouse_pos
                for k in range(0, 100, 8):
                    t = k / 100
                    px = int(last[0] + t * (mx - last[0]))
                    py = int(last[1] + t * (my - last[1]))
                    cv2.circle(vis, (px, py), 1, color, -1)

        # Draw objects
        for i, obj in enumerate(self.objects):
            is_selected = (self.selected_type == 'object' and self.selected_idx == i)
            color = (0, 255, 255) if not is_selected else (255, 255, 0)
            cv2.circle(vis, obj.position, 6, color, -1)
            cv2.circle(vis, obj.position, 6, (0, 0, 0), 1)
            cv2.putText(vis, obj.label, (obj.position[0] + 8, obj.position[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Apply zoom and pan
        if self.zoom != 1.0 or self.pan_x != 0 or self.pan_y != 0:
            M = np.float32([
                [self.zoom, 0, self.pan_x],
                [0, self.zoom, self.pan_y],
            ])
            vis = cv2.warpAffine(vis, M, (self.display_w, self.display_h),
                                 borderValue=(20, 20, 30))
        else:
            vis = vis[:self.display_h, :self.display_w]

        # HUD
        self._draw_hud(vis)
        return vis

    def _draw_hud(self, vis):
        """Draw status bar and mode info."""
        h = vis.shape[0]
        # Background bar
        cv2.rectangle(vis, (0, h - 35), (vis.shape[1], h), (30, 30, 50), -1)

        # Mode
        mode_text = f"Mode: {self.mode.value}"
        cv2.putText(vis, mode_text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (78, 205, 196), 1)

        # Current type
        if self.mode == Mode.ROOM:
            rt = ROOM_TYPES[self.room_type_idx]
            color = ROOM_COLORS.get(rt, (200, 200, 200))
            info = f"Room: {rt}  |  Vertices: {len(self.current_polygon)}"
            cv2.putText(vis, info, (200, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        elif self.mode == Mode.OBJECT:
            ot = OBJECT_TYPES[self.object_type_idx]
            cv2.putText(vis, f"Object: {ot}", (200, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        elif self.mode == Mode.EDIT:
            sel = "None"
            if self.selected_type == 'room' and 0 <= self.selected_idx < len(self.rooms):
                sel = self.rooms[self.selected_idx].label
            elif self.selected_type == 'object' and 0 <= self.selected_idx < len(self.objects):
                sel = self.objects[self.selected_idx].label
            cv2.putText(vis, f"Selected: {sel}", (200, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # Stats
        stats = f"Rooms: {len(self.rooms)}  Objects: {len(self.objects)}  Zoom: {self.zoom:.1f}x"
        cv2.putText(vis, stats, (vis.shape[1] - 300, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # ----------------------------------------------------------
    # Mouse callback
    # ----------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        mx, my = self._screen_to_map(x, y)
        self._mouse_pos = (mx, my)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == Mode.ROOM:
                if self._setting_center:
                    # Set center for selected room
                    if self.selected_type == 'room' and 0 <= self.selected_idx < len(self.rooms):
                        self.rooms[self.selected_idx].center = (mx, my)
                    self._setting_center = False
                else:
                    self.current_polygon.append((mx, my))

            elif self.mode == Mode.OBJECT:
                ot = OBJECT_TYPES[self.object_type_idx]
                # Find which room this point is in
                parent_room = None
                for room in self.rooms:
                    if room.contains(mx, my):
                        parent_room = room.label
                        break
                self.objects.append(ObjectAnnotation(
                    label=ot, position=(mx, my), room=parent_room
                ))

            elif self.mode == Mode.EDIT:
                self._select_nearest(mx, my)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == Mode.OBSTACLE:
                self.drawing = True

        elif event == cv2.EVENT_RBUTTONUP:
            self.drawing = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.mode == Mode.OBSTACLE and self.obstacle_map is not None:
                cv2.circle(self.obstacle_map, (mx, my), self.brush_size, 1, -1)

    def _select_nearest(self, mx: int, my: int):
        """Select the nearest room or object to the click point."""
        best_dist = 30  # Max selection distance
        self.selected_idx = -1
        self.selected_type = None

        # Check objects
        for i, obj in enumerate(self.objects):
            d = np.sqrt((obj.position[0] - mx)**2 + (obj.position[1] - my)**2)
            if d < best_dist:
                best_dist = d
                self.selected_idx = i
                self.selected_type = 'object'

        # Check room centers
        for i, room in enumerate(self.rooms):
            if room.center:
                d = np.sqrt((room.center[0] - mx)**2 + (room.center[1] - my)**2)
                if d < best_dist:
                    best_dist = d
                    self.selected_idx = i
                    self.selected_type = 'room'

    # ----------------------------------------------------------
    # Actions
    # ----------------------------------------------------------

    def finish_polygon(self):
        """Complete current room polygon."""
        if len(self.current_polygon) < 3:
            print("Need at least 3 vertices")
            return

        rt = ROOM_TYPES[self.room_type_idx]
        count = sum(1 for r in self.rooms if r.room_type == rt) + 1
        label = f"{rt}_{count}"

        room = RoomAnnotation(
            label=label,
            room_type=rt,
            polygon=list(self.current_polygon),
        )
        room.compute_center()

        self._undo_stack.append(('add_room', len(self.rooms)))
        self.rooms.append(room)
        self.current_polygon.clear()
        self.selected_idx = len(self.rooms) - 1
        self.selected_type = 'room'
        print(f"  Added room: {label} ({len(room.polygon)} vertices)")

    def delete_selected(self):
        """Delete selected room or object."""
        if self.selected_type == 'room' and 0 <= self.selected_idx < len(self.rooms):
            removed = self.rooms.pop(self.selected_idx)
            self._undo_stack.append(('del_room', self.selected_idx, removed))
            print(f"  Deleted room: {removed.label}")
            self.selected_idx = -1
            self.selected_type = None
        elif self.selected_type == 'object' and 0 <= self.selected_idx < len(self.objects):
            removed = self.objects.pop(self.selected_idx)
            self._undo_stack.append(('del_object', self.selected_idx, removed))
            print(f"  Deleted object: {removed.label}")
            self.selected_idx = -1
            self.selected_type = None

    def undo(self):
        """Undo last action."""
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()
        if action[0] == 'add_room':
            self.rooms.pop(action[1])
        elif action[0] == 'del_room':
            self.rooms.insert(action[1], action[2])
        elif action[0] == 'add_object':
            self.objects.pop(action[1])
        elif action[0] == 'del_object':
            self.objects.insert(action[1], action[2])
        print("  Undo")

    # ----------------------------------------------------------
    # Export / Import
    # ----------------------------------------------------------

    def export_annotations(self, path: str):
        """Export annotations as JSON (compatible with InstructionParser)."""
        def to_native(v):
            """Convert numpy types to Python native for JSON."""
            if isinstance(v, (np.integer, np.int64, np.int32)):
                return int(v)
            if isinstance(v, (np.floating, np.float64, np.float32)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, tuple):
                return [to_native(x) for x in v]
            if isinstance(v, list):
                return [to_native(x) for x in v]
            return v

        data = {
            'map_size': [int(self.w), int(self.h)],
            'resolution_cm': float(self.resolution),
            'rooms': {},
            'objects': [],
            'metadata': {
                'tool': 'MapEditorV2',
                'n_rooms': len(self.rooms),
                'n_objects': len(self.objects),
            },
        }

        for room in self.rooms:
            center = None
            if room.center:
                center = [int(room.center[0]), int(room.center[1])]
            data['rooms'][room.label] = {
                'type': room.room_type,
                'center': center,
                'polygon': [[int(p[0]), int(p[1])] for p in room.polygon],
            }

        for obj in self.objects:
            data['objects'].append({
                'label': obj.label,
                'position': [int(obj.position[0]), int(obj.position[1])],
                'room': obj.room,
            })

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved annotations: {path} ({len(self.rooms)} rooms, {len(self.objects)} objects)")

    def import_annotations(self, path: str):
        """Import annotations from JSON."""
        with open(path) as f:
            data = json.load(f)

        self.rooms.clear()
        self.objects.clear()

        for label, rdata in data.get('rooms', {}).items():
            room = RoomAnnotation(
                label=label,
                room_type=rdata['type'],
                polygon=[tuple(p) for p in rdata['polygon']],
                center=tuple(rdata['center']) if rdata.get('center') else None,
            )
            if room.center is None:
                room.compute_center()
            self.rooms.append(room)

        for odata in data.get('objects', []):
            self.objects.append(ObjectAnnotation(
                label=odata['label'],
                position=tuple(odata['position']),
                room=odata.get('room'),
            ))

        print(f"  Loaded: {len(self.rooms)} rooms, {len(self.objects)} objects")

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------

    def run(self, save_path: str = 'map_annotations.json'):
        """Run the interactive editor."""
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW, self._mouse_callback)

        print("\n" + "=" * 50)
        print("  MAP EDITOR — Room & Object Annotation")
        print("=" * 50)
        print("  Tab: Switch mode  |  Q: Quit  |  Ctrl+S: Save")
        print("  ROOM mode:  Click vertices → Enter to close")
        print("  OBJECT mode: Click to place  |  O: change type")
        print("  EDIT mode:  Click to select  |  Delete to remove")
        print("  R/O: Cycle room/object type  |  C: Set room center")
        print("=" * 50 + "\n")

        while True:
            vis = self.render()
            cv2.imshow(self.WINDOW, vis)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break

            elif key == 9:  # Tab
                modes = list(Mode)
                idx = modes.index(self.mode)
                self.mode = modes[(idx + 1) % len(modes)]
                print(f"  Mode: {self.mode.value}")

            elif key == 13:  # Enter
                if self.mode == Mode.ROOM:
                    self.finish_polygon()

            elif key == 27:  # Escape
                self.current_polygon.clear()
                self._setting_center = False

            elif key == ord('r'):
                self.room_type_idx = (self.room_type_idx + 1) % len(ROOM_TYPES)
                print(f"  Room type: {ROOM_TYPES[self.room_type_idx]}")

            elif key == ord('o'):
                self.object_type_idx = (self.object_type_idx + 1) % len(OBJECT_TYPES)
                print(f"  Object type: {OBJECT_TYPES[self.object_type_idx]}")

            elif key == ord('c'):
                if self.selected_type == 'room':
                    self._setting_center = True
                    print("  Click to set room center...")

            elif key == 127 or key == 255 or key == 8:  # Delete / Backspace
                if self.mode == Mode.EDIT:
                    self.delete_selected()

            elif key == 26 or (key == ord('z') and False):  # Ctrl+Z
                self.undo()

            elif key == 19 or key == ord('s'):  # Ctrl+S or just 's'
                self.export_annotations(save_path)

            elif key == ord('+') or key == ord('='):
                self.zoom = min(5.0, self.zoom * 1.2)
            elif key == ord('-'):
                self.zoom = max(0.2, self.zoom / 1.2)

            # Arrow keys for pan
            elif key == 81 or key == 2:  # Left
                self.pan_x += 30
            elif key == 83 or key == 3:  # Right
                self.pan_x -= 30
            elif key == 82 or key == 0:  # Up
                self.pan_y += 30
            elif key == 84 or key == 1:  # Down
                self.pan_y -= 30

            if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

        # Auto-save if there are annotations
        if self.rooms or self.objects:
            self.export_annotations(save_path)
            print(f"Auto-saved on exit: {save_path}")
        else:
            print("No annotations to save.")


# ================================================================
# CLI
# ================================================================

def load_map_image(args) -> np.ndarray:
    """Load map from image file or map directory.
    
    Priority for --map-dir:
      1. semantic_map_with_trajectory.png (best for annotation - shows objects + path)
      2. semantic_map.png (shows objects)
      3. navigable_map.png (shows obstacles)
      4. occupancy_map.png
      5. Any .png
      6. navigable_map.pt / occupancy_map.pt (tensor files)
    """
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {args.image}")
        print(f"Loaded: {args.image}")
        return img

    if args.map_dir:
        map_dir = Path(args.map_dir)
        
        # Determine which floor to load
        floor = getattr(args, 'floor', None)
        
        # Find floor directories
        floor_dirs = sorted(map_dir.glob('floor_*'))
        
        if floor_dirs:
            if floor is not None:
                # Find matching floor
                target = f"floor_{floor}"
                search_dir = None
                for fd in floor_dirs:
                    if fd.name == target:
                        search_dir = fd
                        break
                if search_dir is None:
                    print(f"Floor {floor} not found. Available: {[d.name for d in floor_dirs]}")
                    search_dir = floor_dirs[0]
            else:
                # Show available floors, let user pick
                print(f"Available floors: {[d.name for d in floor_dirs]}")
                if len(floor_dirs) == 1:
                    search_dir = floor_dirs[0]
                else:
                    print(f"Using first floor: {floor_dirs[0].name}")
                    print(f"  (Use --floor N to select a different floor)")
                    search_dir = floor_dirs[0]
            print(f"Floor directory: {search_dir}")
        else:
            search_dir = map_dir
        
        # Priority order for image files
        priority_names = [
            'semantic_map_with_trajectory.png',
            'semantic_map.png',
            'navigable_map.png',
            'occupancy_map.png',
        ]
        
        for name in priority_names:
            path = search_dir / name
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    print(f"Loaded: {path}")
                    return img
        
        # Fallback: any PNG
        for png_path in sorted(search_dir.glob('*.png')):
            img = cv2.imread(str(png_path))
            if img is not None:
                print(f"Loaded (fallback): {png_path}")
                return img
        
        # Fallback: .pt tensor files
        import torch
        for name in ['navigable_map.pt', 'occupancy_map.pt']:
            pt_path = search_dir / name
            if pt_path.exists():
                tensor = torch.load(pt_path, map_location='cpu')
                arr = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
                if arr.ndim == 2:
                    vis = np.zeros((*arr.shape, 3), dtype=np.uint8)
                    vis[arr < 0.5] = [200, 200, 200]
                    vis[arr >= 0.5] = [60, 60, 60]
                    print(f"Loaded tensor: {pt_path}")
                    return vis

        raise FileNotFoundError(f"No map found in {search_dir}")

    raise ValueError("Need --image or --map-dir")


def main():
    parser = argparse.ArgumentParser(description='Map Editor with Room & Object Annotation')
    parser.add_argument('--image', type=str, help='Map image (PNG)')
    parser.add_argument('--map-dir', type=str, help='Directory with saved maps')
    parser.add_argument('--floor', type=int, default=None,
                        help='Floor number (e.g. 0, -1, 1). Auto-selects first if not set.')
    parser.add_argument('--annotations', type=str, help='Load existing annotations JSON')
    parser.add_argument('--save', type=str, default='map_annotations.json',
                        help='Output annotation file path')
    parser.add_argument('--resolution', type=float, default=5.0,
                        help='Map resolution in cm/pixel')
    args = parser.parse_args()

    map_image = load_map_image(args)
    print(f"Map loaded: {map_image.shape[1]}x{map_image.shape[0]}")

    editor = MapEditorV2(map_image, resolution_cm=args.resolution)

    if args.annotations and Path(args.annotations).exists():
        editor.import_annotations(args.annotations)

    editor.run(save_path=args.save)


if __name__ == '__main__':
    main()