#!/usr/bin/env python3
"""
Scene Graph Visualizer — Publication Quality
==============================================

Renders scene graph on semantic map for thesis figures.
Filters by floor, shows only high-confidence objects, clean layout.

Usage:
    python tools/visualize_scene_graph.py --map-dir my_map/ --floor -1
    python tools/visualize_scene_graph.py --map-dir my_map/ --floor 0 --output fig3_floor0.png
"""

import argparse
import numpy as np
import cv2
import json
from pathlib import Path


COLORS = {
    'chair': (0, 165, 255), 'couch': (30, 144, 255), 'sofa': (30, 144, 255),
    'dining table': (50, 100, 180), 'table': (50, 100, 180),
    'bed': (180, 50, 230), 'toilet': (0, 220, 220), 'sink': (0, 180, 180),
    'tv': (50, 220, 50), 'tv_monitor': (50, 220, 50),
    'refrigerator': (180, 180, 180), 'microwave': (150, 150, 150),
    'oven': (130, 130, 130), 'potted plant': (0, 180, 80), 'plant': (0, 180, 80),
    'book': (60, 20, 200), 'clock': (220, 30, 30), 'laptop': (220, 150, 30),
    'bench': (180, 180, 60), 'vase': (200, 100, 200),
}

ROOM_COLORS = {
    'bedroom': (107, 107, 255), 'bathroom': (196, 205, 78),
    'kitchen': (109, 230, 255), 'living_room': (211, 225, 149),
    'dining_room': (129, 129, 243), 'hallway': (218, 150, 170),
    'office': (231, 92, 108), 'laundry': (211, 186, 252),
}


def world_to_map(wx, wz, origin_x, origin_z, resolution, map_size):
    mx = int((wx - origin_x) * 100 / resolution + map_size // 2)
    my = int((wz - origin_z) * 100 / resolution + map_size // 2)
    return (np.clip(mx, 0, map_size - 1), np.clip(my, 0, map_size - 1))


def filter_nodes_by_floor(nodes, floor_id, floor_heights):
    if not floor_heights or floor_id is None:
        return nodes
    target_base = floor_heights.get(str(floor_id), floor_heights.get(floor_id))
    if target_base is None:
        return nodes
    return [n for n in nodes if target_base - 0.5 <= n['position'][1] <= target_base + 2.5]


def render(map_img, nodes, edges, annotations, origin_x, origin_z,
           resolution, map_size, min_obs=3, title=None):
    vis = map_img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    vis = (vis.astype(np.float32) * 0.85).astype(np.uint8)

    # Room annotations
    if annotations:
        for label, rdata in annotations.get('rooms', {}).items():
            color = ROOM_COLORS.get(rdata.get('type', ''), (136, 136, 136))
            pts = np.array(rdata['polygon'], dtype=np.int32)
            if len(pts) >= 3:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)
                cv2.polylines(vis, [pts], True, color, 1, cv2.LINE_AA)
            if rdata.get('center'):
                cx, cy = int(rdata['center'][0]), int(rdata['center'][1])
                cv2.putText(vis, rdata.get('type', label).replace('_', ' '),
                            (cx - 20, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, color, 1, cv2.LINE_AA)

    valid = [n for n in nodes if n.get('observations', 0) >= min_obs]

    pos_map = {}
    for n in valid:
        p = n['position']
        mx, my = world_to_map(p[0], p[2], origin_x, origin_z, resolution, map_size)
        if 0 <= mx < w and 0 <= my < h:
            pos_map[n['node_id']] = (mx, my)

    # Edges (only 'near', skip colocated to reduce clutter)
    for edge in edges:
        src = edge.get('source_id', edge.get('source', edge.get('src')))
        tgt = edge.get('target_id', edge.get('target', edge.get('dst')))
        if src is None or tgt is None:
            continue
        if src not in pos_map or tgt not in pos_map:
            continue
        rel = edge.get('relationship', edge.get('relation', ''))
        if rel == 'near':
            cv2.line(vis, pos_map[src], pos_map[tgt], (80, 180, 80), 1, cv2.LINE_AA)

    # Nodes + labels (avoid label overlap)
    label_positions = []
    for n in valid:
        nid = n['node_id']
        if nid not in pos_map:
            continue
        mx, my = pos_map[nid]
        cls = n['class_name'].lower()
        color = COLORS.get(cls, (200, 200, 200))
        obs = n.get('observations', 1)
        radius = min(3 + obs // 5, 8)

        cv2.circle(vis, (mx, my), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (mx, my), radius, color, -1, cv2.LINE_AA)

        lx, ly = mx + radius + 4, my + 3
        too_close = any(abs(lx - px) < 50 and abs(ly - py) < 12 for px, py in label_positions)
        if not too_close:
            cv2.putText(vis, cls, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.28, (240, 240, 240), 1, cv2.LINE_AA)
            label_positions.append((lx, ly))

    # Legend
    seen = []
    for n in valid:
        c = n['class_name'].lower()
        if c not in seen: seen.append(c)

    lg_w, lg_h = 110, len(seen) * 14 + 30
    lg_x, lg_y = w - lg_w - 8, h - lg_h - 8
    overlay = vis.copy()
    cv2.rectangle(overlay, (lg_x, lg_y), (w - 8, h - 8), (25, 25, 40), -1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
    cv2.putText(vis, "Scene Graph", (lg_x + 5, lg_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
    for i, cls in enumerate(seen):
        y = lg_y + 28 + i * 14
        cv2.circle(vis, (lg_x + 10, y), 4, COLORS.get(cls, (200, 200, 200)), -1, cv2.LINE_AA)
        cnt = sum(1 for n in valid if n['class_name'].lower() == cls)
        cv2.putText(vis, f"{cls} ({cnt})", (lg_x + 20, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (180, 180, 180), 1, cv2.LINE_AA)
    ey = h - 12
    cv2.line(vis, (lg_x + 5, ey), (lg_x + 18, ey), (80, 180, 80), 1, cv2.LINE_AA)
    cv2.putText(vis, "near (<2m)", (lg_x + 22, ey + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.22, (140, 140, 140), 1, cv2.LINE_AA)

    if title:
        cv2.rectangle(vis, (0, 0), (w, 22), (25, 25, 40), -1)
        cv2.putText(vis, title, (8, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)

    n_vis_edges = sum(1 for e in edges
                      if e.get('relationship', e.get('relation', '')) == 'near'
                      and e.get('source_id', e.get('source')) in pos_map
                      and e.get('target_id', e.get('target')) in pos_map)
    cv2.putText(vis, f"{len(valid)} objects | {n_vis_edges} relationships",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser(description='Scene Graph Visualizer')
    parser.add_argument('--scene-graph', type=str)
    parser.add_argument('--map-image', type=str)
    parser.add_argument('--map-dir', type=str)
    parser.add_argument('--floor', type=int, default=None)
    parser.add_argument('--statistics', type=str)
    parser.add_argument('--annotations', type=str)
    parser.add_argument('--output', type=str, default='scene_graph_vis.png')
    parser.add_argument('--min-obs', type=int, default=3)
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()

    if args.map_dir:
        md = Path(args.map_dir)
        if not args.scene_graph:
            p = md / 'scene_graph.json'
            if p.exists(): args.scene_graph = str(p)
        if not args.statistics:
            p = md / 'statistics.json'
            if p.exists(): args.statistics = str(p)

        # Find floor map image
        if not args.map_image:
            floor_dirs = sorted(md.glob('floor_*'))
            if args.floor is not None:
                target = f'floor_{args.floor}'
                fd = next((d for d in floor_dirs if d.name == target), None)
            else:
                fd = floor_dirs[0] if floor_dirs else None
                if fd:
                    args.floor = int(fd.name.split('_')[1])

            if fd:
                for name in ['semantic_map.png', 'semantic_map_with_trajectory.png']:
                    p = fd / name
                    if p.exists():
                        args.map_image = str(p)
                        break
                if not args.annotations:
                    p = fd / 'map_annotations.json'
                    if p.exists(): args.annotations = str(p)

    if not args.scene_graph:
        print("Need --scene-graph or --map-dir"); return

    with open(args.scene_graph) as f:
        sg = json.load(f)
    nodes = sg.get('nodes', [])
    edges = sg.get('edges', [])
    print(f"Loaded: {len(nodes)} nodes, {len(edges)} edges")

    origin_x, origin_z, resolution, map_size = 0, 0, 5.0, 480
    floor_heights = {}
    if args.statistics:
        with open(args.statistics) as f:
            stats = json.load(f)
        origin_x = stats.get('origin_x', 0)
        origin_z = stats.get('origin_z', 0)
        resolution = stats.get('resolution_cm', 5.0)
        map_size = stats.get('map_size', 480)

    if args.map_dir:
        fi = Path(args.map_dir) / 'floor_info.json'
        if fi.exists():
            with open(fi) as f:
                floor_heights = json.load(f).get('base_heights', {})

    if args.floor is not None and floor_heights:
        before = len(nodes)
        nodes = filter_nodes_by_floor(nodes, args.floor, floor_heights)
        visible_ids = {n['node_id'] for n in nodes}
        edges = [e for e in edges
                 if e.get('source_id', e.get('source')) in visible_ids
                 and e.get('target_id', e.get('target')) in visible_ids]
        print(f"Floor {args.floor}: {len(nodes)}/{before} nodes, {len(edges)} edges")

    map_img = cv2.imread(args.map_image) if args.map_image else np.full((map_size, map_size, 3), 40, np.uint8)
    annotations = None
    if args.annotations:
        with open(args.annotations) as f:
            annotations = json.load(f)

    floor_str = f" (Floor {args.floor})" if args.floor is not None else ""
    title = args.title or f"Online Scene Graph{floor_str}"
    vis = render(map_img, nodes, edges, annotations, origin_x, origin_z,
                 resolution, map_size, min_obs=args.min_obs, title=title)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, vis)
    print(f"Saved: {args.output}")
    cv2.imshow("Scene Graph", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()