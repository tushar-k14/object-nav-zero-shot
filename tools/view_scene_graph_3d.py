import json
import argparse
import numpy as np
import open3d as o3d
from matplotlib import cm


REL_COLORS = {
    "near":[0.3,0.9,0.3],
    "next_to":[1.0,0.6,0.2],
    "on":[1.0,0.3,0.3],
    "above":[0.7,0.4,1.0]
}


def make_sphere(pos, color, r=0.15):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    mesh.compute_vertex_normals()
    mesh.translate(pos)
    mesh.paint_uniform_color(color)
    return mesh


def make_line(p1, p2, color):

    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p1,p2]),
        lines=o3d.utility.Vector2iVector([[0,1]])
    )

    line.colors = o3d.utility.Vector3dVector([color])
    return line


def make_floor(y):

    floor = o3d.geometry.TriangleMesh.create_box(20,0.02,20)
    floor.translate([-10,y,-10])
    floor.paint_uniform_color([0.5,0.5,0.5])
    return floor


def detect_floor_levels(nodes):

    ys = sorted([n["position"][1] for n in nodes])

    floors = []
    cur = [ys[0]]

    for y in ys[1:]:

        if abs(y-cur[-1]) < 1.2:
            cur.append(y)
        else:
            floors.append(np.mean(cur))
            cur=[y]

    floors.append(np.mean(cur))

    return floors


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-graph")
    parser.add_argument("--target",default=None)
    args = parser.parse_args()

    with open(args.scene_graph) as f:
        sg = json.load(f)

    nodes = sg["nodes"]
    edges = sg["edges"]
    rooms = sg.get("room_assignments",{})

    node_pos={}
    geometries=[]

    # room colors
    room_names=list(set(rooms.values()))
    cmap=cm.get_cmap("tab20",len(room_names))

    room_colors={
        r:list(cmap(i)[:3]) for i,r in enumerate(room_names)
    }

    # nodes
    for n in nodes:

        nid=n["node_id"]
        pos=np.array(n["position"])

        node_pos[nid]=pos

        room=rooms.get(str(nid),"unknown")
        color=room_colors.get(room,[0.7,0.7,0.7])

        if args.target and n["class_name"]==args.target:
            color=[1,0,0]

        sphere=make_sphere(pos,color)
        geometries.append(sphere)

    # edges
    for e in edges:

        s=e["source_id"]
        t=e["target_id"]
        rel=e["relationship"]

        if s not in node_pos or t not in node_pos:
            continue

        p1=node_pos[s]
        p2=node_pos[t]

        color=REL_COLORS.get(rel,[0.7,0.7,0.7])

        geometries.append(make_line(p1,p2,color))

    # floor planes
    floors=detect_floor_levels(nodes)

    for y in floors:
        geometries.append(make_floor(y-0.1))

    o3d.visualization.draw_geometries(geometries)


if __name__=="__main__":
    main()