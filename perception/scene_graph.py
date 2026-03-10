"""
Lightweight Scene Graph for ObjectNav
======================================

Builds a scene graph from tracked objects (detected by YOLO) and
room classifications (from VLM). This gives the navigation policy
structured context about the environment beyond raw pixel observations.

Graph structure:
    Nodes: Objects (from YOLO tracker) + Rooms (from VLM classifier)
    Edges: spatial relationships (near, in_room, colocated)

Used to:
    1. Provide structured context to VLM queries (instead of just an image)
    2. Improve frontier scoring based on scene structure
    3. Enable "I've seen a couch, so the TV is probably nearby" reasoning

This is NOT a full 3D scene graph (like ConceptGraph or SG-Nav).
It's a lightweight, online-built structure optimized for real-time
navigation decisions.

Usage:
    from perception.scene_graph import SceneGraph

    sg = SceneGraph()
    sg.update_objects(mapper.get_confirmed_objects(), occ_map)
    sg.update_room('living_room', agent_map_pos)

    # Get context string for VLM
    context = sg.describe()

    # Get frontier boost based on scene structure
    boost = sg.score_frontier_by_context(frontier, target_object)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


# Spatial relationship thresholds (in map pixels, 5cm resolution)
NEAR_THRESHOLD = 40       # 2m
COLOCATED_THRESHOLD = 80  # 4m
ROOM_RADIUS = 60          # 3m — objects within this radius of agent when
                          #       room was classified belong to that room

# Object-room priors (probability of object appearing in each room)
OBJECT_ROOM_PRIORS = {
    'bed':          {'bedroom': 0.95, 'office': 0.05},
    'toilet':       {'bathroom': 0.98},
    'sofa':         {'living_room': 0.80, 'office': 0.10, 'bedroom': 0.10},
    'couch':        {'living_room': 0.80, 'office': 0.10, 'bedroom': 0.10},
    'tv':           {'living_room': 0.70, 'bedroom': 0.25, 'office': 0.05},
    'tv_monitor':   {'living_room': 0.70, 'bedroom': 0.25, 'office': 0.05},
    'chair':        {'dining_room': 0.35, 'office': 0.25, 'living_room': 0.20, 'bedroom': 0.20},
    'plant':        {'living_room': 0.40, 'hallway': 0.30, 'dining_room': 0.20, 'office': 0.10},
    'sink':         {'kitchen': 0.50, 'bathroom': 0.45, 'laundry': 0.05},
    'refrigerator': {'kitchen': 0.98},
    'oven':         {'kitchen': 0.98},
    'microwave':    {'kitchen': 0.95},
    'dining table': {'dining_room': 0.70, 'kitchen': 0.25},
    'book':         {'office': 0.30, 'living_room': 0.30, 'bedroom': 0.30},
    'clock':        {'living_room': 0.30, 'bedroom': 0.30, 'kitchen': 0.20, 'office': 0.20},
    'vase':         {'living_room': 0.50, 'dining_room': 0.30, 'hallway': 0.20},
    'laptop':       {'office': 0.50, 'bedroom': 0.25, 'living_room': 0.25},
}


@dataclass
class SGNode:
    """A node in the scene graph."""
    node_id: str
    node_type: str          # 'object' or 'room'
    label: str              # e.g. 'chair', 'bedroom'
    position: Tuple[int, int]  # map coords (mx, my)
    confidence: float = 1.0
    observations: int = 1


@dataclass
class SGEdge:
    """An edge between two scene graph nodes."""
    source: str
    target: str
    relation: str           # 'near', 'in_room', 'colocated'
    distance: float = 0.0


class SceneGraph:
    """
    Online scene graph built from YOLO detections + VLM room labels.
    """

    def __init__(self):
        self.nodes: Dict[str, SGNode] = {}
        self.edges: List[SGEdge] = []
        self.room_assignments: Dict[str, str] = {}  # object_id -> room_id
        self._room_counter = 0

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self.room_assignments.clear()
        self._room_counter = 0

    # ----------------------------------------------------------
    # Update from detections
    # ----------------------------------------------------------

    def update_objects(self, tracked_objects: dict, occ_map) -> None:
        """
        Update graph with current tracked objects from the mapper.

        Args:
            tracked_objects: dict from mapper.get_confirmed_objects()
            occ_map: FixedOccupancyMap for coordinate conversion
        """
        for obj_id, obj in tracked_objects.items():
            node_id = f"obj_{obj_id}"
            pos = obj.mean_position
            mx, my = occ_map._world_to_map(pos[0], pos[2])

            self.nodes[node_id] = SGNode(
                node_id=node_id,
                node_type='object',
                label=obj.class_name.lower(),
                position=(mx, my),
                confidence=getattr(obj, 'mean_confidence', 0.5),
                observations=getattr(obj, 'observations',
                                     getattr(obj, 'num_observations', 1)),
            )

        # Rebuild edges
        self._rebuild_edges()

    def update_room(
        self,
        room_type: str,
        agent_pos: Tuple[int, int],
    ) -> str:
        """
        Add/update a room node at the agent's current position.
        Assigns nearby objects to this room.

        Returns room node_id.
        """
        # Check if there's already a room of this type nearby
        for nid, node in self.nodes.items():
            if (node.node_type == 'room' and node.label == room_type):
                dist = np.sqrt(
                    (node.position[0] - agent_pos[0]) ** 2 +
                    (node.position[1] - agent_pos[1]) ** 2
                )
                if dist < ROOM_RADIUS:
                    # Update existing room position (running average)
                    node.observations += 1
                    node.position = (
                        int((node.position[0] * (node.observations - 1) + agent_pos[0]) / node.observations),
                        int((node.position[1] * (node.observations - 1) + agent_pos[1]) / node.observations),
                    )
                    self._assign_objects_to_room(nid, agent_pos)
                    return nid

        # New room
        self._room_counter += 1
        room_id = f"room_{self._room_counter}"
        self.nodes[room_id] = SGNode(
            node_id=room_id,
            node_type='room',
            label=room_type,
            position=agent_pos,
        )
        self._assign_objects_to_room(room_id, agent_pos)
        return room_id

    def _assign_objects_to_room(self, room_id: str, center: Tuple[int, int]):
        """Assign nearby objects to a room."""
        for nid, node in self.nodes.items():
            if node.node_type != 'object':
                continue
            dist = np.sqrt(
                (node.position[0] - center[0]) ** 2 +
                (node.position[1] - center[1]) ** 2
            )
            if dist < ROOM_RADIUS:
                self.room_assignments[nid] = room_id

    def _rebuild_edges(self):
        """Rebuild spatial relationship edges."""
        self.edges.clear()
        obj_nodes = [n for n in self.nodes.values() if n.node_type == 'object']

        for i, a in enumerate(obj_nodes):
            for b in obj_nodes[i + 1:]:
                dist = np.sqrt(
                    (a.position[0] - b.position[0]) ** 2 +
                    (a.position[1] - b.position[1]) ** 2
                )
                if dist < NEAR_THRESHOLD:
                    self.edges.append(SGEdge(a.node_id, b.node_id, 'near', dist))
                elif dist < COLOCATED_THRESHOLD:
                    self.edges.append(SGEdge(a.node_id, b.node_id, 'colocated', dist))

        # Add in_room edges
        for obj_id, room_id in self.room_assignments.items():
            if obj_id in self.nodes and room_id in self.nodes:
                self.edges.append(SGEdge(obj_id, room_id, 'in_room', 0))

    # ----------------------------------------------------------
    # Query / Describe
    # ----------------------------------------------------------

    def describe(self, max_objects: int = 20) -> str:
        """
        Generate a text description of the scene graph for VLM context.

        Returns a structured string like:
            "Rooms discovered: living_room, bedroom.
             Objects: chair (in living_room), couch (in living_room), bed (in bedroom).
             Relationships: chair near couch, bed near nightstand."
        """
        rooms = [n for n in self.nodes.values() if n.node_type == 'room']
        objects = sorted(
            [n for n in self.nodes.values() if n.node_type == 'object'],
            key=lambda n: n.observations, reverse=True,
        )[:max_objects]

        parts = []

        if rooms:
            room_list = ", ".join(r.label for r in rooms)
            parts.append(f"Rooms found: {room_list}")

        if objects:
            obj_strs = []
            for obj in objects:
                room_id = self.room_assignments.get(obj.node_id)
                room_label = self.nodes[room_id].label if room_id and room_id in self.nodes else None
                if room_label:
                    obj_strs.append(f"{obj.label} (in {room_label})")
                else:
                    obj_strs.append(obj.label)
            parts.append(f"Objects: {', '.join(obj_strs)}")

        near_edges = [e for e in self.edges if e.relation == 'near']
        if near_edges:
            rel_strs = []
            for e in near_edges[:10]:
                a = self.nodes.get(e.source)
                b = self.nodes.get(e.target)
                if a and b:
                    rel_strs.append(f"{a.label} near {b.label}")
            if rel_strs:
                parts.append(f"Nearby pairs: {', '.join(rel_strs)}")

        return ". ".join(parts) if parts else "No objects discovered yet."

    def get_objects_in_room(self, room_type: str) -> List[str]:
        """Get object labels assigned to a specific room type."""
        result = []
        for obj_id, room_id in self.room_assignments.items():
            room_node = self.nodes.get(room_id)
            obj_node = self.nodes.get(obj_id)
            if room_node and obj_node and room_node.label == room_type:
                result.append(obj_node.label)
        return result

    def get_likely_rooms_for_target(self, target: str) -> List[Tuple[str, float]]:
        """
        Get room types likely to contain the target, combining:
          1. Prior knowledge (OBJECT_ROOM_PRIORS)
          2. Observed evidence (what objects are in which rooms)

        Returns list of (room_type, score) sorted by likelihood.
        """
        priors = OBJECT_ROOM_PRIORS.get(target, {})

        # Start with prior scores
        scores = dict(priors)

        # Boost rooms where we've seen co-occurring objects
        from core.frontier import ROOM_COOCCURRENCE
        cooccur = ROOM_COOCCURRENCE.get(target, [])

        for obj_id, room_id in self.room_assignments.items():
            obj_node = self.nodes.get(obj_id)
            room_node = self.nodes.get(room_id)
            if obj_node and room_node and obj_node.label in cooccur:
                room_type = room_node.label
                scores[room_type] = scores.get(room_type, 0) + 0.3

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def score_frontier_by_context(
        self,
        frontier_pos: Tuple[int, int],
        target: str,
    ) -> float:
        """
        Score a frontier based on scene graph context.

        Boosts frontiers near rooms that are likely to contain the target.
        Penalizes frontiers near rooms already explored for the target.

        Returns multiplier (1.0 = neutral).
        """
        likely_rooms = self.get_likely_rooms_for_target(target)
        if not likely_rooms:
            return 1.0

        best_boost = 1.0

        for room_type, score in likely_rooms:
            # Find room nodes of this type
            for node in self.nodes.values():
                if node.node_type == 'room' and node.label == room_type:
                    dist = np.sqrt(
                        (node.position[0] - frontier_pos[0]) ** 2 +
                        (node.position[1] - frontier_pos[1]) ** 2
                    )
                    if dist < COLOCATED_THRESHOLD:
                        # Frontier is near a promising room
                        proximity = 1.0 - dist / COLOCATED_THRESHOLD
                        boost = 1.0 + score * proximity * 1.5
                        best_boost = max(best_boost, boost)

        return best_boost

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    def stats(self) -> dict:
        n_objects = sum(1 for n in self.nodes.values() if n.node_type == 'object')
        n_rooms = sum(1 for n in self.nodes.values() if n.node_type == 'room')
        n_edges = len(self.edges)
        return {'objects': n_objects, 'rooms': n_rooms, 'edges': n_edges}