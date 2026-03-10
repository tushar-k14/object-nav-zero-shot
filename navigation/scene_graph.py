"""
Scene Graph Module
==================

Builds a semantic scene graph from:
1. Detected and tracked objects
2. Spatial relationships between objects
3. Room/region segmentation

The scene graph represents:
- Nodes: Objects with attributes (class, position, size, confidence)
- Edges: Spatial relationships (near, on, in, next_to, etc.)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class SceneNode:
    """A node in the scene graph representing an object."""
    node_id: int
    class_name: str
    position: np.ndarray  # World position [x, y, z]
    confidence: float
    observations: int
    bbox_size: Optional[Tuple[float, float, float]] = None  # Estimated size
    attributes: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'class_name': self.class_name,
            'position': self.position.tolist(),
            'confidence': float(self.confidence),
            'observations': self.observations,
            'bbox_size': self.bbox_size,
            'attributes': self.attributes
        }


@dataclass 
class SceneEdge:
    """An edge in the scene graph representing a relationship."""
    source_id: int
    target_id: int
    relationship: str  # 'near', 'on', 'next_to', 'above', 'below', etc.
    distance: float
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship': self.relationship,
            'distance': float(self.distance),
            'confidence': float(self.confidence)
        }


# Typical object heights for relationship inference (in meters)
OBJECT_HEIGHTS = {
    'chair': 0.8,
    'couch': 0.8,
    'bed': 0.5,
    'dining table': 0.75,
    'toilet': 0.4,
    'tv': 0.6,
    'laptop': 0.03,
    'refrigerator': 1.7,
    'oven': 0.9,
    'microwave': 0.3,
    'sink': 0.85,
    'book': 0.03,
    'clock': 0.3,
    'vase': 0.3,
    'potted plant': 0.5,
    'bottle': 0.25,
    'cup': 0.1,
    'bowl': 0.1,
}

# Objects that can support other objects
SUPPORTING_OBJECTS = {'dining table', 'desk', 'bed', 'couch', 'counter', 'shelf'}

# Objects typically found in specific rooms
ROOM_INDICATORS = {
    'kitchen': ['refrigerator', 'oven', 'microwave', 'sink', 'toaster'],
    'bathroom': ['toilet', 'sink', 'bathtub', 'shower'],
    'bedroom': ['bed', 'wardrobe', 'dresser'],
    'living_room': ['couch', 'tv', 'coffee table'],
    'dining_room': ['dining table', 'chair'],
    'office': ['desk', 'chair', 'laptop', 'monitor'],
}


class SceneGraph:
    """
    Builds and maintains a scene graph from semantic mapping.
    
    The graph captures:
    1. Object nodes with positions and attributes
    2. Spatial relationships between objects
    3. Room/region associations
    """
    
    def __init__(
        self,
        near_threshold: float = 2.0,      # meters
        on_threshold: float = 0.5,         # meters (vertical)
        same_height_threshold: float = 0.3 # meters
    ):
        """
        Initialize scene graph.
        
        Args:
            near_threshold: Distance to consider objects "near" each other
            on_threshold: Vertical distance for "on" relationship
            same_height_threshold: Height difference for "same level"
        """
        self.near_threshold = near_threshold
        self.on_threshold = on_threshold
        self.same_height_threshold = same_height_threshold
        
        # Graph structure
        self.nodes: Dict[int, SceneNode] = {}
        self.edges: List[SceneEdge] = []
        
        # Adjacency lists for quick lookup
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)
        
        # Room assignments
        self.room_assignments: Dict[int, str] = {}  # node_id -> room_key
        self.detected_rooms: Dict[str, Dict] = {}   # room_key -> room_info
        
        # Statistics
        self.total_objects = 0
    
    def reset(self):
        """Reset the scene graph."""
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        self.room_assignments.clear()
        self.detected_rooms.clear()
        self.total_objects = 0
    
    def update_from_tracker(self, tracked_objects: Dict):
        """
        Update scene graph from object tracker.
        
        Args:
            tracked_objects: Dict of TrackedObject from detector
        """
        # Add/update nodes for each tracked object
        for obj_id, obj in tracked_objects.items():
            if obj.observations < 3:  # Skip unconfirmed
                continue
            
            if obj_id in self.nodes:
                # Update existing node
                node = self.nodes[obj_id]
                node.position = obj.mean_position
                node.confidence = obj.mean_confidence
                node.observations = obj.observations
            else:
                # Create new node
                node = SceneNode(
                    node_id=obj_id,
                    class_name=obj.class_name,
                    position=obj.mean_position,
                    confidence=obj.mean_confidence,
                    observations=obj.observations,
                    bbox_size=self._estimate_size(obj.class_name)
                )
                self.nodes[obj_id] = node
                self.total_objects += 1
        
        # Rebuild edges
        self._compute_relationships()
        
        # Infer room assignments
        self._infer_rooms()
    
    def _estimate_size(self, class_name: str) -> Tuple[float, float, float]:
        """Estimate object size based on class."""
        height = OBJECT_HEIGHTS.get(class_name.lower(), 0.5)
        # Rough estimates for width/depth based on object type
        if class_name.lower() in ['chair', 'toilet']:
            return (0.5, 0.5, height)
        elif class_name.lower() in ['couch', 'bed']:
            return (2.0, 1.0, height)
        elif class_name.lower() in ['dining table']:
            return (1.5, 0.9, height)
        elif class_name.lower() in ['refrigerator']:
            return (0.7, 0.7, height)
        elif class_name.lower() in ['tv', 'laptop']:
            return (0.8, 0.1, height)
        else:
            return (0.3, 0.3, height)
    
    def _compute_relationships(self):
        """Compute spatial relationships between all object pairs."""
        self.edges.clear()
        self.adjacency.clear()
        
        node_ids = list(self.nodes.keys())
        
        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i+1:]:
                node1 = self.nodes[id1]
                node2 = self.nodes[id2]
                
                # Compute distance
                pos1 = node1.position
                pos2 = node2.position
                
                horizontal_dist = np.sqrt(
                    (pos1[0] - pos2[0])**2 + (pos1[2] - pos2[2])**2
                )
                vertical_diff = pos1[1] - pos2[1]  # Y is up
                
                total_dist = np.linalg.norm(pos1 - pos2)
                
                # Determine relationships
                relationships = []
                
                # Near relationship
                if horizontal_dist < self.near_threshold:
                    relationships.append(('near', total_dist))
                
                # On/above/below relationships
                if horizontal_dist < 1.0:  # Objects must be roughly aligned
                    if abs(vertical_diff) < self.on_threshold:
                        # Same level
                        relationships.append(('next_to', horizontal_dist))
                    elif vertical_diff > self.on_threshold:
                        # node1 is above node2
                        # Check if node2 can support node1
                        if node2.class_name.lower() in SUPPORTING_OBJECTS:
                            relationships.append(('on', total_dist))
                        else:
                            relationships.append(('above', total_dist))
                    elif vertical_diff < -self.on_threshold:
                        # node1 is below node2
                        if node1.class_name.lower() in SUPPORTING_OBJECTS:
                            # node2 is ON node1 (reverse edge)
                            edge = SceneEdge(
                                source_id=id2,
                                target_id=id1,
                                relationship='on',
                                distance=total_dist
                            )
                            self.edges.append(edge)
                            self.adjacency[id2].add(id1)
                            self.adjacency[id1].add(id2)
                        else:
                            relationships.append(('below', total_dist))
                
                # Add edges for all relationships
                for rel, dist in relationships:
                    edge = SceneEdge(
                        source_id=id1,
                        target_id=id2,
                        relationship=rel,
                        distance=dist
                    )
                    self.edges.append(edge)
                    self.adjacency[id1].add(id2)
                    self.adjacency[id2].add(id1)
    
    def _infer_rooms(self):
        """Infer room types based on object clusters."""
        self.room_assignments.clear()
        self.detected_rooms = {}  # room_type -> {'center': position, 'objects': [node_ids]}
        
        # First pass: identify room indicator objects
        room_indicators_found = {}  # room_type -> list of (node_id, position)
        
        for node_id, node in self.nodes.items():
            class_lower = node.class_name.lower()
            for room_type, indicators in ROOM_INDICATORS.items():
                if class_lower in indicators:
                    if room_type not in room_indicators_found:
                        room_indicators_found[room_type] = []
                    room_indicators_found[room_type].append((node_id, node.position))
        
        # Second pass: cluster indicators into distinct rooms
        # (e.g., two toilets far apart = two bathrooms)
        room_cluster_threshold = 5.0  # meters - objects within this are same room
        
        for room_type, indicators in room_indicators_found.items():
            clusters = []
            
            for node_id, position in indicators:
                # Find if this indicator belongs to existing cluster
                assigned = False
                for cluster in clusters:
                    cluster_center = np.mean([pos for _, pos in cluster], axis=0)
                    dist = np.linalg.norm(position - cluster_center)
                    if dist < room_cluster_threshold:
                        cluster.append((node_id, position))
                        assigned = True
                        break
                
                if not assigned:
                    clusters.append([(node_id, position)])
            
            # Create room entries for each cluster
            for i, cluster in enumerate(clusters):
                room_key = f"{room_type}_{i}" if len(clusters) > 1 else room_type
                center = np.mean([pos for _, pos in cluster], axis=0)
                self.detected_rooms[room_key] = {
                    'type': room_type,
                    'center': center,
                    'indicator_objects': [nid for nid, _ in cluster]
                }
        
        # Third pass: assign all objects to nearest room
        for node_id, node in self.nodes.items():
            if not self.detected_rooms:
                continue
            
            # Find nearest room
            min_dist = float('inf')
            nearest_room = None
            
            for room_key, room_info in self.detected_rooms.items():
                dist = np.linalg.norm(node.position - room_info['center'])
                if dist < min_dist and dist < 8.0:  # Max 8m to assign to room
                    min_dist = dist
                    nearest_room = room_key
            
            if nearest_room:
                self.room_assignments[node_id] = nearest_room
    
    def get_objects_by_class(self, class_name: str) -> List[SceneNode]:
        """Get all objects of a specific class."""
        return [
            node for node in self.nodes.values()
            if node.class_name.lower() == class_name.lower()
        ]
    
    def get_objects_in_room(self, room_type: str) -> List[SceneNode]:
        """Get all objects in a specific room type."""
        return [
            self.nodes[node_id]
            for node_id, room in self.room_assignments.items()
            if room == room_type and node_id in self.nodes
        ]
    
    def get_nearby_objects(self, node_id: int, max_distance: float = None) -> List[Tuple[SceneNode, float]]:
        """
        Get objects near a specific object.
        
        Returns:
            List of (node, distance) tuples sorted by distance
        """
        if node_id not in self.nodes:
            return []
        
        source = self.nodes[node_id]
        nearby = []
        
        for other_id, other in self.nodes.items():
            if other_id == node_id:
                continue
            
            dist = np.linalg.norm(source.position - other.position)
            
            if max_distance is None or dist < max_distance:
                nearby.append((other, dist))
        
        return sorted(nearby, key=lambda x: x[1])
    
    def find_path_to_object(
        self,
        start_pos: np.ndarray,
        target_class: str
    ) -> Optional[SceneNode]:
        """
        Find the nearest object of target class.
        
        Args:
            start_pos: Starting position [x, y, z]
            target_class: Target object class
            
        Returns:
            Nearest SceneNode of target class, or None
        """
        targets = self.get_objects_by_class(target_class)
        
        if not targets:
            return None
        
        # Find nearest
        nearest = None
        min_dist = float('inf')
        
        for node in targets:
            dist = np.linalg.norm(start_pos - node.position)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def get_statistics(self) -> Dict:
        """Get scene graph statistics."""
        # Count objects by class
        class_counts = defaultdict(int)
        for node in self.nodes.values():
            class_counts[node.class_name] += 1
        
        # Count relationships
        rel_counts = defaultdict(int)
        for edge in self.edges:
            rel_counts[edge.relationship] += 1
        
        # Count unique rooms (not objects per room)
        room_counts = {}
        if hasattr(self, 'detected_rooms'):
            for room_key, room_info in self.detected_rooms.items():
                room_type = room_info['type']
                room_counts[room_type] = room_counts.get(room_type, 0) + 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'class_counts': dict(class_counts),
            'relationship_counts': dict(rel_counts),
            'room_counts': room_counts,
            'total_rooms': len(getattr(self, 'detected_rooms', {}))
        }
    
    def to_dict(self) -> Dict:
        """Export scene graph as dictionary."""
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges],
            'room_assignments': {
                str(k): v for k, v in self.room_assignments.items()
            },
            'statistics': self.get_statistics()
        }
    
    def save(self, filepath: str):
        """Save scene graph to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Scene graph saved to {filepath}")
    
    def visualize_text(self) -> str:
        """Create text visualization of scene graph."""
        lines = []
        lines.append("=" * 50)
        lines.append("SCENE GRAPH")
        lines.append("=" * 50)
        
        # Show detected rooms
        if hasattr(self, 'detected_rooms') and self.detected_rooms:
            lines.append(f"\n[DETECTED ROOMS: {len(self.detected_rooms)}]")
            for room_key, room_info in self.detected_rooms.items():
                center = room_info['center']
                n_indicators = len(room_info['indicator_objects'])
                lines.append(f"  - {room_key}: center ({center[0]:.1f}, {center[2]:.1f}), "
                           f"{n_indicators} indicator objects")
        
        # Objects by room
        rooms_with_objects = defaultdict(list)
        unassigned = []
        
        for node_id, node in self.nodes.items():
            room = self.room_assignments.get(node_id, None)
            if room:
                rooms_with_objects[room].append(node)
            else:
                unassigned.append(node)
        
        for room, objects in sorted(rooms_with_objects.items()):
            lines.append(f"\n[{room.upper()}] ({len(objects)} objects)")
            for obj in objects[:5]:  # Limit to 5 per room
                pos = obj.position
                lines.append(f"  - {obj.class_name} @ ({pos[0]:.1f}, {pos[2]:.1f})")
            if len(objects) > 5:
                lines.append(f"  ... and {len(objects) - 5} more")
        
        if unassigned:
            lines.append(f"\n[UNASSIGNED] ({len(unassigned)} objects)")
            for obj in unassigned[:5]:
                pos = obj.position
                lines.append(f"  - {obj.class_name} @ ({pos[0]:.1f}, {pos[2]:.1f})")
            if len(unassigned) > 5:
                lines.append(f"  ... and {len(unassigned) - 5} more")
        
        # Relationships summary
        if self.edges:
            rel_counts = defaultdict(int)
            for edge in self.edges:
                rel_counts[edge.relationship] += 1
            
            lines.append(f"\n[RELATIONSHIPS: {len(self.edges)} total]")
            for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {rel}: {count}")
        
        lines.append("=" * 50)
        return "\n".join(lines)