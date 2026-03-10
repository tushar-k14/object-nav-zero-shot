"""
Pose Graph Optimization
=======================

Maintains a graph of poses connected by odometry and loop closure constraints.
Optimizes the graph to correct accumulated drift.

Structure:
- Nodes: Robot poses (keyframes)
- Edges: Relative pose constraints (odometry, loop closures)

Uses Gauss-Newton optimization (simplified version).
For production, use GTSAM or g2o.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Pose


class EdgeType(Enum):
    """Type of pose graph edge."""
    ODOMETRY = "odometry"
    LOOP_CLOSURE = "loop_closure"
    PRIOR = "prior"


@dataclass
class PoseNode:
    """A node in the pose graph."""
    id: int
    pose: Pose
    fixed: bool = False  # If true, pose is not optimized


@dataclass
class PoseEdge:
    """An edge (constraint) in the pose graph."""
    id: int
    from_node: int
    to_node: int
    measurement: Pose  # Relative pose measurement
    information: np.ndarray  # 6x6 information matrix (inverse covariance)
    edge_type: EdgeType = EdgeType.ODOMETRY


@dataclass
class OptimizationResult:
    """Result from pose graph optimization."""
    success: bool
    iterations: int
    initial_error: float
    final_error: float
    improvement: float


class PoseGraph:
    """
    Pose graph for SLAM optimization.
    """
    
    def __init__(self):
        self.nodes: Dict[int, PoseNode] = {}
        self.edges: Dict[int, PoseEdge] = {}
        
        self.next_node_id = 0
        self.next_edge_id = 0
        
        # Optimization parameters
        self.max_iterations = 100
        self.convergence_threshold = 1e-6
        self.lambda_init = 1e-3  # Levenberg-Marquardt damping
    
    def add_node(self, pose: Pose, fixed: bool = False) -> int:
        """
        Add a pose node to the graph.
        
        Args:
            pose: The pose
            fixed: If True, this pose won't be optimized
            
        Returns:
            Node ID
        """
        node_id = self.next_node_id
        self.next_node_id += 1
        
        self.nodes[node_id] = PoseNode(
            id=node_id,
            pose=pose,
            fixed=fixed
        )
        
        # First node is always fixed as reference
        if node_id == 0:
            self.nodes[node_id].fixed = True
        
        return node_id
    
    def add_edge(
        self,
        from_node: int,
        to_node: int,
        measurement: Pose,
        information: np.ndarray = None,
        edge_type: EdgeType = EdgeType.ODOMETRY
    ) -> int:
        """
        Add a constraint edge between two nodes.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            measurement: Relative pose measurement (from -> to)
            information: 6x6 information matrix (default: identity)
            edge_type: Type of constraint
            
        Returns:
            Edge ID
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Nodes {from_node} or {to_node} not in graph")
        
        edge_id = self.next_edge_id
        self.next_edge_id += 1
        
        if information is None:
            # Default information based on edge type
            if edge_type == EdgeType.LOOP_CLOSURE:
                # Loop closures typically more uncertain
                information = np.eye(6) * 100
            else:
                information = np.eye(6) * 1000
        
        self.edges[edge_id] = PoseEdge(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            measurement=measurement,
            information=information,
            edge_type=edge_type
        )
        
        return edge_id
    
    def add_odometry_edge(
        self,
        from_node: int,
        to_node: int,
        measurement: Pose,
        covariance: np.ndarray = None
    ) -> int:
        """Add an odometry constraint."""
        information = np.linalg.inv(covariance) if covariance is not None else None
        return self.add_edge(from_node, to_node, measurement, information, EdgeType.ODOMETRY)
    
    def add_loop_closure_edge(
        self,
        from_node: int,
        to_node: int,
        measurement: Pose,
        covariance: np.ndarray = None
    ) -> int:
        """Add a loop closure constraint."""
        information = np.linalg.inv(covariance) if covariance is not None else None
        return self.add_edge(from_node, to_node, measurement, information, EdgeType.LOOP_CLOSURE)
    
    def optimize(self) -> OptimizationResult:
        """
        Optimize the pose graph using Gauss-Newton.
        
        Returns:
            OptimizationResult
        """
        if len(self.nodes) < 2 or len(self.edges) == 0:
            return OptimizationResult(
                success=True,
                iterations=0,
                initial_error=0,
                final_error=0,
                improvement=0
            )
        
        initial_error = self._compute_total_error()
        prev_error = initial_error
        
        for iteration in range(self.max_iterations):
            # Build linear system
            H, b = self._build_linear_system()
            
            # Solve for update
            try:
                # Add damping for Levenberg-Marquardt
                H_damped = H + self.lambda_init * np.diag(np.diag(H))
                dx = np.linalg.solve(H_damped, b)
            except np.linalg.LinAlgError:
                break
            
            # Apply update
            self._apply_update(dx)
            
            # Check convergence
            current_error = self._compute_total_error()
            
            if abs(prev_error - current_error) < self.convergence_threshold:
                break
            
            prev_error = current_error
        
        final_error = self._compute_total_error()
        
        return OptimizationResult(
            success=True,
            iterations=iteration + 1,
            initial_error=initial_error,
            final_error=final_error,
            improvement=(initial_error - final_error) / max(initial_error, 1e-10)
        )
    
    def _compute_total_error(self) -> float:
        """Compute total error (sum of squared residuals)."""
        total_error = 0.0
        
        for edge in self.edges.values():
            residual = self._compute_residual(edge)
            error = residual.T @ edge.information @ residual
            total_error += error
        
        return total_error
    
    def _compute_residual(self, edge: PoseEdge) -> np.ndarray:
        """
        Compute residual for an edge.
        
        Residual = log(measurement^-1 * (from_pose^-1 * to_pose))
        """
        from_pose = self.nodes[edge.from_node].pose
        to_pose = self.nodes[edge.to_node].pose
        
        # Predicted relative pose
        T_from = from_pose.to_matrix()
        T_to = to_pose.to_matrix()
        T_predicted = np.linalg.inv(T_from) @ T_to
        
        # Measured relative pose
        T_measured = edge.measurement.to_matrix()
        
        # Error transform
        T_error = np.linalg.inv(T_measured) @ T_predicted
        
        # Convert to 6D error vector (simplified: just use translation and rotation vector)
        translation_error = T_error[:3, 3]
        
        # Rotation error as axis-angle
        R_error = T_error[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        
        if abs(angle) < 1e-6:
            rotation_error = np.zeros(3)
        else:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle))
            rotation_error = axis * angle
        
        return np.concatenate([translation_error, rotation_error])
    
    def _build_linear_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the linear system H * dx = b."""
        n_nodes = len(self.nodes)
        n_params = n_nodes * 6  # 6 DoF per pose
        
        H = np.zeros((n_params, n_params))
        b = np.zeros(n_params)
        
        for edge in self.edges.values():
            from_idx = edge.from_node * 6
            to_idx = edge.to_node * 6
            
            # Compute Jacobians (simplified: use identity for small updates)
            Ji = -np.eye(6)  # Jacobian w.r.t. from_pose
            Jj = np.eye(6)   # Jacobian w.r.t. to_pose
            
            residual = self._compute_residual(edge)
            omega = edge.information
            
            # Update H and b
            # H += J^T * Omega * J
            # b += J^T * Omega * r
            
            if not self.nodes[edge.from_node].fixed:
                H[from_idx:from_idx+6, from_idx:from_idx+6] += Ji.T @ omega @ Ji
                b[from_idx:from_idx+6] += Ji.T @ omega @ residual
                
                if not self.nodes[edge.to_node].fixed:
                    H[from_idx:from_idx+6, to_idx:to_idx+6] += Ji.T @ omega @ Jj
                    H[to_idx:to_idx+6, from_idx:from_idx+6] += Jj.T @ omega @ Ji
            
            if not self.nodes[edge.to_node].fixed:
                H[to_idx:to_idx+6, to_idx:to_idx+6] += Jj.T @ omega @ Jj
                b[to_idx:to_idx+6] += Jj.T @ omega @ residual
        
        return H, b
    
    def _apply_update(self, dx: np.ndarray):
        """Apply the computed update to all non-fixed nodes."""
        for node_id, node in self.nodes.items():
            if node.fixed:
                continue
            
            idx = node_id * 6
            update = dx[idx:idx+6]
            
            # Update translation
            node.pose.position = node.pose.position + update[:3]
            
            # Update rotation (simplified: small angle approximation)
            dR = self._exp_so3(update[3:6])
            if isinstance(node.pose.rotation, np.ndarray) and node.pose.rotation.shape == (3, 3):
                node.pose.rotation = dR @ node.pose.rotation
    
    def _exp_so3(self, omega: np.ndarray) -> np.ndarray:
        """Exponential map from so(3) to SO(3)."""
        angle = np.linalg.norm(omega)
        
        if angle < 1e-10:
            return np.eye(3)
        
        axis = omega / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R
    
    def get_optimized_poses(self) -> Dict[int, Pose]:
        """Get all optimized poses."""
        return {node_id: node.pose for node_id, node in self.nodes.items()}
    
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory as array of positions."""
        positions = []
        for node_id in sorted(self.nodes.keys()):
            positions.append(self.nodes[node_id].pose.position)
        return np.array(positions)
    
    def save(self, filepath: str):
        """Save pose graph to file."""
        data = {
            'nodes': [
                {
                    'id': node.id,
                    'position': node.pose.position.tolist(),
                    'rotation': node.pose.rotation.tolist() if isinstance(node.pose.rotation, np.ndarray) else node.pose.rotation,
                    'fixed': node.fixed
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'id': edge.id,
                    'from': edge.from_node,
                    'to': edge.to_node,
                    'type': edge.edge_type.value,
                    'measurement_position': edge.measurement.position.tolist(),
                    'measurement_rotation': edge.measurement.rotation.tolist() if isinstance(edge.measurement.rotation, np.ndarray) else edge.measurement.rotation
                }
                for edge in self.edges.values()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_statistics(self) -> Dict:
        """Get pose graph statistics."""
        n_odometry = sum(1 for e in self.edges.values() if e.edge_type == EdgeType.ODOMETRY)
        n_loop = sum(1 for e in self.edges.values() if e.edge_type == EdgeType.LOOP_CLOSURE)
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_odometry_edges': n_odometry,
            'num_loop_closure_edges': n_loop,
            'total_error': self._compute_total_error()
        }
