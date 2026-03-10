"""
SLAM Module
===========

Visual SLAM system with:
- Visual Odometry (ORB features + ICP refinement)
- Loop Closure Detection (global descriptors + geometric verification)  
- Pose Graph Optimization (Gauss-Newton)
- Ground Truth Comparison (ATE, RPE metrics)
"""

from .feature_extractor import FeatureExtractor, FeatureTracker, FeatureMatch
from .visual_odometry import VisualOdometry, VOConfig, OdometryResult
from .loop_closure import LoopClosureDetector, LoopClosureConfig, GlobalDescriptorExtractor
from .pose_graph import PoseGraph, PoseNode, PoseEdge, EdgeType, OptimizationResult
from .slam_system import VisualSLAM, SLAMConfig, SLAMResult

__all__ = [
    # Feature extraction
    'FeatureExtractor',
    'FeatureTracker', 
    'FeatureMatch',
    
    # Visual Odometry
    'VisualOdometry',
    'VOConfig',
    'OdometryResult',
    
    # Loop Closure
    'LoopClosureDetector',
    'LoopClosureConfig',
    'GlobalDescriptorExtractor',
    
    # Pose Graph
    'PoseGraph',
    'PoseNode',
    'PoseEdge',
    'EdgeType',
    'OptimizationResult',
    
    # Complete SLAM
    'VisualSLAM',
    'SLAMConfig',
    'SLAMResult',
]