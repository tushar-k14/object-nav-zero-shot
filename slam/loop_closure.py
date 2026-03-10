"""
Loop Closure Detection
======================

Detects when the robot revisits a previously seen location.
Uses global image descriptors for place recognition and
geometric verification to confirm matches.

Approach:
1. NetVLAD-style global descriptors (simplified version using feature aggregation)
2. Candidate selection based on descriptor similarity
3. Geometric verification with feature matching + RANSAC
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Pose, LoopClosureResult, Keyframe
from slam.feature_extractor import FeatureExtractor


@dataclass
class LoopClosureConfig:
    """Loop closure configuration."""
    # Place recognition
    descriptor_dim: int = 256           # Global descriptor dimension
    min_keyframe_gap: int = 20          # Minimum frames between loop candidates
    similarity_threshold: float = 0.7   # Minimum similarity for candidates
    max_candidates: int = 5             # Max candidates to verify
    
    # Geometric verification
    min_inliers: int = 30               # Minimum inliers for valid loop
    ransac_threshold: float = 5.0       # RANSAC pixel threshold
    
    # Covisibility
    covisibility_threshold: int = 15    # Min shared features for covisibility


class GlobalDescriptorExtractor:
    """
    Extract global image descriptors for place recognition.
    
    Uses a simplified NetVLAD-style approach:
    1. Extract local features (ORB)
    2. Cluster into visual words
    3. Aggregate into VLAD descriptor
    
    For production, use actual NetVLAD or similar learned descriptors.
    """
    
    def __init__(
        self,
        n_clusters: int = 64,
        descriptor_dim: int = 256,
        n_features: int = 500
    ):
        self.n_clusters = n_clusters
        self.descriptor_dim = descriptor_dim
        self.n_features = n_features
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(n_features=n_features)
        
        # Visual vocabulary (will be built online)
        self.vocabulary = None
        self.all_descriptors = []
        self.vocab_built = False
        self.min_descriptors_for_vocab = 5000
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract global descriptor from image.
        
        Args:
            image: RGB or grayscale image
            
        Returns:
            Global descriptor [descriptor_dim]
        """
        # Extract local features
        _, descriptors = self.feature_extractor.extract(image)
        
        if len(descriptors) == 0:
            return np.zeros(self.descriptor_dim)
        
        # Store for vocabulary building
        if not self.vocab_built:
            self.all_descriptors.append(descriptors)
            total_descs = sum(len(d) for d in self.all_descriptors)
            
            if total_descs >= self.min_descriptors_for_vocab:
                self._build_vocabulary()
        
        # If vocabulary not ready, use simple aggregation
        if self.vocabulary is None:
            return self._simple_aggregation(descriptors)
        
        # VLAD encoding
        return self._vlad_encode(descriptors)
    
    def _simple_aggregation(self, descriptors: np.ndarray) -> np.ndarray:
        """Simple descriptor aggregation before vocabulary is built."""
        # Convert to float and normalize
        desc_float = descriptors.astype(np.float32)
        
        # Mean pooling
        mean_desc = np.mean(desc_float, axis=0)
        
        # Pad or truncate to target dimension
        if len(mean_desc) < self.descriptor_dim:
            result = np.zeros(self.descriptor_dim)
            result[:len(mean_desc)] = mean_desc
        else:
            result = mean_desc[:self.descriptor_dim]
        
        # L2 normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result
    
    def _build_vocabulary(self):
        """Build visual vocabulary using k-means."""
        print("Building visual vocabulary...")
        
        # Concatenate all descriptors
        all_descs = np.vstack(self.all_descriptors).astype(np.float32)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(
            all_descs,
            self.n_clusters,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS
        )
        
        self.vocabulary = centers
        self.vocab_built = True
        
        # Clear stored descriptors to save memory
        self.all_descriptors = []
        
        print(f"Vocabulary built with {self.n_clusters} clusters")
    
    def _vlad_encode(self, descriptors: np.ndarray) -> np.ndarray:
        """VLAD encoding of local descriptors."""
        desc_float = descriptors.astype(np.float32)
        
        # Assign each descriptor to nearest cluster
        dists = np.linalg.norm(
            desc_float[:, np.newaxis, :] - self.vocabulary[np.newaxis, :, :],
            axis=2
        )
        assignments = np.argmin(dists, axis=1)
        
        # Compute VLAD
        vlad = np.zeros((self.n_clusters, self.vocabulary.shape[1]))
        
        for i, cluster_idx in enumerate(assignments):
            vlad[cluster_idx] += desc_float[i] - self.vocabulary[cluster_idx]
        
        # Flatten
        vlad = vlad.flatten()
        
        # Intra-normalization
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        
        # L2 normalize
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad = vlad / norm
        
        # Reduce dimension if needed
        if len(vlad) > self.descriptor_dim:
            # Simple PCA-like reduction (just truncate for simplicity)
            vlad = vlad[:self.descriptor_dim]
        elif len(vlad) < self.descriptor_dim:
            padded = np.zeros(self.descriptor_dim)
            padded[:len(vlad)] = vlad
            vlad = padded
        
        return vlad


class LoopClosureDetector:
    """
    Detect loop closures between keyframes.
    """
    
    def __init__(self, config: LoopClosureConfig = None):
        self.config = config or LoopClosureConfig()
        
        # Global descriptor extractor
        self.global_extractor = GlobalDescriptorExtractor(
            descriptor_dim=self.config.descriptor_dim
        )
        
        # Feature extractor for geometric verification
        self.feature_extractor = FeatureExtractor(n_features=1000)
        
        # Database of keyframe descriptors
        self.keyframe_descriptors: Dict[int, np.ndarray] = {}
        self.keyframe_features: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Statistics
        self.total_queries = 0
        self.total_detections = 0
    
    def add_keyframe(
        self,
        keyframe_id: int,
        image: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None
    ):
        """
        Add a keyframe to the database.
        
        Args:
            keyframe_id: Unique keyframe ID
            image: RGB or grayscale image
            keypoints: Optional pre-computed keypoints
            descriptors: Optional pre-computed descriptors
        """
        # Extract global descriptor
        global_desc = self.global_extractor.extract(image)
        self.keyframe_descriptors[keyframe_id] = global_desc
        
        # Store local features for geometric verification
        if keypoints is None or descriptors is None:
            keypoints, descriptors = self.feature_extractor.extract(image)
        
        self.keyframe_features[keyframe_id] = (keypoints, descriptors)
    
    def detect(
        self,
        query_keyframe_id: int,
        query_image: np.ndarray,
        query_keypoints: Optional[np.ndarray] = None,
        query_descriptors: Optional[np.ndarray] = None
    ) -> LoopClosureResult:
        """
        Detect if current keyframe closes a loop with any previous keyframe.
        
        Args:
            query_keyframe_id: ID of query keyframe
            query_image: Query image
            query_keypoints: Optional pre-computed keypoints
            query_descriptors: Optional pre-computed descriptors
            
        Returns:
            LoopClosureResult
        """
        self.total_queries += 1
        
        # Extract global descriptor
        query_global = self.global_extractor.extract(query_image)
        
        # Find candidates (excluding recent keyframes)
        candidates = self._find_candidates(
            query_keyframe_id,
            query_global
        )
        
        if not candidates:
            return LoopClosureResult(
                detected=False,
                query_keyframe_id=query_keyframe_id
            )
        
        # Extract local features if not provided
        if query_keypoints is None or query_descriptors is None:
            query_keypoints, query_descriptors = self.feature_extractor.extract(query_image)
        
        # Geometric verification
        for candidate_id, similarity in candidates:
            result = self._verify_candidate(
                query_keyframe_id,
                query_keypoints,
                query_descriptors,
                candidate_id,
                similarity
            )
            
            if result.verified:
                self.total_detections += 1
                return result
        
        return LoopClosureResult(
            detected=False,
            query_keyframe_id=query_keyframe_id
        )
    
    def _find_candidates(
        self,
        query_id: int,
        query_descriptor: np.ndarray
    ) -> List[Tuple[int, float]]:
        """Find loop closure candidates based on global descriptor similarity."""
        candidates = []
        
        for kf_id, kf_desc in self.keyframe_descriptors.items():
            # Skip recent keyframes
            if abs(kf_id - query_id) < self.config.min_keyframe_gap:
                continue
            
            # Compute cosine similarity
            similarity = np.dot(query_descriptor, kf_desc)
            
            if similarity > self.config.similarity_threshold:
                candidates.append((kf_id, similarity))
        
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        return candidates[:self.config.max_candidates]
    
    def _verify_candidate(
        self,
        query_id: int,
        query_keypoints: np.ndarray,
        query_descriptors: np.ndarray,
        candidate_id: int,
        similarity: float
    ) -> LoopClosureResult:
        """Geometrically verify a loop closure candidate."""
        # Get candidate features
        if candidate_id not in self.keyframe_features:
            return LoopClosureResult(
                detected=False,
                query_keyframe_id=query_id,
                match_keyframe_id=candidate_id,
                similarity_score=similarity
            )
        
        cand_keypoints, cand_descriptors = self.keyframe_features[candidate_id]
        
        # Match features with geometric check
        inlier_matches, F, inlier_count = self.feature_extractor.match_with_geometric_check(
            query_keypoints,
            query_descriptors,
            cand_keypoints,
            cand_descriptors,
            self.config.ransac_threshold
        )
        
        verified = inlier_count >= self.config.min_inliers
        
        # Estimate relative pose if verified
        relative_pose = None
        if verified and len(inlier_matches) >= 8:
            relative_pose = self._estimate_relative_pose(
                query_keypoints,
                cand_keypoints,
                inlier_matches
            )
        
        return LoopClosureResult(
            detected=verified,
            query_keyframe_id=query_id,
            match_keyframe_id=candidate_id,
            similarity_score=similarity,
            relative_pose=relative_pose,
            inlier_count=inlier_count,
            verified=verified
        )
    
    def _estimate_relative_pose(
        self,
        kp1: np.ndarray,
        kp2: np.ndarray,
        matches: list
    ) -> Optional[Pose]:
        """Estimate relative pose between two views."""
        if len(matches) < 8:
            return None
        
        pts1 = np.array([kp1[m.idx1] for m in matches])
        pts2 = np.array([kp2[m.idx2] for m in matches])
        
        # Estimate essential matrix
        # Note: This requires camera intrinsics for accurate pose
        # For now, return None and let pose graph handle it
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get loop closure statistics."""
        return {
            'total_queries': self.total_queries,
            'total_detections': self.total_detections,
            'detection_rate': self.total_detections / max(1, self.total_queries),
            'database_size': len(self.keyframe_descriptors)
        }
