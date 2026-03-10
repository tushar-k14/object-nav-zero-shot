"""
Feature Extractor
=================

Extracts and matches visual features for odometry and loop closure.
Uses ORB for robustness and speed.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class FeatureMatch:
    """A feature match between two frames."""
    idx1: int  # Index in frame 1
    idx2: int  # Index in frame 2
    distance: float


class FeatureExtractor:
    """
    ORB feature extractor with robust matching.
    """
    
    def __init__(
        self,
        n_features: int = 1000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        fast_threshold: int = 20,
        match_ratio: float = 0.75,  # Lowe's ratio test
        cross_check: bool = True
    ):
        """
        Initialize ORB extractor.
        
        Args:
            n_features: Maximum number of features
            scale_factor: Pyramid scale factor
            n_levels: Number of pyramid levels
            fast_threshold: FAST corner threshold
            match_ratio: Ratio test threshold
            cross_check: Use cross-check matching
        """
        self.n_features = n_features
        self.match_ratio = match_ratio
        self.cross_check = cross_check
        
        # Create ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            fastThreshold=fast_threshold
        )
        
        # Create matchers
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        self.flann_matcher = None  # For larger scale matching
    
    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ORB features from image.
        
        Args:
            image: Grayscale or BGR image
            mask: Optional mask (255 = valid regions)
            
        Returns:
            keypoints: [N, 2] pixel coordinates
            descriptors: [N, 32] ORB descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect and compute
        kps, descs = self.orb.detectAndCompute(gray, mask)
        
        if kps is None or len(kps) == 0:
            return np.zeros((0, 2)), np.zeros((0, 32), dtype=np.uint8)
        
        # Convert keypoints to array
        keypoints = np.array([kp.pt for kp in kps], dtype=np.float32)
        
        if descs is None:
            descs = np.zeros((len(keypoints), 32), dtype=np.uint8)
        
        return keypoints, descs
    
    def extract_with_depth(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        camera_fx: float,
        camera_fy: float,
        camera_cx: float,
        camera_cy: float,
        min_depth: float = 0.1,
        max_depth: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features with 3D positions.
        
        Args:
            image: RGB or grayscale image
            depth: Depth image in meters
            camera_*: Camera intrinsics
            min_depth, max_depth: Valid depth range
            
        Returns:
            keypoints: [N, 2] pixel coordinates
            descriptors: [N, 32] ORB descriptors
            points_3d: [N, 3] 3D positions in camera frame
        """
        keypoints, descriptors = self.extract(image)
        
        if len(keypoints) == 0:
            return keypoints, descriptors, np.zeros((0, 3))
        
        # Get depths at keypoint locations
        u = keypoints[:, 0].astype(int)
        v = keypoints[:, 1].astype(int)
        
        # Clamp to image bounds
        u = np.clip(u, 0, depth.shape[1] - 1)
        v = np.clip(v, 0, depth.shape[0] - 1)
        
        depths = depth[v, u]
        
        # Filter by valid depth
        valid = (depths > min_depth) & (depths < max_depth)
        
        keypoints = keypoints[valid]
        descriptors = descriptors[valid]
        depths = depths[valid]
        
        # Unproject to 3D
        u = keypoints[:, 0]
        v = keypoints[:, 1]
        
        x = (u - camera_cx) * depths / camera_fx
        y = (v - camera_cy) * depths / camera_fy
        z = depths
        
        points_3d = np.stack([x, y, z], axis=1)
        
        return keypoints, descriptors, points_3d
    
    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        max_distance: float = 50.0
    ) -> List[FeatureMatch]:
        """
        Match descriptors between two sets.
        
        Args:
            desc1: [N, 32] descriptors from frame 1
            desc2: [M, 32] descriptors from frame 2
            max_distance: Maximum Hamming distance
            
        Returns:
            List of FeatureMatch objects
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        if self.cross_check:
            # Simple cross-check matching
            matches = self.bf_matcher.match(desc1, desc2)
            
            # Filter by distance
            good_matches = [
                FeatureMatch(m.queryIdx, m.trainIdx, m.distance)
                for m in matches
                if m.distance < max_distance
            ]
        else:
            # Ratio test matching
            matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
            
            good_matches = []
            for m_list in matches:
                if len(m_list) >= 2:
                    m, n = m_list[0], m_list[1]
                    if m.distance < self.match_ratio * n.distance:
                        if m.distance < max_distance:
                            good_matches.append(
                                FeatureMatch(m.queryIdx, m.trainIdx, m.distance)
                            )
                elif len(m_list) == 1:
                    m = m_list[0]
                    if m.distance < max_distance:
                        good_matches.append(
                            FeatureMatch(m.queryIdx, m.trainIdx, m.distance)
                        )
        
        return good_matches
    
    def match_with_geometric_check(
        self,
        kp1: np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
        desc2: np.ndarray,
        ransac_threshold: float = 3.0
    ) -> Tuple[List[FeatureMatch], np.ndarray, int]:
        """
        Match with RANSAC geometric verification.
        
        Args:
            kp1, desc1: Keypoints and descriptors from frame 1
            kp2, desc2: Keypoints and descriptors from frame 2
            ransac_threshold: RANSAC inlier threshold in pixels
            
        Returns:
            inlier_matches: Geometrically consistent matches
            H: Homography matrix (or None)
            inlier_count: Number of inliers
        """
        # Initial matching
        matches = self.match(desc1, desc2)
        
        if len(matches) < 8:
            return matches, None, len(matches)
        
        # Get matched points
        pts1 = np.array([kp1[m.idx1] for m in matches])
        pts2 = np.array([kp2[m.idx2] for m in matches])
        
        # RANSAC with fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_threshold)
        
        if mask is None:
            return matches, None, len(matches)
        
        # Filter to inliers
        mask = mask.ravel().astype(bool)
        inlier_matches = [m for m, is_inlier in zip(matches, mask) if is_inlier]
        
        return inlier_matches, F, int(mask.sum())


class FeatureTracker:
    """
    Track features across consecutive frames using optical flow.
    """
    
    def __init__(
        self,
        max_corners: int = 500,
        quality_level: float = 0.01,
        min_distance: int = 10,
        block_size: int = 3
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.prev_gray = None
        self.prev_points = None
    
    def track(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Track features from previous frame.
        
        Args:
            image: Current grayscale image
            mask: Optional mask for new feature detection
            
        Returns:
            prev_points: [N, 2] points in previous frame
            curr_points: [N, 2] corresponding points in current frame
            status: [N] tracking status (1 = success)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # First frame - detect initial points
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                mask=mask
            )
            return None, None, None
        
        if self.prev_points is None or len(self.prev_points) == 0:
            # Re-detect features
            self.prev_points = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                mask=mask
            )
            if self.prev_points is None:
                self.prev_gray = gray
                return None, None, None
        
        # Track with optical flow
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        status = status.ravel()
        
        # Filter good tracks
        prev_good = self.prev_points[status == 1].reshape(-1, 2)
        curr_good = curr_points[status == 1].reshape(-1, 2)
        
        # Update state
        self.prev_gray = gray
        
        # Re-detect if too few points
        if len(curr_good) < self.max_corners // 2:
            new_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners - len(curr_good),
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                mask=mask
            )
            if new_points is not None:
                curr_good = np.vstack([curr_good, new_points.reshape(-1, 2)])
        
        self.prev_points = curr_good.reshape(-1, 1, 2).astype(np.float32)
        
        return prev_good, curr_good, status[status == 1]
    
    def reset(self):
        """Reset tracker state."""
        self.prev_gray = None
        self.prev_points = None
