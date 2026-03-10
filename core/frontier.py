"""
Frontier Exploration
====================
Detect, score, and select frontiers for exploration.

A frontier is a boundary between explored free space and unexplored space.
This module provides the detection and scoring logic; path planning is in
core.planning.

Scoring pipeline (pluggable):
    base_score = size / (distance + alpha)
    + semantic_bias  (co-occurring objects hint at room type)
    + clip_score     (CLIP cosine similarity, added in Phase 2)
    + vlm_score      (VLM room classification, added in Phase 3)

Usage:
    from core.frontier import FrontierExplorer, Frontier

    explorer = FrontierExplorer()
    frontiers = explorer.detect(explored_map, obstacle_map)
    best = explorer.select(frontiers, agent_pos=(ax, ay), target='toilet',
                           detected_objects=[('sink', (100, 200))])
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class Frontier:
    """A connected frontier region."""
    id: int
    centroid: Tuple[int, int]       # (x, y) map coords
    size: int                       # number of cells
    cells: np.ndarray = None        # Nx2 array of (x, y), optional (large)
    bbox: Tuple[int, int, int, int] = None  # x1, y1, x2, y2
    score: float = 0.0              # final composite score

    # Per-scorer breakdown (for ablation logging)
    score_base: float = 0.0
    score_semantic: float = 0.0
    score_clip: float = 0.0
    score_vlm: float = 0.0

    def distance_to(self, pos: Tuple[int, int]) -> float:
        return float(np.sqrt(
            (self.centroid[0] - pos[0]) ** 2 +
            (self.centroid[1] - pos[1]) ** 2
        ))


# ============================================================
#  Room co-occurrence heuristic
# ============================================================

# If searching for <key>, seeing <values> nearby boosts frontier score
ROOM_COOCCURRENCE = {
    'toilet':       ['sink', 'bathtub', 'shower', 'mirror'],
    'bed':          ['nightstand', 'lamp', 'dresser', 'pillow', 'clock'],
    'refrigerator': ['oven', 'microwave', 'sink', 'dining table'],
    'couch':        ['tv', 'remote', 'coffee table', 'chair', 'book'],
    'sofa':         ['tv', 'remote', 'coffee table', 'chair', 'couch', 'book'],
    'tv_monitor':   ['couch', 'sofa', 'remote', 'chair', 'book'],
    'tv':           ['couch', 'sofa', 'remote', 'chair', 'book'],
    'dining table': ['chair', 'bowl', 'cup', 'bottle'],
    'plant':        ['vase', 'couch', 'sofa', 'chair', 'dining table'],
    'sink':         ['toilet', 'mirror', 'bathtub'],
    'chair':        ['dining table', 'desk', 'laptop', 'book'],
}


class FrontierExplorer:
    """
    Detect and score frontiers.

    Keeps no state between calls — pass maps each time.
    Designed for easy extension: register custom scorers via add_scorer().
    """

    def __init__(
        self,
        min_size: int = 5,
        distance_weight: float = 10.0,
        semantic_strength: float = 1.5,
        semantic_radius: int = 60,       # pixels
        store_cells: bool = False,       # save memory if False
    ):
        self.min_size = min_size
        self.distance_weight = distance_weight
        self.semantic_strength = semantic_strength
        self.semantic_radius = semantic_radius
        self.store_cells = store_cells

        # Pluggable scorers: list of (name, fn)
        # fn(frontier, context) -> float multiplier (1.0 = no effect)
        self._extra_scorers: List[Tuple[str, Callable]] = []

    # ----------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------

    def detect(
        self,
        explored_map: np.ndarray,
        obstacle_map: np.ndarray,
    ) -> List[Frontier]:
        """
        Detect frontier regions.

        Args:
            explored_map: HxW float, >0.5 = explored
            obstacle_map: HxW float, >0.5 = obstacle

        Returns:
            List of Frontier objects (unscored).
        """
        explored = explored_map > 0.5
        free = obstacle_map < 0.5
        explored_free = explored & free
        unexplored = ~explored

        kernel = np.ones((3, 3), np.uint8)
        near_unexplored = cv2.dilate(unexplored.astype(np.uint8), kernel) > 0

        frontier_mask = explored_free & near_unexplored
        n_labels, labels = cv2.connectedComponents(frontier_mask.astype(np.uint8))

        frontiers = []
        for lbl in range(1, n_labels):
            ys, xs = np.where(labels == lbl)
            size = len(xs)
            if size < self.min_size:
                continue

            cx, cy = int(xs.mean()), int(ys.mean())
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

            f = Frontier(
                id=lbl - 1,
                centroid=(cx, cy),
                size=size,
                bbox=bbox,
            )
            if self.store_cells:
                f.cells = np.stack([xs, ys], axis=1)

            frontiers.append(f)

        return frontiers

    def score(
        self,
        frontiers: List[Frontier],
        agent_pos: Tuple[int, int],
        target: str = None,
        detected_objects: List[Tuple[str, Tuple[int, int]]] = None,
        context: dict = None,
    ) -> List[Frontier]:
        """
        Score frontiers using weighted combination of geometric and semantic signals.

        Scoring formula:
            score = w_geo * norm_base + w_sem * semantic + w_clip * clip + w_vlm * vlm

        This additive approach ensures CLIP/VLM signals have real influence
        even when geometric scores (size/distance) vary widely.
        """
        target_lower = target.lower().strip() if target else None
        context = context or {}

        # ---- Phase 1: compute raw geometric scores ----
        for f in frontiers:
            dist = f.distance_to(agent_pos)
            f.score_base = f.size / (dist + self.distance_weight)

        # ---- Phase 2: normalize base scores to [0, 1] ----
        if frontiers:
            base_scores = [f.score_base for f in frontiers]
            max_base = max(base_scores)
            if max_base > 0:
                for f in frontiers:
                    f._norm_base = f.score_base / max_base
            else:
                for f in frontiers:
                    f._norm_base = 0.0
        
        # ---- Phase 3: compute semantic + custom scores ----
        for f in frontiers:
            # Semantic co-occurrence
            f.score_semantic = 0.0
            if target_lower and detected_objects:
                raw_sem = self._semantic_bias(f, target_lower, detected_objects)
                f.score_semantic = (raw_sem - 1.0)  # Convert from multiplier to 0-based

            # Custom scorers (CLIP, VLM, etc.)
            f.score_clip = 0.0
            f.score_vlm = 0.0
            for name, scorer_fn in self._extra_scorers:
                try:
                    mult = scorer_fn(f, context)
                except Exception:
                    mult = 1.0
                # Convert from multiplier (0.5-2.5) to normalized (0.0-1.0)
                # 0.5 → 0.0, 1.0 → 0.25, 1.5 → 0.5, 2.5 → 1.0
                normalized = np.clip((mult - 0.5) / 2.0, 0.0, 1.0)
                if name == 'clip':
                    f.score_clip = normalized
                elif name == 'vlm':
                    f.score_vlm = normalized
                # For non-FOV frontiers: mult=1.0 → normalized=0.25
                # This is fine: CLIP "don't know" = 0.25, "looks promising" = 0.5-1.0

            # ---- Phase 4: weighted combination ----
            # Weights control influence of each signal
            W_GEO  = 0.4    # Geometric (size/distance) — always active
            W_SEM  = 0.2    # Semantic co-occurrence — if objects detected
            W_CLIP = 0.25   # CLIP vision-language — if CLIP loaded
            W_VLM  = 0.15   # VLM room context — if VLM available

            f.score = (
                W_GEO  * f._norm_base +
                W_SEM  * f.score_semantic +
                W_CLIP * f.score_clip +
                W_VLM  * f.score_vlm
            )

        frontiers.sort(key=lambda x: x.score, reverse=True)

        # Store for diagnostics (used by adaptive VLM trigger)
        self._last_scored_frontiers = frontiers

        return frontiers

    def select(
        self,
        frontiers: List[Frontier] = None,
        agent_pos: Tuple[int, int] = None,
        target: str = None,
        detected_objects: List[Tuple[str, Tuple[int, int]]] = None,
        blacklist: set = None,
        explored_map: np.ndarray = None,
        obstacle_map: np.ndarray = None,
        context: dict = None,
    ) -> Optional[Frontier]:
        """
        Convenience: detect + score + pick best non-blacklisted.

        Can pass pre-detected frontiers, or pass maps to detect fresh.
        """
        if frontiers is None:
            if explored_map is None or obstacle_map is None:
                return None
            frontiers = self.detect(explored_map, obstacle_map)

        if not frontiers:
            return None

        if agent_pos is not None:
            frontiers = self.score(
                frontiers, agent_pos, target, detected_objects, context
            )

        blacklist = blacklist or set()
        for f in frontiers:
            if f.id not in blacklist:
                return f
        return None

    def add_scorer(self, name: str, fn: Callable):
        """
        Register a custom scorer.

        fn(frontier: Frontier, context: dict) -> float
            Return a multiplier (1.0 = no change, >1 = boost, <1 = penalize).
        """
        self._extra_scorers.append((name, fn))

    # ----------------------------------------------------------
    #  Internal
    # ----------------------------------------------------------

    def _semantic_bias(
        self,
        frontier: Frontier,
        target: str,
        detected_objects: List[Tuple[str, Tuple[int, int]]],
    ) -> float:
        """Boost if co-occurring objects are near this frontier."""
        hints = ROOM_COOCCURRENCE.get(target, [])
        if not hints:
            return 1.0

        best = 1.0
        for obj_class, obj_pos in detected_objects:
            if obj_class.lower() in hints:
                dist = frontier.distance_to(obj_pos)
                if dist < self.semantic_radius:
                    strength = self.semantic_strength * (
                        1.0 - dist / self.semantic_radius
                    )
                    best = max(best, 1.0 + strength)

        return best

    # ----------------------------------------------------------
    #  Visualization
    # ----------------------------------------------------------

    def visualize(
        self,
        base_map: np.ndarray,
        frontiers: List[Frontier],
        agent_pos: Tuple[int, int] = None,
        highlight_id: int = None,
    ) -> np.ndarray:
        """Draw frontiers on map image."""
        vis = (
            cv2.cvtColor(base_map, cv2.COLOR_GRAY2BGR)
            if len(base_map.shape) == 2
            else base_map.copy()
        )

        for f in frontiers:
            color = (0, 255, 255) if f.id == highlight_id else (0, 165, 255)
            cv2.circle(vis, f.centroid, max(3, int(np.sqrt(f.size))), color, -1)
            cv2.putText(
                vis,
                f"{f.id}:{f.score:.1f}",
                (f.centroid[0] + 5, f.centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )

        if agent_pos:
            cv2.circle(vis, agent_pos, 6, (255, 0, 0), -1)

        return vis


# ============================================================
#  Exploration State Manager
# ============================================================

class ExplorationManager:
    """
    Stateful wrapper: tracks current target, blacklist, replan timing.

    Usage:
        mgr = ExplorationManager()
        target_xy = mgr.update(explored, obstacles, agent_pos, 'toilet', dets)
        if target_xy is None:
            # exploration complete
    """

    def __init__(
        self,
        explorer: FrontierExplorer = None,
        replan_interval: int = 40,
        reach_threshold: int = 15,
    ):
        self.explorer = explorer or FrontierExplorer()
        self.replan_interval = replan_interval
        self.reach_threshold = reach_threshold

        self.current_target: Optional[Frontier] = None
        self.steps_since_replan = 0
        self.blacklist: set = set()
        self.exploration_complete = False

    def update(
        self,
        explored_map: np.ndarray,
        obstacle_map: np.ndarray,
        agent_pos: Tuple[int, int],
        target_object: str = None,
        detected_objects: List[Tuple[str, Tuple[int, int]]] = None,
        context: dict = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Get next exploration target. Returns (x, y) or None if done.
        """
        self.steps_since_replan += 1

        need_replan = (
            self.current_target is None
            or self.steps_since_replan >= self.replan_interval
            or self._reached(agent_pos)
        )

        if need_replan:
            self.steps_since_replan = 0
            frontiers = self.explorer.detect(explored_map, obstacle_map)

            self.current_target = self.explorer.select(
                frontiers=frontiers,
                agent_pos=agent_pos,
                target=target_object,
                detected_objects=detected_objects,
                blacklist=self.blacklist,
                context=context,
            )

            if self.current_target is None:
                # Try clearing blacklist
                self.blacklist.clear()
                self.current_target = self.explorer.select(
                    frontiers=frontiers,
                    agent_pos=agent_pos,
                    target=target_object,
                    detected_objects=detected_objects,
                    context=context,
                )

            if self.current_target is None:
                self.exploration_complete = True
                return None

        return self.current_target.centroid if self.current_target else None

    def _reached(self, agent_pos: Tuple[int, int]) -> bool:
        if self.current_target is None:
            return True
        if self.current_target.distance_to(agent_pos) < self.reach_threshold:
            self.blacklist.add(self.current_target.id)
            return True
        return False

    def reset(self):
        self.current_target = None
        self.steps_since_replan = 0
        self.blacklist.clear()
        self.exploration_complete = False