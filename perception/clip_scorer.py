"""
CLIP Frontier Scoring
=====================

Scores exploration frontiers using CLIP cosine similarity between:
  - Image crops from the agent's RGB observation near each frontier direction
  - Text embedding of the target object description

Inspired by VLFM (Yokoyama et al., 2024) but using lightweight CLIP-ViT-B/32
instead of BLIP-2 to stay within 24GB GPU budget alongside YOLO.

Architecture:
    Each frontier has a direction from the agent. We crop the RGB image
    in that direction and compute CLIP similarity with target text.
    High similarity = frontier likely leads toward target.

    For frontiers outside current FOV, we use a VLM room classifier
    (Phase 3) or fall back to semantic co-occurrence heuristic.

Usage:
    from perception.clip_scorer import CLIPFrontierScorer

    scorer = CLIPFrontierScorer()
    # Register with frontier explorer:
    explorer.add_scorer('clip', scorer.score_frontier)
    # Or use directly:
    scores = scorer.score_frontiers(frontiers, rgb, agent_pos, agent_yaw, target, occ)

Integration with eval:
    Pass via context dict:
        context = {'clip_scorer': scorer, 'rgb': rgb, 'agent_yaw': yaw}
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import time


class CLIPFrontierScorer:
    """
    Score frontiers using CLIP image-text similarity.

    Loads CLIP model once, caches text embeddings per target.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = None,
        cache_text: bool = True,
        fov_degrees: float = 90.0,
        score_boost: float = 2.0,
        min_score: float = 0.5,
    ):
        """
        Args:
            model_name: CLIP model variant
            device: 'cuda' or 'cpu'
            cache_text: cache text embeddings for repeated targets
            fov_degrees: camera horizontal FOV
            score_boost: max multiplier for high-similarity frontiers
            min_score: minimum CLIP multiplier (1.0 = no penalty)
        """
        self.model_name = model_name
        self.device = device
        self.cache_text = cache_text
        self.fov_degrees = fov_degrees
        self.score_boost = score_boost
        self.min_score = min_score

        self._model = None
        self._preprocess = None
        self._text_cache = {}
        self._loaded = False

    def load(self):
        """Load CLIP model. Tries open_clip first, then openai clip."""
        if self._loaded:
            return

        import torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try open_clip_torch first (better PyTorch compatibility)
        try:
            import open_clip
            self._backend = 'open_clip'

            # Map model names: OpenAI format -> open_clip format
            model_map = {
                'ViT-B/32': ('ViT-B-32', 'openai'),
                'ViT-B/16': ('ViT-B-16', 'openai'),
                'ViT-L/14': ('ViT-L-14', 'openai'),
            }
            if self.model_name in model_map:
                arch, pretrained = model_map[self.model_name]
            else:
                arch, pretrained = self.model_name, 'openai'

            print(f"Loading CLIP {arch} (open_clip, {pretrained}) on {self.device}...")
            t0 = time.time()
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained, device=self.device,
            )
            self._tokenizer = open_clip.get_tokenizer(arch)
            self._model.eval()
            print(f"  CLIP loaded in {time.time() - t0:.1f}s")
            self._loaded = True
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"  open_clip failed: {e}")

        # Fallback: OpenAI CLIP
        try:
            import clip
            self._backend = 'openai_clip'

            print(f"Loading CLIP {self.model_name} (openai) on {self.device}...")
            t0 = time.time()
            self._model, self._preprocess = clip.load(self.model_name, device=self.device)
            self._tokenizer = None  # openai clip uses clip.tokenize()
            self._model.eval()
            print(f"  CLIP loaded in {time.time() - t0:.1f}s")
            self._loaded = True
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"  openai clip failed: {e}")

        print("WARNING: No CLIP backend available.")
        print("  Install one of:")
        print("    pip install open_clip_torch")
        print("    pip install git+https://github.com/openai/CLIP.git")
        self._loaded = True

    @property
    def available(self) -> bool:
        self.load()
        return self._model is not None

    # ----------------------------------------------------------
    # Text embedding (cached)
    # ----------------------------------------------------------

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get normalized text embedding, cached."""
        if not self.available:
            return None

        if text in self._text_cache:
            return self._text_cache[text]

        import torch

        with torch.no_grad():
            if self._backend == 'open_clip':
                tokens = self._tokenizer([text]).to(self.device)
                emb = self._model.encode_text(tokens)
            else:
                import clip
                tokens = clip.tokenize([text]).to(self.device)
                emb = self._model.encode_text(tokens)

            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_np = emb.cpu().numpy()[0]

        if self.cache_text:
            self._text_cache[text] = emb_np
        return emb_np

    # ----------------------------------------------------------
    # Image embedding
    # ----------------------------------------------------------

    def _get_image_embedding(self, image_crop: np.ndarray) -> Optional[np.ndarray]:
        """Get normalized image embedding from RGB crop."""
        if not self.available:
            return None

        import torch
        from PIL import Image

        # Convert numpy to PIL
        if image_crop.dtype != np.uint8:
            image_crop = (image_crop * 255).astype(np.uint8)
        pil_img = Image.fromarray(image_crop)

        # Preprocess and encode
        img_input = self._preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._model.encode_image(img_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    # ----------------------------------------------------------
    # Frontier scoring
    # ----------------------------------------------------------

    def _frontier_to_bearing(
        self,
        frontier_centroid: Tuple[int, int],
        agent_map_pos: Tuple[int, int],
    ) -> float:
        """
        Compute bearing angle from agent to frontier in map coords.
        Returns angle in radians (0 = forward, positive = left).
        """
        dx = frontier_centroid[0] - agent_map_pos[0]
        dy = frontier_centroid[1] - agent_map_pos[1]
        # Map coords: +x = right, +y = down (image convention)
        angle = np.arctan2(-dx, -dy)  # 0 = up in image = forward
        return float(angle)

    def _get_frontier_crop(
        self,
        rgb: np.ndarray,
        frontier_bearing: float,
        agent_yaw: float,
        crop_width_frac: float = 0.25,
    ) -> Optional[np.ndarray]:
        """
        Extract RGB crop in the direction of the frontier.

        Maps frontier bearing to a horizontal slice of the RGB image.
        Returns None if frontier is outside camera FOV.
        """
        # Relative angle: frontier bearing relative to agent heading
        rel_angle = frontier_bearing - agent_yaw
        # Normalize to [-pi, pi]
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

        # Check if within FOV
        half_fov = np.radians(self.fov_degrees / 2)
        if abs(rel_angle) > half_fov:
            return None  # Outside FOV

        # Map angle to horizontal pixel position
        h, w = rgb.shape[:2]
        # rel_angle=0 -> center, rel_angle=-half_fov -> left edge
        frac = 0.5 - (rel_angle / (2 * half_fov))
        center_x = int(frac * w)

        # Crop window
        crop_w = int(w * crop_width_frac)
        x1 = max(0, center_x - crop_w // 2)
        x2 = min(w, x1 + crop_w)
        x1 = max(0, x2 - crop_w)  # Adjust if hit right edge

        # Vertical: take middle portion (skip sky/floor)
        y1 = h // 6
        y2 = 5 * h // 6

        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def score_frontier(self, frontier, context: dict) -> float:
        """
        Scorer function compatible with FrontierExplorer.add_scorer().

        OPTIMIZED: Uses a single full-frame CLIP inference per step (cached),
        then distributes score based on frontier direction relative to agent.
        
        Returns:
            float multiplier (1.0 = no change, >1 = boost, <1 = penalize)
        """
        if not self.available:
            return 1.0

        rgb = context.get('rgb')
        agent_pos = context.get('agent_map_pos')
        agent_yaw = context.get('agent_yaw')
        target = context.get('target')

        if rgb is None or agent_pos is None or agent_yaw is None or target is None:
            return 1.0

        # Cache per-step: compute directional scores periodically
        # Running CLIP 3× per step is expensive (~100ms). Recompute every 5 steps.
        step = context.get('step', 0)
        frame_key = step // 5  # Same scores for 5 consecutive steps
        if not hasattr(self, '_dir_cache_key') or self._dir_cache_key != frame_key:
            self._dir_cache_key = frame_key
            self._dir_scores = self._compute_directional_scores(rgb, agent_yaw, target)

        # Look up score for this frontier's direction
        bearing = self._frontier_to_bearing(frontier.centroid, agent_pos)
        rel_angle = bearing - agent_yaw
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

        half_fov = np.radians(self.fov_degrees / 2)

        if abs(rel_angle) <= half_fov:
            # Map to one of the directional bins
            frac = (rel_angle + half_fov) / (2 * half_fov)  # 0=left, 1=right
            bin_idx = min(int(frac * len(self._dir_scores)), len(self._dir_scores) - 1)
            return self._dir_scores[bin_idx]

        # Outside FOV: return neutral (no CLIP signal)
        return 1.0

    def _compute_directional_scores(
        self, rgb: np.ndarray, agent_yaw: float, target: str
    ) -> List[float]:
        """
        Compute CLIP scores for 3 directional slices of the current view.
        Only 3 CLIP inference calls per step (left, center, right).

        Returns list of 3 multipliers.
        """
        h, w = rgb.shape[:2]
        thirds = [
            rgb[h//6:5*h//6, 0:w//3],         # Left third
            rgb[h//6:5*h//6, w//3:2*w//3],     # Center third
            rgb[h//6:5*h//6, 2*w//3:w],        # Right third
        ]

        scores = []
        raw_sims = []
        for crop in thirds:
            if crop.size == 0:
                scores.append(1.0)
                raw_sims.append(0.0)
                continue
            # Get raw similarity for debugging
            img_emb = self._get_image_embedding(crop)
            if img_emb is not None:
                prompts = [f"a photo of a {target}", f"a room with a {target}"]
                sims = []
                for p in prompts:
                    te = self._get_text_embedding(p)
                    if te is not None:
                        sims.append(float(np.dot(img_emb, te)))
                raw_sims.append(np.mean(sims) if sims else 0.0)
            else:
                raw_sims.append(0.0)

            s = self._compute_clip_score(crop, target)
            scores.append(s if s is not None else 1.0)

        # Log raw sims periodically (every ~50 calls)
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 50 == 1:
            print(f"    [CLIP raw] target={target} sims=[{raw_sims[0]:.3f}, {raw_sims[1]:.3f}, {raw_sims[2]:.3f}] "
                  f"mults=[{scores[0]:.2f}, {scores[1]:.2f}, {scores[2]:.2f}]")

        return scores

    def _compute_clip_score(self, crop: np.ndarray, target: str) -> Optional[float]:
        """Compute CLIP similarity and map to multiplier.
        
        Uses absolute similarity (not contrastive) because CLIP cosine sims
        for indoor scenes are tightly clustered (0.20-0.28). Contrastive
        pos-neg differences are too small to be useful.
        """
        img_emb = self._get_image_embedding(crop)
        if img_emb is None:
            return None

        # Multiple prompts for robustness
        prompts = [
            f"a photo of a {target}",
            f"a room with a {target}",
            f"a {target} in a house",
        ]

        sims = []
        for prompt in prompts:
            text_emb = self._get_text_embedding(prompt)
            if text_emb is not None:
                sims.append(float(np.dot(img_emb, text_emb)))

        if not sims:
            return None

        avg_sim = np.mean(sims)

        # Map absolute similarity to multiplier
        # CLIP ViT-B/32 indoor scene sims typically range 0.18-0.30
        # 0.20 → low (0.5×), 0.24 → neutral (1.0×), 0.28+ → boost (2.5×)
        normalized = np.clip((avg_sim - 0.20) / 0.08, 0.0, 1.0)
        multiplier = self.min_score + normalized * (self.score_boost - self.min_score)
        return float(multiplier)

    def score_frontiers_batch(
        self,
        frontiers,
        rgb: np.ndarray,
        agent_map_pos: Tuple[int, int],
        agent_yaw: float,
        target: str,
    ) -> Dict[int, float]:
        """
        Score all frontiers at once (more efficient for many frontiers).

        Returns:
            Dict mapping frontier.id -> CLIP multiplier
        """
        if not self.available:
            return {f.id: 1.0 for f in frontiers}

        scores = {}
        # Get text embedding once
        text_embs = []
        for prompt in [f"a photo of a {target}", f"a room containing a {target}"]:
            emb = self._get_text_embedding(prompt)
            if emb is not None:
                text_embs.append(emb)

        if not text_embs:
            return {f.id: 1.0 for f in frontiers}

        for f in frontiers:
            bearing = self._frontier_to_bearing(f.centroid, agent_map_pos)
            crop = self._get_frontier_crop(rgb, bearing, agent_yaw)

            if crop is None:
                scores[f.id] = 1.0
                continue

            img_emb = self._get_image_embedding(crop)
            if img_emb is None:
                scores[f.id] = 1.0
                continue

            sims = [float(np.dot(img_emb, te)) for te in text_embs]
            avg_sim = np.mean(sims)

            normalized = np.clip((avg_sim - 0.20) / 0.10, 0.0, 1.0)
            multiplier = self.min_score + normalized * (self.score_boost - self.min_score)
            scores[f.id] = float(multiplier)

        return scores


class VLMRoomClassifier:
    """
    Phase 3: Use local VLM (Ollama) to classify what room type
    a frontier likely leads to. Only called for high-value frontiers
    or when CLIP signal is ambiguous.

    Uses Ollama with LLaVA for local inference.
    """

    # Which rooms typically contain which objects
    OBJECT_ROOM_PRIORS = {
        'bed':        ['bedroom'],
        'toilet':     ['bathroom'],
        'sofa':       ['living_room', 'lounge'],
        'tv_monitor': ['living_room', 'bedroom'],
        'chair':      ['dining_room', 'living_room', 'office', 'bedroom'],
        'plant':      ['living_room', 'hallway', 'entrance'],
        'sink':       ['kitchen', 'bathroom'],
        'refrigerator': ['kitchen'],
    }

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llava:7b",
        room_boost: float = 2.5,
        query_interval: int = 50,
        timeout: float = 10.0,
    ):
        """
        Args:
            ollama_host: Ollama API endpoint
            model: Ollama model name
            room_boost: multiplier when room matches target
            query_interval: min steps between VLM queries (rate limiting)
            timeout: API timeout in seconds
        """
        self.ollama_host = ollama_host
        self.model = model
        self.room_boost = room_boost
        self.query_interval = query_interval
        self.timeout = timeout
        self._last_query_step = -999
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import requests
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def classify_room(
        self,
        rgb: np.ndarray,
        step: int = 0,
    ) -> Optional[str]:
        """
        Classify the room visible in the current RGB frame.

        Returns room type string or None if unavailable/rate-limited.
        """
        if not self.available:
            return None
        if step - self._last_query_step < self.query_interval:
            return None

        self._last_query_step = step

        try:
            import requests
            import base64
            import cv2

            # Encode image
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buf).decode('utf-8')

            prompt = (
                "Look at this image. What room is this? "
                "Reply with exactly one word from this list: "
                "bedroom, bathroom, kitchen, living_room, dining_room, "
                "hallway, office, garage, laundry, closet"
            )

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 30},
            }

            # Longer timeout for first call (model loading)
            timeout = 60.0 if self._last_query_step <= 0 else self.timeout

            r = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=timeout,
            )

            if r.status_code == 200:
                text = r.json().get('response', '').lower().strip()

                # Flexible parsing: map various phrasings to room types
                ROOM_ALIASES = {
                    'bedroom': 'bedroom',
                    'bed room': 'bedroom',
                    'sleeping': 'bedroom',
                    'bathroom': 'bathroom',
                    'bath room': 'bathroom',
                    'restroom': 'bathroom',
                    'washroom': 'bathroom',
                    'toilet': 'bathroom',
                    'kitchen': 'kitchen',
                    'living room': 'living_room',
                    'living_room': 'living_room',
                    'lounge': 'living_room',
                    'family room': 'living_room',
                    'sitting room': 'living_room',
                    'den': 'living_room',
                    'dining room': 'dining_room',
                    'dining_room': 'dining_room',
                    'dining': 'dining_room',
                    'hallway': 'hallway',
                    'hall': 'hallway',
                    'corridor': 'hallway',
                    'entrance': 'hallway',
                    'foyer': 'hallway',
                    'entryway': 'hallway',
                    'office': 'office',
                    'study': 'office',
                    'workspace': 'office',
                    'garage': 'garage',
                    'laundry': 'laundry',
                    'closet': 'closet',
                    'storage': 'closet',
                }

                # Try exact room words first, then check aliases
                for alias, room in ROOM_ALIASES.items():
                    if alias in text:
                        return room

                # Debug: log unrecognized responses
                print(f"    [VLM raw] '{text[:80]}'")

        except requests.exceptions.Timeout:
            print(f"    [VLM] timeout at step {step}")
        except Exception as e:
            print(f"    [VLM] error: {e}")
        return None

    def score_frontier(self, frontier, context: dict) -> float:
        """
        VLM scorer compatible with FrontierExplorer.add_scorer().

        Uses room classification to boost frontiers that likely lead
        to rooms containing the target object.
        """
        target = context.get('target')
        current_room = context.get('current_room')

        if not target or not current_room:
            return 1.0

        # Check if current room is likely to contain target
        likely_rooms = self.OBJECT_ROOM_PRIORS.get(target, [])

        if current_room in likely_rooms:
            # We're already in a promising room - boost nearby frontiers
            return self.room_boost

        return 1.0