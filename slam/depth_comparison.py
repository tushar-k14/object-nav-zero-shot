"""
Depth Comparison Tool
=====================

Compare depth from different sources:
1. Hardware/Simulator depth (ground truth)
2. Monocular depth estimation (Depth Anything V2, MiDaS)

Metrics:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Scale-Invariant Error
- Edge accuracy
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import time


@dataclass
class DepthMetrics:
    """Depth comparison metrics."""
    mae: float                # Mean Absolute Error
    rmse: float               # Root Mean Square Error
    scale_invariant: float    # Scale-invariant error
    abs_rel: float            # Absolute relative error
    valid_ratio: float        # Ratio of valid pixels
    scale_factor: float       # Estimated scale (mono is relative)
    inference_time_ms: float  # Inference time


class DepthComparator:
    """
    Compare hardware depth with monocular estimation.
    
    Usage:
        comparator = DepthComparator(model='depth_anything_v2')
        
        # Each frame:
        metrics = comparator.compare(rgb, hardware_depth)
        print(f"MAE: {metrics.mae:.3f}m, Scale: {metrics.scale_factor:.2f}")
        
        # Get visualization
        vis = comparator.visualize(rgb, hardware_depth)
    """
    
    def __init__(
        self,
        model: str = 'depth_anything_v2',
        device: str = 'auto'
    ):
        """
        Initialize depth comparator.
        
        Args:
            model: 'depth_anything_v2', 'midas_small', 'midas_large'
            device: 'cpu', 'cuda', or 'auto'
        """
        self.model_name = model
        self.model = None
        self.transform = None
        self.device = device
        
        self._load_model(model)
        
        # History for statistics
        self.metrics_history: list = []
    
    def _load_model(self, model: str):
        """Load depth estimation model."""
        print(f"Loading {model}...")
        
        try:
            import torch
            
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            if model == 'depth_anything_v2':
                try:
                    from transformers import pipeline
                    self.model = pipeline(
                        "depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device=0 if self.device == 'cuda' else -1
                    )
                    print(f"Loaded Depth Anything V2 on {self.device}")
                except Exception as e:
                    print(f"Failed to load Depth Anything V2: {e}")
                    print("Falling back to MiDaS...")
                    self._load_midas('small')
                    
            elif model.startswith('midas'):
                size = 'small' if 'small' in model else 'large'
                self._load_midas(size)
                
        except ImportError as e:
            print(f"Required libraries not available: {e}")
            self.model = None
    
    def _load_midas(self, size: str = 'small'):
        """Load MiDaS model."""
        try:
            import torch
            
            model_name = "MiDaS_small" if size == 'small' else "DPT_Large"
            self.model = torch.hub.load("intel-isl/MiDaS", model_name)
            self.model.eval()
            
            if self.device == 'cuda':
                self.model = self.model.cuda()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if size == 'small':
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.dpt_transform
            
            self.model_name = f'midas_{size}'
            print(f"Loaded MiDaS ({model_name}) on {self.device}")
            
        except Exception as e:
            print(f"Failed to load MiDaS: {e}")
            self.model = None
    
    def estimate_depth(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth from RGB image.
        
        Args:
            rgb: RGB image [H, W, 3]
            
        Returns:
            Estimated depth [H, W] or None
        """
        if self.model is None:
            return None
        
        try:
            import torch
            
            if self.model_name == 'depth_anything_v2':
                from PIL import Image
                pil_img = Image.fromarray(rgb)
                result = self.model(pil_img)
                depth = np.array(result["depth"])
                
                # Resize to match input
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
                
                # Invert if needed (DA outputs inverse depth)
                depth = depth.max() - depth
                
            else:  # MiDaS
                input_batch = self.transform(rgb)
                
                if self.device == 'cuda':
                    input_batch = input_batch.cuda()
                
                with torch.no_grad():
                    prediction = self.model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()
                
                depth = prediction.cpu().numpy()
                
                # MiDaS outputs inverse depth
                depth = 1.0 / (depth + 1e-6)
            
            return depth.astype(np.float32)
            
        except Exception as e:
            print(f"Depth estimation failed: {e}")
            return None
    
    def compare(
        self,
        rgb: np.ndarray,
        gt_depth: np.ndarray,
        min_depth: float = 0.1,
        max_depth: float = 10.0
    ) -> Optional[DepthMetrics]:
        """
        Compare estimated depth with ground truth.
        
        Args:
            rgb: RGB image
            gt_depth: Ground truth depth from hardware/simulator
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth
            
        Returns:
            DepthMetrics or None if estimation fails
        """
        t_start = time.time()
        
        # Estimate depth
        est_depth = self.estimate_depth(rgb)
        
        inference_time = (time.time() - t_start) * 1000
        
        if est_depth is None:
            return None
        
        # Valid mask
        valid = (gt_depth > min_depth) & (gt_depth < max_depth)
        valid_ratio = np.mean(valid)
        
        if valid_ratio < 0.1:
            return None
        
        gt_valid = gt_depth[valid]
        est_valid = est_depth[valid]
        
        # Find optimal scale (monocular depth is up to scale)
        scale_factor = np.median(gt_valid) / (np.median(est_valid) + 1e-6)
        est_scaled = est_valid * scale_factor
        
        # Metrics
        diff = np.abs(gt_valid - est_scaled)
        mae = np.mean(diff)
        rmse = np.sqrt(np.mean(diff ** 2))
        abs_rel = np.mean(diff / (gt_valid + 1e-6))
        
        # Scale-invariant error
        log_diff = np.log(gt_valid + 1e-6) - np.log(est_scaled + 1e-6)
        scale_invariant = np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2)
        
        metrics = DepthMetrics(
            mae=mae,
            rmse=rmse,
            scale_invariant=scale_invariant,
            abs_rel=abs_rel,
            valid_ratio=valid_ratio,
            scale_factor=scale_factor,
            inference_time_ms=inference_time,
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def visualize(
        self,
        rgb: np.ndarray,
        gt_depth: np.ndarray,
        est_depth: np.ndarray = None
    ) -> np.ndarray:
        """
        Create side-by-side visualization.
        
        Args:
            rgb: RGB image
            gt_depth: Ground truth depth
            est_depth: Estimated depth (computed if None)
            
        Returns:
            Visualization image
        """
        if est_depth is None:
            est_depth = self.estimate_depth(rgb)
        
        h, w = rgb.shape[:2]
        
        # Scale estimated depth to GT range
        if est_depth is not None:
            valid = (gt_depth > 0.1) & (gt_depth < 10)
            if np.any(valid):
                scale = np.median(gt_depth[valid]) / (np.median(est_depth[valid]) + 1e-6)
                est_depth = est_depth * scale
        
        # Convert to colormaps
        def depth_to_color(d, max_d=5.0):
            d_norm = np.clip(d / max_d, 0, 1)
            return cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gt_color = depth_to_color(gt_depth)
        
        if est_depth is not None:
            est_color = depth_to_color(est_depth)
            
            # Error map
            error = np.abs(gt_depth - est_depth)
            error_color = depth_to_color(error, max_d=1.0)  # 1m max error
            
            # Combine
            top = np.hstack([rgb_bgr, gt_color])
            bottom = np.hstack([est_color, error_color])
            vis = np.vstack([top, bottom])
            
            # Labels
            cv2.putText(vis, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "GT Depth", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"Est Depth ({self.model_name})", (10, h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "Error", (w + 10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            vis = np.hstack([rgb_bgr, gt_color])
            cv2.putText(vis, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "GT Depth", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def get_summary(self) -> Dict:
        """Get summary statistics from history."""
        if not self.metrics_history:
            return {}
        
        return {
            'n_frames': len(self.metrics_history),
            'mae_mean': np.mean([m.mae for m in self.metrics_history]),
            'mae_std': np.std([m.mae for m in self.metrics_history]),
            'rmse_mean': np.mean([m.rmse for m in self.metrics_history]),
            'abs_rel_mean': np.mean([m.abs_rel for m in self.metrics_history]),
            'scale_factor_mean': np.mean([m.scale_factor for m in self.metrics_history]),
            'inference_time_mean_ms': np.mean([m.inference_time_ms for m in self.metrics_history]),
        }
    
    @property
    def available(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


class DepthSourceSelector:
    """
    Select between depth sources with fallback.
    
    Priority:
    1. Hardware depth (if available and valid)
    2. Monocular estimation (if hardware fails)
    """
    
    def __init__(self, monocular_model: str = 'depth_anything_v2'):
        self.comparator = DepthComparator(model=monocular_model)
        self.use_monocular = False
        self.scale_factor = 1.0  # Learned from valid hardware depth
        
        # Calibration
        self.scale_samples: list = []
        self.calibrated = False
    
    def get_depth(
        self,
        rgb: np.ndarray,
        hardware_depth: np.ndarray = None
    ) -> Tuple[np.ndarray, str]:
        """
        Get best available depth.
        
        Args:
            rgb: RGB image
            hardware_depth: Depth from sensor (may be None or invalid)
            
        Returns:
            (depth, source) where source is 'hardware' or 'monocular'
        """
        # Check hardware depth validity
        if hardware_depth is not None:
            valid_ratio = np.mean((hardware_depth > 0.1) & (hardware_depth < 10))
            
            if valid_ratio > 0.5:
                # Use hardware depth
                
                # Calibrate monocular scale
                if not self.calibrated and self.comparator.available:
                    mono = self.comparator.estimate_depth(rgb)
                    if mono is not None:
                        valid = (hardware_depth > 0.1) & (hardware_depth < 10)
                        scale = np.median(hardware_depth[valid]) / (np.median(mono[valid]) + 1e-6)
                        self.scale_samples.append(scale)
                        
                        if len(self.scale_samples) >= 10:
                            self.scale_factor = np.median(self.scale_samples)
                            self.calibrated = True
                            print(f"Monocular depth calibrated: scale = {self.scale_factor:.3f}")
                
                return hardware_depth, 'hardware'
        
        # Fall back to monocular
        if self.comparator.available:
            mono = self.comparator.estimate_depth(rgb)
            if mono is not None:
                # Apply calibrated scale
                mono = mono * self.scale_factor
                return mono, 'monocular'
        
        # No depth available
        return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32), 'none'
    
    def set_monocular_mode(self, enabled: bool):
        """Force monocular mode (for testing)."""
        self.use_monocular = enabled
