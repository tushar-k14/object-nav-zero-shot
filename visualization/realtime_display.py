"""
Real-time Visualization
=======================

Real-time display of exploration with:
1. RGB + Depth + Segmentation + Map view
2. MP4 video recording
3. Statistics overlay

Uses matplotlib for display (works without GTK) and cv2 for recording.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import time

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend which is more portable
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available for display")


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    window_width: int = 1280
    window_height: int = 720
    panel_rows: int = 2
    panel_cols: int = 2
    fps: int = 30
    font_scale: float = 0.5
    show_stats: bool = True


class RealtimeDisplay:
    """
    Real-time visualization with recording capability.
    
    Layout (2x2):
    +----------------+----------------+
    |      RGB       |     Depth      |
    +----------------+----------------+
    |   Detection    |      Map       |
    +----------------+----------------+
    """
    
    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        window_name: str = "ObjectNav Explorer",
        use_matplotlib: bool = True
    ):
        """
        Initialize display.
        
        Args:
            config: Visualization configuration
            window_name: Window name
            use_matplotlib: Use matplotlib (True) or OpenCV (False)
        """
        self.config = config or VisualizationConfig()
        self.window_name = window_name
        self.use_matplotlib = use_matplotlib and MATPLOTLIB_AVAILABLE
        
        # Panel dimensions
        self.panel_w = self.config.window_width // self.config.panel_cols
        self.panel_h = self.config.window_height // self.config.panel_rows
        
        # Recording
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_path: Optional[str] = None
        self.frame_count = 0
        self.record_start_time = 0
        
        # Matplotlib setup
        if self.use_matplotlib:
            plt.ion()  # Interactive mode
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 9))
            self.fig.canvas.manager.set_window_title(window_name)
            
            # Set titles
            self.axes[0, 0].set_title('RGB')
            self.axes[0, 1].set_title('Depth')
            self.axes[1, 0].set_title('Detection')
            self.axes[1, 1].set_title('Map')
            
            for ax in self.axes.flat:
                ax.axis('off')
            
            plt.tight_layout()
            
            # Image handles for updating
            self.im_handles = [[None, None], [None, None]]
            self.stats_text = None
            
            # Key press handling
            self.last_key = None
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            self.fig.canvas.mpl_connect('close_event', self._on_close)
            self.window_closed = False
        else:
            # Try OpenCV
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 
                                self.config.window_width, 
                                self.config.window_height)
            except cv2.error as e:
                print(f"OpenCV window error: {e}")
                print("Falling back to headless mode (recording only)")
                self.use_matplotlib = False
    
    def _on_key_press(self, event):
        """Handle matplotlib key press."""
        if event.key:
            self.last_key = event.key
    
    def _on_close(self, event):
        """Handle window close."""
        self.window_closed = True
    
    def create_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        detection_vis: np.ndarray,
        map_vis: np.ndarray,
        stats: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Create composite visualization frame.
        
        Returns:
            Composite frame (window_height x window_width x 3)
        """
        # Resize all panels
        rgb_panel = cv2.resize(rgb, (self.panel_w, self.panel_h))
        
        # Depth visualization
        depth_norm = np.clip(depth / 10.0, 0, 1)
        depth_colored = cv2.applyColorMap(
            (depth_norm * 255).astype(np.uint8), 
            cv2.COLORMAP_VIRIDIS
        )
        depth_panel = cv2.resize(depth_colored, (self.panel_w, self.panel_h))
        depth_panel = cv2.cvtColor(depth_panel, cv2.COLOR_BGR2RGB)
        
        det_panel = cv2.resize(detection_vis, (self.panel_w, self.panel_h))
        map_panel = cv2.resize(map_vis, (self.panel_w, self.panel_h))
        
        # Compose
        top_row = np.hstack([rgb_panel, depth_panel])
        bottom_row = np.hstack([det_panel, map_panel])
        frame = np.vstack([top_row, bottom_row])
        
        # Add stats overlay
        if stats and self.config.show_stats:
            frame = self._add_stats_overlay(frame, stats)
        
        # Add recording indicator
        if self.is_recording:
            frame = self._add_recording_indicator(frame)
        
        return frame
    
    def _add_stats_overlay(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """Add statistics overlay to frame."""
        lines = [
            f"Frame: {stats.get('frame', stats.get('step', 0))}",
            f"Explored: {stats.get('explored_percent', 0):.1f}%",
            f"Objects: {stats.get('tracked_objects', 0)}",
            f"FPS: {stats.get('fps', 0):.1f}"
        ]
        
        y = 30
        for line in lines:
            cv2.putText(frame, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                       (0, 255, 0), 1)
            y += 20
        
        return frame
    
    def _add_recording_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Add recording indicator."""
        h, w = frame.shape[:2]
        cv2.circle(frame, (w - 30, 20), 8, (255, 0, 0), -1)
        cv2.putText(frame, "REC", (w - 70, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        elapsed = time.time() - self.record_start_time
        time_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        cv2.putText(frame, time_str, (w - 70, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def show(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        detection_vis: np.ndarray,
        map_vis: np.ndarray,
        stats: Optional[Dict] = None
    ) -> int:
        """
        Display frame and handle input.
        
        Returns:
            Key code pressed (or -1 if none)
        """
        # Create composite frame for recording
        frame = self.create_frame(rgb, depth, detection_vis, map_vis, stats)
        
        # Record if active
        if self.is_recording and self.video_writer is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)
            self.frame_count += 1
        
        key = -1
        
        if self.use_matplotlib and not self.window_closed:
            # Depth visualization for matplotlib
            depth_norm = np.clip(depth / 10.0, 0, 1)
            
            # Update images
            panels = [
                (0, 0, rgb),
                (0, 1, depth_norm),
                (1, 0, detection_vis),
                (1, 1, map_vis)
            ]
            
            for i, j, img in panels:
                if self.im_handles[i][j] is None:
                    if j == 1 and i == 0:  # Depth
                        self.im_handles[i][j] = self.axes[i, j].imshow(
                            img, cmap='viridis', vmin=0, vmax=1
                        )
                    else:
                        self.im_handles[i][j] = self.axes[i, j].imshow(img)
                else:
                    self.im_handles[i][j].set_data(img)
            
            # Update stats text
            if stats:
                stats_str = (f"Step: {stats.get('step', 0)} | "
                           f"Explored: {stats.get('explored_percent', 0):.1f}% | "
                           f"Objects: {stats.get('tracked_objects', 0)} | "
                           f"FPS: {stats.get('fps', 0):.1f}")
                if self.is_recording:
                    stats_str += " | 🔴 REC"
                
                self.fig.suptitle(stats_str, fontsize=10)
            
            # Refresh
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            
            # Get key
            if self.last_key:
                key = ord(self.last_key[0]) if len(self.last_key) == 1 else -1
                self.last_key = None
            
            if self.window_closed:
                key = ord('q')
        
        elif not self.use_matplotlib:
            # OpenCV display (may not work)
            try:
                cv2.imshow(self.window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
            except:
                pass
        
        return key
    
    def start_recording(self, output_path: str = "exploration_recording.mp4"):
        """Start video recording."""
        if self.is_recording:
            print("Already recording!")
            return
        
        self.record_path = output_path
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.config.fps,
            (self.config.window_width, self.config.window_height)
        )
        
        self.is_recording = True
        self.frame_count = 0
        self.record_start_time = time.time()
        
        print(f"Started recording to {output_path}")
    
    def stop_recording(self):
        """Stop video recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        elapsed = time.time() - self.record_start_time
        print(f"Stopped recording. Saved {self.frame_count} frames ({elapsed:.1f}s) to {self.record_path}")
    
    def toggle_recording(self, output_path: str = "exploration_recording.mp4"):
        """Toggle recording on/off."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording(output_path)
    
    def close(self):
        """Close display and cleanup."""
        self.stop_recording()
        
        if self.use_matplotlib:
            plt.close(self.fig)
        else:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass


class InteractiveExplorer:
    """
    Interactive exploration with keyboard control.
    
    Controls:
    - W/↑: Move forward
    - S/↓: Move backward  
    - A/←: Turn left
    - D/→: Turn right
    - R: Toggle recording
    - Q/ESC: Quit
    - SPACE: Pause
    """
    
    ACTION_MAP = {
        ord('w'): 'move_forward',
        ord('s'): 'move_backward',
        ord('a'): 'turn_left',
        ord('d'): 'turn_right',
        82: 'move_forward',   # Up arrow
        84: 'move_backward',  # Down arrow
        81: 'turn_left',      # Left arrow
        83: 'turn_right',     # Right arrow
    }
    
    def __init__(
        self,
        sim,  # Habitat simulator
        semantic_map,  # SemanticMap instance
        intrinsics,  # CameraIntrinsics
        display: Optional[RealtimeDisplay] = None,
        record_path: str = "exploration.mp4"
    ):
        """
        Initialize interactive explorer.
        """
        self.sim = sim
        self.semantic_map = semantic_map
        self.intrinsics = intrinsics
        self.display = display or RealtimeDisplay()
        self.record_path = record_path
        
        self.running = False
        self.paused = False
        self.step_count = 0
        self.last_time = time.time()
        self.fps = 0
    
    def get_agent_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get agent position and rotation."""
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        
        position = np.array(state.position)
        rotation = np.array([
            state.rotation.x,
            state.rotation.y,
            state.rotation.z,
            state.rotation.w
        ])
        
        return position, rotation
    
    def step(self, action: str) -> Dict:
        """Take action and update map."""
        self.sim.step(action)
        self.step_count += 1
        
        obs = self.sim.get_sensor_observations()
        rgb = obs['color_sensor'][:, :, :3]
        depth = obs['depth_sensor']
        
        position, rotation = self.get_agent_state()
        
        stats = self.semantic_map.update(
            rgb, depth, position, rotation, self.intrinsics
        )
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.fps = 1.0 / dt if dt > 0 else 0
        self.last_time = current_time
        
        return {
            'rgb': rgb,
            'depth': depth,
            'position': position,
            'rotation': rotation,
            'stats': stats,
            'step': self.step_count,
            'fps': self.fps
        }
    
    def run(self):
        """Run interactive exploration loop."""
        self.running = True
        print("\n" + "="*50)
        print("Interactive Exploration")
        print("="*50)
        print("Controls:")
        print("  W: Forward    S: Backward")
        print("  A: Left       D: Right")
        print("  R: Toggle recording")
        print("  SPACE: Pause")
        print("  Q: Quit")
        print("="*50 + "\n")
        
        # Initial observation
        obs = self.sim.get_sensor_observations()
        rgb = obs['color_sensor'][:, :, :3]
        depth = obs['depth_sensor']
        position, rotation = self.get_agent_state()
        
        while self.running:
            # Get current detections for visualization
            detections, obj_ids = self.semantic_map.get_detections(
                rgb, depth, position, rotation, self.intrinsics
            )
            
            # Create visualizations
            det_vis = self.semantic_map.visualize_detection(rgb, detections, obj_ids)
            map_vis = self.semantic_map.visualize()
            
            # Get stats
            stats = self.semantic_map.get_statistics()
            stats['fps'] = self.fps
            stats['step'] = self.step_count
            stats['paused'] = self.paused
            
            # Display
            key = self.display.show(rgb, depth, det_vis, map_vis, stats)
            
            # Handle input
            if key == ord('q') or key == 27:  # Q or ESC
                self.running = False
            elif key == ord('r'):
                self.display.toggle_recording(self.record_path)
            elif key == ord(' '):
                self.paused = not self.paused
                print("Paused" if self.paused else "Resumed")
            elif not self.paused and key in self.ACTION_MAP:
                action = self.ACTION_MAP[key]
                result = self.step(action)
                rgb = result['rgb']
                depth = result['depth']
                position = result['position']
                rotation = result['rotation']
        
        # Cleanup
        self.display.close()
        print(f"\nExploration complete. Steps: {self.step_count}")
        
        return self.semantic_map
