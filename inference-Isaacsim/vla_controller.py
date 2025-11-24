"""
VLA-based controller for Isaac Sim robot control.
Uses vision-language-action model to predict robot commands from camera images.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple, List
import time
from PIL import Image

# Add parent directory to path to import inference API
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from inference_api import VLAInferenceAPI


class VLAController:
    """
    VLA-based controller that predicts robot commands from camera images.
    Provides same interface as KeyboardController for drop-in replacement.
    
    The controller:
    1. Maintains a history of camera images (ego + top)
    2. Runs inference at specified frequency
    3. Converts trajectory predictions to velocity commands
    4. Applies smoothing and filtering
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        instruction: str = "Navigate to the goal",
        max_vx: float = 2.0,
        max_vy: float = 2.0,
        max_yaw: float = 2.0,
        inference_freq: float = 2.0,  # Hz - how often to run inference
        update_dt: float = 0.02,  # 50Hz update rate for command smoothing
        device: str = 'cuda',
        history_length: int = 5,
        trajectory_length: int = 16,
        trajectory_to_velocity_scale: float = 1.0,  # Scale factor for trajectory to velocity conversion
        smoothing_alpha: float = 0.7,  # Smoothing factor for velocity commands (0-1, higher = more smoothing)
        use_first_step_only: bool = True,  # If True, use only first trajectory step; if False, use average of first N steps
        num_steps_for_velocity: int = 3,  # Number of trajectory steps to use for velocity calculation
    ):
        """
        Initialize VLA controller.
        
        Args:
            checkpoint_path: Path to model checkpoint
            instruction: Instruction text for the task
            max_vx: Maximum forward/backward velocity (m/s)
            max_vy: Maximum lateral velocity (m/s)
            max_yaw: Maximum rotation velocity (rad/s)
            inference_freq: Frequency to run inference (Hz)
            update_dt: Command update period (seconds)
            device: Device to run inference on ('cuda' or 'cpu')
            history_length: Number of image frames to maintain in history
            trajectory_length: Length of predicted trajectory
            trajectory_to_velocity_scale: Scale factor for converting trajectory to velocity
            smoothing_alpha: Smoothing factor for velocity commands (0-1)
            use_first_step_only: If True, use only first trajectory step
            num_steps_for_velocity: Number of trajectory steps to average for velocity
        """
        self.checkpoint_path = checkpoint_path
        self.instruction = instruction
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_yaw = max_yaw
        self.inference_freq = inference_freq
        self.update_dt = update_dt
        self.history_length = history_length
        self.trajectory_length = trajectory_length
        self.trajectory_to_velocity_scale = trajectory_to_velocity_scale
        self.smoothing_alpha = smoothing_alpha
        self.use_first_step_only = use_first_step_only
        self.num_steps_for_velocity = num_steps_for_velocity
        
        # Initialize inference API
        print(f"Loading VLA model from {checkpoint_path}...")
        self.inference_api = VLAInferenceAPI(
            checkpoint_path=checkpoint_path,
            device=device,
            history_length=history_length,
            trajectory_length=trajectory_length
        )
        print("VLA model loaded successfully")
        
        # Image history buffers
        self.ego_image_history: List[np.ndarray] = []
        self.top_image_history: List[np.ndarray] = []
        
        # Command state
        self._vx_cmd = 0.0
        self._vy_cmd = 0.0
        self._yaw_cmd = 0.0
        
        # Inference timing
        self.last_inference_time = 0.0
        self.inference_interval = 1.0 / inference_freq if inference_freq > 0 else float('inf')
        
        # Latest trajectory prediction
        self.latest_trajectory: Optional[np.ndarray] = None
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # Quit flag (for compatibility with KeyboardController interface)
        self._quit_requested = False
    
    def update_image_history(self, ego_image: np.ndarray, top_image: np.ndarray):
        """
        Update image history with new frames.
        
        Args:
            ego_image: Ego camera image (H, W, 3) RGB, values in [0, 255] or [0, 1]
            top_image: Top camera image (H, W, 3) RGB, values in [0, 255] or [0, 1]
        """
        # Normalize images to [0, 1] if needed
        if ego_image.max() > 1.0:
            ego_image = ego_image.astype(np.float32) / 255.0
        if top_image.max() > 1.0:
            top_image = top_image.astype(np.float32) / 255.0
        
        # Ensure images are float32
        ego_image = ego_image.astype(np.float32)
        top_image = top_image.astype(np.float32)
        
        # Add to history
        self.ego_image_history.append(ego_image.copy())
        self.top_image_history.append(top_image.copy())
        
        # Maintain history length
        if len(self.ego_image_history) > self.history_length:
            self.ego_image_history.pop(0)
            self.top_image_history.pop(0)
    
    def _convert_trajectory_to_velocity(
        self,
        trajectory: np.ndarray,
        dt: float = 0.1  # Time step between trajectory points (seconds)
    ) -> Tuple[float, float, float]:
        """
        Convert trajectory prediction to velocity commands.
        
        Args:
            trajectory: Predicted trajectory [T, 3] with (x, y, z) positions
            dt: Time step between trajectory points (seconds)
        
        Returns:
            (vx, vy, yaw_rate): Velocity commands
        """
        if trajectory is None or len(trajectory) == 0:
            return (0.0, 0.0, 0.0)
        
        if self.use_first_step_only:
            # Use only the first step: velocity = position / dt
            if len(trajectory) >= 1:
                dx = trajectory[0, 0]
                dy = trajectory[0, 1]
                dz = trajectory[0, 2]
            else:
                return (0.0, 0.0, 0.0)
        else:
            # Average first N steps
            n_steps = min(self.num_steps_for_velocity, len(trajectory))
            if n_steps > 0:
                dx = np.mean(trajectory[:n_steps, 0])
                dy = np.mean(trajectory[:n_steps, 1])
                dz = np.mean(trajectory[:n_steps, 2])
            else:
                return (0.0, 0.0, 0.0)
        
        # Convert position differences to velocities
        # Scale by trajectory_to_velocity_scale and divide by dt
        vx = (dx / dt) * self.trajectory_to_velocity_scale
        vy = (dy / dt) * self.trajectory_to_velocity_scale
        
        # For yaw, we can use the direction of movement in XY plane
        # yaw_rate = atan2(dy, dx) / dt, but we'll use a simpler approach
        # Calculate yaw from direction of movement
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            yaw = np.arctan2(dy, dx)
            # Convert yaw angle to yaw rate (rad/s)
            # We can use the yaw angle directly scaled by frequency
            yaw_rate = yaw * self.inference_freq * self.trajectory_to_velocity_scale
        else:
            yaw_rate = 0.0
        
        # Clamp to max velocities
        vx = np.clip(vx, -self.max_vx, self.max_vx)
        vy = np.clip(vy, -self.max_vy, self.max_vy)
        yaw_rate = np.clip(yaw_rate, -self.max_yaw, self.max_yaw)
        
        return (float(vx), float(vy), float(yaw_rate))
    
    def _run_inference(self) -> Optional[np.ndarray]:
        """
        Run inference on current image history.
        
        Returns:
            Predicted trajectory [T, 3] or None if inference fails
        """
        # Check if we have enough history
        if len(self.ego_image_history) < self.history_length:
            return None
        
        try:
            # Convert image history to tensors
            # Images are [H, W, 3] in [0, 1] range
            # Need to convert to [H, C, H_img, W_img] format for model
            
            ego_images_list = []
            top_images_list = []
            
            for ego_img, top_img in zip(self.ego_image_history, self.top_image_history):
                # Resize to 224x224 if needed (model expects this size)
                if ego_img.shape[:2] != (224, 224):
                    ego_pil = Image.fromarray((ego_img * 255).astype(np.uint8))
                    ego_pil = ego_pil.resize((224, 224))
                    ego_img = np.array(ego_pil).astype(np.float32) / 255.0
                
                if top_img.shape[:2] != (224, 224):
                    top_pil = Image.fromarray((top_img * 255).astype(np.uint8))
                    top_pil = top_pil.resize((224, 224))
                    top_img = np.array(top_pil).astype(np.float32) / 255.0
                
                # Convert to [C, H, W]
                ego_tensor = torch.from_numpy(ego_img).permute(2, 0, 1)  # [3, H, W]
                top_tensor = torch.from_numpy(top_img).permute(2, 0, 1)  # [3, H, W]
                
                ego_images_list.append(ego_tensor)
                top_images_list.append(top_tensor)
            
            # Stack to [H, C, H_img, W_img]
            ego_images = torch.stack(ego_images_list)  # [H, 3, 224, 224]
            top_images = torch.stack(top_images_list)  # [H, 3, 224, 224]
            
            # Run inference
            inference_start = time.time()
            trajectory, _ = self.inference_api.predict(
                ego_images,
                top_images,
                self.instruction,
                capture_intermediates=False  # Don't capture intermediates for performance
            )
            inference_time = time.time() - inference_start
            
            # Track performance
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return trajectory
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def update(self):
        """
        Update controller state.
        Should be called periodically from main loop (e.g., at 50Hz).
        Runs inference at specified frequency and updates velocity commands.
        """
        current_time = time.time()
        
        # Check if it's time to run inference
        if current_time - self.last_inference_time >= self.inference_interval:
            # Run inference
            trajectory = self._run_inference()
            
            if trajectory is not None:
                self.latest_trajectory = trajectory
                
                # Convert trajectory to velocity
                vx_new, vy_new, yaw_new = self._convert_trajectory_to_velocity(trajectory)
                
                # Apply smoothing
                self._vx_cmd = self.smoothing_alpha * self._vx_cmd + (1.0 - self.smoothing_alpha) * vx_new
                self._vy_cmd = self.smoothing_alpha * self._vy_cmd + (1.0 - self.smoothing_alpha) * vy_new
                self._yaw_cmd = self.smoothing_alpha * self._yaw_cmd + (1.0 - self.smoothing_alpha) * yaw_new
                
                # Clamp to max velocities
                self._vx_cmd = np.clip(self._vx_cmd, -self.max_vx, self.max_vx)
                self._vy_cmd = np.clip(self._vy_cmd, -self.max_vy, self.max_vy)
                self._yaw_cmd = np.clip(self._yaw_cmd, -self.max_yaw, self.max_yaw)
            
            self.last_inference_time = current_time
        
        # Apply decay if no recent inference (safety)
        # This helps prevent the robot from continuing with stale commands
        time_since_inference = current_time - self.last_inference_time
        if time_since_inference > 2.0:  # If no inference for 2 seconds, decay commands
            decay_factor = 0.95
            self._vx_cmd *= decay_factor
            self._vy_cmd *= decay_factor
            self._yaw_cmd *= decay_factor
    
    def get_command(self) -> np.ndarray:
        """
        Get current velocity command.
        
        Returns:
            np.ndarray: [vx, vy, yaw] command array
        """
        return np.array([self._vx_cmd, self._vy_cmd, self._yaw_cmd])
    
    def is_quit_requested(self) -> bool:
        """
        Check if quit is requested.
        
        Returns:
            bool: True if quit requested
        """
        return self._quit_requested
    
    def request_quit(self):
        """Request controller to quit"""
        self._quit_requested = True
    
    def reset(self):
        """Reset command state"""
        self._vx_cmd = 0.0
        self._vy_cmd = 0.0
        self._yaw_cmd = 0.0
        self.ego_image_history.clear()
        self.top_image_history.clear()
        self.latest_trajectory = None
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if self.inference_count == 0:
            return {}
        
        avg_inference_time = self.total_inference_time / self.inference_count
        return {
            'inference_count': self.inference_count,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'total_inference_time_s': self.total_inference_time,
            'inference_freq_hz': self.inference_freq,
        }
    
    def start(self):
        """Start controller (for compatibility with KeyboardController interface)"""
        pass
    
    def stop(self):
        """Stop controller (for compatibility with KeyboardController interface)"""
        self._quit_requested = True

