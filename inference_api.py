"""
Enhanced Inference API for Simple VLA Model
Provides visualization of intermediate outputs and performance metrics
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import time
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import SimpleVLA
import json


class VLAInferenceAPI:
    """
    API class for running inference with the Simple VLA model.
    Provides visualization of intermediate outputs and performance metrics.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        history_length: int = 5,
        trajectory_length: int = 16
    ):
        """
        Initialize the inference API.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            history_length: History length for input images
            trajectory_length: Length of predicted trajectory
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.history_length = history_length
        self.trajectory_length = trajectory_length
        
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Performance tracking
        self.inference_times = []
        self.component_times = {
            'vision_encoding': [],
            'language_encoding': [],
            'feature_projection': [],
            'diffusion': []
        }
        
        # Intermediate outputs storage
        self.intermediate_outputs = {}
    
    def _load_model(self, checkpoint_path: str) -> SimpleVLA:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model config from checkpoint
        history_length = self.history_length
        trajectory_length = self.trajectory_length
        use_lora = True
        lora_r = 16
        lora_alpha = 32
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            history_length = config.get('history_length', self.history_length)
            trajectory_length = config.get('trajectory_length', self.trajectory_length)
            use_lora = config.get('use_lora', True)
            lora_r = config.get('lora_r', 16)
            lora_alpha = config.get('lora_alpha', 32)
        elif 'config' in checkpoint:
            config = checkpoint['config']
            history_length = config.get('history_length', self.history_length)
            trajectory_length = config.get('trajectory_length', self.trajectory_length)
            use_lora = config.get('use_lora', True)
            lora_r = config.get('lora_r', 16)
            lora_alpha = config.get('lora_alpha', 32)
        
        self.history_length = history_length
        self.trajectory_length = trajectory_length
        
        # Create model
        model = SimpleVLA(
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            trajectory_length=trajectory_length,
            history_length=history_length
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Model loaded successfully")
        print(f"  - History length: {history_length}")
        print(f"  - Trajectory length: {trajectory_length}")
        print(f"  - Device: {self.device}")
        if 'epoch' in checkpoint:
            print(f"  - Trained for {checkpoint['epoch']} epochs")
        if 'val_loss' in checkpoint:
            print(f"  - Best validation loss: {checkpoint['val_loss']:.4f}")
        
        return model
    
    def _encode_with_intermediates(
        self,
        ego_images: torch.Tensor,
        top_images: torch.Tensor,
        instructions: List[str]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encode images and text with intermediate outputs captured.
        
        Returns:
            features: Encoded features
            intermediates: Dictionary of intermediate outputs
        """
        intermediates = {}
        batch_size = ego_images.shape[0]
        device = self.model.device if hasattr(self.model, 'device') else ego_images.device
        
        # Prepare images
        combined_images = []
        for b in range(batch_size):
            image_list = []
            for h in range(self.history_length):
                ego_img = ego_images[b, h].cpu().permute(1, 2, 0).numpy()
                top_img = top_images[b, h].cpu().permute(1, 2, 0).numpy()
                
                from PIL import Image as PILImage
                ego_pil = PILImage.fromarray((ego_img * 255).astype('uint8'))
                top_pil = PILImage.fromarray((top_img * 255).astype('uint8'))
                
                combined = PILImage.new('RGB', (ego_pil.width * 2, ego_pil.height))
                combined.paste(ego_pil, (0, 0))
                combined.paste(top_pil, (ego_pil.width, 0))
                combined_images.append(combined)
        
        # Store input images for visualization
        intermediates['input_images'] = {
            'ego': ego_images.cpu().numpy(),
            'top': top_images.cpu().numpy(),
            'combined': [np.array(img) for img in combined_images]
        }
        
        # Process with PaliGemma processor
        instructions_with_image = [f"<image>\n{instr}" for instr in instructions]
        
        t0 = time.time()
        inputs = self.model.processor(
            text=instructions_with_image,
            images=[combined_images[-1] for _ in range(batch_size)],
            return_tensors="pt",
            padding=True
        ).to(device)
        t1 = time.time()
        self.component_times['vision_encoding'].append(t1 - t0)
        
        # Get features from PaliGemma with intermediate outputs
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.paligemma(**inputs, output_hidden_states=True)
        
        # Extract vision encoder outputs if available
        if hasattr(outputs, 'vision_model_outputs'):
            intermediates['vision_outputs'] = {
                'last_hidden_state': outputs.vision_model_outputs.last_hidden_state.cpu().numpy() if hasattr(outputs.vision_model_outputs, 'last_hidden_state') else None,
                'pooler_output': outputs.vision_model_outputs.pooler_output.cpu().numpy() if hasattr(outputs.vision_model_outputs, 'pooler_output') else None,
            }
        
        # Extract language model hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_size]
            intermediates['language_hidden_states'] = hidden_states.cpu().numpy()
            intermediates['language_hidden_states_mean'] = hidden_states.mean(dim=1).cpu().numpy()
            features = hidden_states.mean(dim=1)  # [B, hidden_size]
        else:
            if hasattr(outputs, 'logits'):
                features = outputs.logits.mean(dim=1)
                intermediates['language_logits'] = outputs.logits.cpu().numpy()
            else:
                raise ValueError("Could not extract features from PaliGemma outputs")
        
        t1 = time.time()
        self.component_times['language_encoding'].append(t1 - t0)
        
        # Feature projection
        t0 = time.time()
        features_proj = self.model.feature_proj(features)
        intermediates['projected_features'] = features_proj.cpu().numpy()
        t1 = time.time()
        self.component_times['feature_projection'].append(t1 - t0)
        
        return features_proj, intermediates
    
    def predict(
        self,
        ego_images: torch.Tensor,
        top_images: torch.Tensor,
        instruction: str,
        capture_intermediates: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Predict trajectory from images and instruction.
        
        Args:
            ego_images: [H, C, H_img, W_img] ego camera images
            top_images: [H, C, H_img, W_img] top camera images
            instruction: Instruction string
            capture_intermediates: Whether to capture intermediate outputs
        
        Returns:
            trajectory: [T, 3] array of predicted (x, y, z) positions
            intermediates: Dictionary of intermediate outputs
        """
        # Add batch dimension
        ego_images = ego_images.unsqueeze(0).to(self.device) if len(ego_images.shape) == 4 else ego_images.to(self.device)
        top_images = top_images.unsqueeze(0).to(self.device) if len(top_images.shape) == 4 else top_images.to(self.device)
        
        intermediates = {}
        
        # Encode with intermediates
        if capture_intermediates:
            features, enc_intermediates = self._encode_with_intermediates(
                ego_images, top_images, [instruction]
            )
            intermediates.update(enc_intermediates)
        else:
            features = self.model.encode_images_and_text(ego_images, top_images, [instruction])
        
        # Diffusion inference
        t0 = time.time()
        with torch.no_grad():
            trajectory = self.model.action_head(features)
        
        # Capture diffusion intermediate states if possible
        if capture_intermediates:
            # Store diffusion features
            intermediates['diffusion_features'] = features.cpu().numpy()
            intermediates['diffusion_input_shape'] = trajectory.shape
        
        t1 = time.time()
        self.component_times['diffusion'].append(t1 - t0)
        
        # Convert to numpy
        trajectory = trajectory.cpu().numpy()[0]  # [T, 3]
        intermediates['final_trajectory'] = trajectory
        
        # Track total inference time
        total_time = sum([
            self.component_times['vision_encoding'][-1],
            self.component_times['language_encoding'][-1],
            self.component_times['feature_projection'][-1],
            self.component_times['diffusion'][-1]
        ])
        self.inference_times.append(total_time)
        
        return trajectory, intermediates
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        stats = {
            'total_inferences': len(self.inference_times),
            'mean_inference_time_ms': np.mean(self.inference_times) * 1000,
            'std_inference_time_ms': np.std(self.inference_times) * 1000,
            'min_inference_time_ms': np.min(self.inference_times) * 1000,
            'max_inference_time_ms': np.max(self.inference_times) * 1000,
            'mean_fps': 1.0 / np.mean(self.inference_times),
            'components': {}
        }
        
        for component, times in self.component_times.items():
            if times:
                stats['components'][component] = {
                    'mean_ms': np.mean(times) * 1000,
                    'std_ms': np.std(times) * 1000,
                    'percentage': (np.mean(times) / np.mean(self.inference_times)) * 100
                }
        
        return stats
    
    def visualize_inference(
        self,
        trajectory: np.ndarray,
        intermediates: Dict,
        instruction: str,
        output_path: Optional[str] = None,
        show_plot: bool = False
    ):
        """
        Visualize inference results including intermediate outputs.
        
        Args:
            trajectory: Predicted trajectory [T, 3]
            intermediates: Dictionary of intermediate outputs
            instruction: Instruction text
            output_path: Path to save visualization
            show_plot: Whether to display the plot
        """
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Input Images (Ego and Top views)
        if 'input_images' in intermediates:
            ax1 = fig.add_subplot(gs[0, 0])
            ego_imgs = intermediates['input_images']['ego'][0]  # [H, C, H_img, W_img]
            # Show last ego image
            last_ego = ego_imgs[-1].transpose(1, 2, 0)
            ax1.imshow(last_ego)
            ax1.set_title('Input: Last Ego Image', fontsize=10)
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            top_imgs = intermediates['input_images']['top'][0]
            last_top = top_imgs[-1].transpose(1, 2, 0)
            ax2.imshow(last_top)
            ax2.set_title('Input: Last Top Image', fontsize=10)
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            if 'combined' in intermediates['input_images']:
                combined = intermediates['input_images']['combined'][-1]
                ax3.imshow(combined)
                ax3.set_title('Combined Input (Ego+Top)', fontsize=10)
                ax3.axis('off')
        
        # 2. Language Model Features
        if 'language_hidden_states_mean' in intermediates:
            ax4 = fig.add_subplot(gs[0, 3])
            lang_features = intermediates['language_hidden_states_mean'][0]
            # Show first 100 dimensions as bar plot
            n_dims = min(100, len(lang_features))
            ax4.bar(range(n_dims), lang_features[:n_dims])
            ax4.set_title(f'Language Features (first {n_dims} dims)', fontsize=10)
            ax4.set_xlabel('Dimension')
            ax4.set_ylabel('Value')
        
        # 3. Projected Features
        if 'projected_features' in intermediates:
            ax5 = fig.add_subplot(gs[1, 0])
            proj_features = intermediates['projected_features'][0]
            n_dims = min(100, len(proj_features))
            ax5.bar(range(n_dims), proj_features[:n_dims])
            ax5.set_title(f'Projected Features (first {n_dims} dims)', fontsize=10)
            ax5.set_xlabel('Dimension')
            ax5.set_ylabel('Value')
        
        # 4. Diffusion Features
        if 'diffusion_features' in intermediates:
            ax6 = fig.add_subplot(gs[1, 1])
            diff_features = intermediates['diffusion_features'][0]
            n_dims = min(100, len(diff_features))
            ax6.bar(range(n_dims), diff_features[:n_dims])
            ax6.set_title(f'Diffusion Input Features (first {n_dims} dims)', fontsize=10)
            ax6.set_xlabel('Dimension')
            ax6.set_ylabel('Value')
        
        # 5-7. Trajectory plots (X, Y, Z over time)
        time_steps = np.arange(len(trajectory))
        
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.plot(time_steps, trajectory[:, 0], 'b-', marker='o', linewidth=2, markersize=4)
        ax7.set_xlabel('Time Step')
        ax7.set_ylabel('X Position')
        ax7.set_title('X Position over Time', fontsize=10)
        ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.plot(time_steps, trajectory[:, 1], 'g-', marker='o', linewidth=2, markersize=4)
        ax8.set_xlabel('Time Step')
        ax8.set_ylabel('Y Position')
        ax8.set_title('Y Position over Time', fontsize=10)
        ax8.grid(True, alpha=0.3)
        
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.plot(time_steps, trajectory[:, 2], 'r-', marker='o', linewidth=2, markersize=4)
        ax9.set_xlabel('Time Step')
        ax9.set_ylabel('Z Position')
        ax9.set_title('Z Position over Time', fontsize=10)
        ax9.grid(True, alpha=0.3)
        
        # 8. 3D Trajectory
        ax10 = fig.add_subplot(gs[2, 1], projection='3d')
        ax10.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', marker='o', linewidth=2, markersize=4)
        ax10.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='green', s=100, label='Start')
        ax10.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='red', s=100, label='End')
        ax10.set_xlabel('X')
        ax10.set_ylabel('Y')
        ax10.set_zlabel('Z')
        ax10.set_title('3D Trajectory', fontsize=10)
        ax10.legend()
        
        # 9. Performance stats
        ax11 = fig.add_subplot(gs[2, 2])
        stats = self.get_performance_stats()
        if stats:
            component_names = list(stats['components'].keys())
            component_times = [stats['components'][c]['mean_ms'] for c in component_names]
            colors = plt.cm.viridis(np.linspace(0, 1, len(component_names)))
            ax11.barh(component_names, component_times, color=colors)
            ax11.set_xlabel('Time (ms)')
            ax11.set_title('Component Timing', fontsize=10)
            ax11.grid(True, alpha=0.3, axis='x')
        
        # 10. Instruction and stats text
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.axis('off')
        info_text = f"Instruction: {instruction}\n\n"
        if stats:
            info_text += f"Mean Inference Time: {stats['mean_inference_time_ms']:.2f} ms\n"
            info_text += f"Inference Speed: {stats['mean_fps']:.2f} Hz\n"
            info_text += f"Std Dev: {stats['std_inference_time_ms']:.2f} ms\n"
            info_text += f"Min: {stats['min_inference_time_ms']:.2f} ms\n"
            info_text += f"Max: {stats['max_inference_time_ms']:.2f} ms"
        ax12.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('VLA Model Inference Analysis', fontsize=16, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def benchmark(
        self,
        ego_images: torch.Tensor,
        top_images: torch.Tensor,
        instruction: str,
        num_runs: int = 10,
        warmup_runs: int = 2
    ) -> Dict:
        """
        Benchmark inference performance.
        
        Args:
            ego_images: Input ego images
            top_images: Input top images
            instruction: Instruction text
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
        
        Returns:
            Performance statistics
        """
        print(f"Benchmarking with {warmup_runs} warmup and {num_runs} runs...")
        
        # Warmup
        for _ in range(warmup_runs):
            self.predict(ego_images, top_images, instruction, capture_intermediates=False)
        
        # Clear previous stats
        self.inference_times = []
        self.component_times = {k: [] for k in self.component_times.keys()}
        
        # Benchmark
        for i in range(num_runs):
            self.predict(ego_images, top_images, instruction, capture_intermediates=False)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs...")
        
        stats = self.get_performance_stats()
        print(f"\nBenchmark Results:")
        print(f"  Mean inference time: {stats['mean_inference_time_ms']:.2f} ms")
        print(f"  Inference speed: {stats['mean_fps']:.2f} Hz")
        print(f"  Std deviation: {stats['std_inference_time_ms']:.2f} ms")
        
        return stats


def load_images_from_paths(
    ego_image_paths: List[str],
    top_image_paths: List[str],
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess images from file paths."""
    ego_images = []
    top_images = []
    
    for ego_path, top_path in zip(ego_image_paths, top_image_paths):
        # Load ego image
        if os.path.exists(ego_path):
            ego_img = Image.open(ego_path).convert('RGB')
            ego_img = ego_img.resize(image_size)
            ego_array = np.array(ego_img).astype(np.float32) / 255.0
            ego_tensor = torch.from_numpy(ego_array).permute(2, 0, 1)  # [C, H, W]
        else:
            ego_tensor = torch.zeros(3, image_size[1], image_size[0])
        ego_images.append(ego_tensor)
        
        # Load top image
        if os.path.exists(top_path):
            top_img = Image.open(top_path).convert('RGB')
            top_img = top_img.resize(image_size)
            top_array = np.array(top_img).astype(np.float32) / 255.0
            top_tensor = torch.from_numpy(top_array).permute(2, 0, 1)
        else:
            top_tensor = torch.zeros(3, image_size[1], image_size[0])
        top_images.append(top_tensor)
    
    # Stack into [H, C, H_img, W_img]
    ego_images = torch.stack(ego_images)
    top_images = torch.stack(top_images)
    
    return ego_images, top_images


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced VLA Inference with Visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--ego_images', type=str, nargs='+', required=True,
                        help='Paths to ego camera images')
    parser.add_argument('--top_images', type=str, nargs='+', required=True,
                        help='Paths to top camera images')
    parser.add_argument('--instruction', type=str, required=True,
                        help='Instruction text')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for trajectory (.npy)')
    parser.add_argument('--viz_output', type=str, default='inference_analysis.png',
                        help='Output file for visualization')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of benchmark runs')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize API
    api = VLAInferenceAPI(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Load images
    print("Loading images...")
    ego_images, top_images = load_images_from_paths(args.ego_images, args.top_images)
    print(f"Loaded {len(args.ego_images)} image pairs")
    
    # Run inference
    print(f"Running inference with instruction: '{args.instruction}'")
    trajectory, intermediates = api.predict(
        ego_images, top_images, args.instruction, capture_intermediates=True
    )
    
    print(f"\nPredicted trajectory shape: {trajectory.shape}")
    print(f"First 5 steps:")
    for i in range(min(5, len(trajectory))):
        print(f"  Step {i}: x={trajectory[i, 0]:.4f}, y={trajectory[i, 1]:.4f}, z={trajectory[i, 2]:.4f}")
    
    # Save trajectory
    if args.output:
        np.save(args.output, trajectory)
        print(f"\nTrajectory saved to {args.output}")
    
    # Benchmark if requested
    if args.benchmark:
        print("\n" + "="*50)
        stats = api.benchmark(ego_images, top_images, args.instruction, num_runs=args.num_runs)
        print("="*50)
    
    # Visualize
    print(f"\nGenerating visualization...")
    api.visualize_inference(
        trajectory, intermediates, args.instruction,
        output_path=args.viz_output, show_plot=False
    )
    
    # Print final stats
    stats = api.get_performance_stats()
    if stats:
        print(f"\nPerformance Summary:")
        print(f"  Mean inference time: {stats['mean_inference_time_ms']:.2f} ms")
        print(f"  Inference speed: {stats['mean_fps']:.2f} Hz")
        print(f"  Component breakdown:")
        for comp, comp_stats in stats['components'].items():
            print(f"    {comp}: {comp_stats['mean_ms']:.2f} ms ({comp_stats['percentage']:.1f}%)")

