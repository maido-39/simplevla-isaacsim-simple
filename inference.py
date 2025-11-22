import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import argparse
import json
from model import SimpleVLA
from typing import List, Tuple
import matplotlib.pyplot as plt


def load_images_from_paths(
    ego_image_paths: List[str],
    top_image_paths: List[str],
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess images from file paths.
    
    Args:
        ego_image_paths: List of paths to ego camera images
        top_image_paths: List of paths to top camera images
        image_size: Target image size
    
    Returns:
        ego_images: [H, C, H_img, W_img] tensor
        top_images: [H, C, H_img, W_img] tensor
    """
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
            # Create black image if not found
            ego_tensor = torch.zeros(3, image_size[1], image_size[0])
        ego_images.append(ego_tensor)
        
        # Load top image
        if os.path.exists(top_path):
            top_img = Image.open(top_path).convert('RGB')
            top_img = top_img.resize(image_size)
            top_array = np.array(top_img).astype(np.float32) / 255.0
            top_tensor = torch.from_numpy(top_array).permute(2, 0, 1)  # [C, H, W]
        else:
            # Create black image if not found
            top_tensor = torch.zeros(3, image_size[1], image_size[0])
        top_images.append(top_tensor)
    
    # Stack into [H, C, H_img, W_img]
    ego_images = torch.stack(ego_images)
    top_images = torch.stack(top_images)
    
    return ego_images, top_images


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    history_length: int = 5,
    trajectory_length: int = 16
) -> SimpleVLA:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        history_length: History length
        trajectory_length: Trajectory length
    
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get model config from checkpoint or use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
        history_length = config.get('history_length', history_length)
        trajectory_length = config.get('trajectory_length', trajectory_length)
        use_lora = config.get('use_lora', True)
        lora_r = config.get('lora_r', 16)
        lora_alpha = config.get('lora_alpha', 32)
    else:
        # Use defaults
        use_lora = True
        lora_r = 16
        lora_alpha = 32
    
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
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    return model


def predict_trajectory(
    model: SimpleVLA,
    ego_images: torch.Tensor,
    top_images: torch.Tensor,
    instruction: str,
    device: torch.device
) -> np.ndarray:
    """
    Predict trajectory from images and instruction.
    
    Args:
        model: Trained model
        ego_images: [H, C, H_img, W_img] ego camera images
        top_images: [H, C, H_img, W_img] top camera images
        instruction: Instruction string
        device: Device to run inference on
    
    Returns:
        predicted_positions: [T, 3] array of predicted (x, y, z) positions
    """
    # Add batch dimension: [H, C, H_img, W_img] -> [1, H, C, H_img, W_img]
    ego_images = ego_images.unsqueeze(0).to(device)
    top_images = top_images.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predicted_positions = model(
            ego_images=ego_images,
            top_images=top_images,
            instructions=[instruction]
        )
    
    # Convert to numpy: [1, T, 3] -> [T, 3]
    predicted_positions = predicted_positions.cpu().numpy()[0]
    
    return predicted_positions


def visualize_trajectory(
    trajectory: np.ndarray,
    output_path: str = None,
    title: str = "Predicted Trajectory"
):
    """
    Visualize predicted trajectory.
    
    Args:
        trajectory: [T, 3] array of (x, y, z) positions
        output_path: Path to save visualization (optional)
        title: Plot title
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Plot x, y, z over time
    time_steps = np.arange(len(trajectory))
    
    ax1 = plt.subplot(1, 3, 1)
    plt.plot(time_steps, trajectory[:, 0], 'b-', marker='o', label='X')
    plt.xlabel('Time Step')
    plt.ylabel('Position X')
    plt.title('X Position over Time')
    plt.grid(True)
    plt.legend()
    
    ax2 = plt.subplot(1, 3, 2)
    plt.plot(time_steps, trajectory[:, 1], 'g-', marker='o', label='Y')
    plt.xlabel('Time Step')
    plt.ylabel('Position Y')
    plt.title('Y Position over Time')
    plt.grid(True)
    plt.legend()
    
    ax3 = plt.subplot(1, 3, 3)
    plt.plot(time_steps, trajectory[:, 2], 'r-', marker='o', label='Z')
    plt.xlabel('Time Step')
    plt.ylabel('Position Z')
    plt.title('Z Position over Time')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory visualization saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Inference with Simple VLA Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--ego_images', type=str, nargs='+', required=True,
                        help='Paths to ego camera images (history length images)')
    parser.add_argument('--top_images', type=str, nargs='+', required=True,
                        help='Paths to top camera images (history length images)')
    parser.add_argument('--instruction', type=str, required=True,
                        help='Instruction text for the task')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save trajectory (numpy .npy format)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predicted trajectory')
    parser.add_argument('--viz_output', type=str, default=None,
                        help='Path to save trajectory visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Validate image paths
    if len(args.ego_images) != len(args.top_images):
        raise ValueError(f"Number of ego images ({len(args.ego_images)}) must match number of top images ({len(args.top_images)})")
    
    history_length = len(args.ego_images)
    print(f"History length: {history_length}")
    
    # Load model
    model = load_model_from_checkpoint(
        args.checkpoint,
        device,
        history_length=history_length
    )
    
    # Load images
    print("Loading images...")
    ego_images, top_images = load_images_from_paths(
        args.ego_images,
        args.top_images
    )
    print(f"Loaded {len(args.ego_images)} image pairs")
    
    # Predict trajectory
    print(f"Running inference with instruction: '{args.instruction}'")
    trajectory = predict_trajectory(
        model,
        ego_images,
        top_images,
        args.instruction,
        device
    )
    
    print(f"\nPredicted trajectory shape: {trajectory.shape}")
    print(f"Trajectory (first 5 steps):")
    for i in range(min(5, len(trajectory))):
        print(f"  Step {i}: x={trajectory[i, 0]:.4f}, y={trajectory[i, 1]:.4f}, z={trajectory[i, 2]:.4f}")
    
    # Save trajectory
    if args.output:
        np.save(args.output, trajectory)
        print(f"\nTrajectory saved to {args.output}")
    
    # Visualize
    if args.visualize or args.viz_output:
        visualize_trajectory(
            trajectory,
            output_path=args.viz_output,
            title=f"Predicted Trajectory: {args.instruction}"
        )
    
    return trajectory


if __name__ == '__main__':
    trajectory = main()

