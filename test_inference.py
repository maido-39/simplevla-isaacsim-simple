#!/usr/bin/env python3
"""
Test script for inference API
"""

import os
import sys
from inference_api import VLAInferenceAPI, load_images_from_paths

def main():
    # Paths
    checkpoint_path = "./checkpoints/best.pth"
    data_root = "/home/syaro/MikuchanRemote/Remotecontrol-Demo/Isaacsim-spot-remotecontroldemo/expr_data/20251121_Proto_Data"
    
    # Sample data directory
    sample_dir = os.path.join(data_root, "251120_203659-moonjihun_moving")
    
    # Get image paths (history_length = 5)
    ego_dir = os.path.join(sample_dir, "camera", "ego")
    top_dir = os.path.join(sample_dir, "camera", "top")
    
    # Get first 5 frames
    ego_images = []
    top_images = []
    for i in range(5):
        ego_path = os.path.join(ego_dir, f"frame{i}-*.jpg")
        top_path = os.path.join(top_dir, f"frame{i}-*.jpg")
        
        # Use glob to find actual files
        import glob
        ego_files = glob.glob(ego_path)
        top_files = glob.glob(top_path)
        
        if ego_files and top_files:
            ego_images.append(ego_files[0])
            top_images.append(top_files[0])
        else:
            print(f"Warning: Could not find frame {i}")
            break
    
    if len(ego_images) < 5:
        print(f"Error: Need 5 images, found {len(ego_images)}")
        return
    
    # Get instruction
    instruction_file = os.path.join(sample_dir, "moonjihun_moving_Instruction.txt")
    if os.path.exists(instruction_file):
        with open(instruction_file, 'r') as f:
            instruction = f.read().strip()
    else:
        instruction = "Move forward"
    
    print(f"Using instruction: {instruction}")
    print(f"Ego images: {ego_images}")
    print(f"Top images: {top_images}")
    
    # Initialize API
    print("\n" + "="*50)
    print("Initializing Inference API...")
    print("="*50)
    api = VLAInferenceAPI(
        checkpoint_path=checkpoint_path,
        device='cuda'
    )
    
    # Load images
    print("\nLoading images...")
    ego_tensors, top_tensors = load_images_from_paths(ego_images, top_images)
    print(f"Loaded {len(ego_images)} image pairs")
    
    # Run inference
    print("\n" + "="*50)
    print("Running Inference...")
    print("="*50)
    trajectory, intermediates = api.predict(
        ego_tensors, top_tensors, instruction, capture_intermediates=True
    )
    
    print(f"\nPredicted trajectory shape: {trajectory.shape}")
    print(f"First 5 steps:")
    for i in range(min(5, len(trajectory))):
        print(f"  Step {i}: x={trajectory[i, 0]:.4f}, y={trajectory[i, 1]:.4f}, z={trajectory[i, 2]:.4f}")
    
    # Benchmark
    print("\n" + "="*50)
    print("Running Performance Benchmark...")
    print("="*50)
    stats = api.benchmark(ego_tensors, top_tensors, instruction, num_runs=10, warmup_runs=2)
    
    # Visualize
    print("\n" + "="*50)
    print("Generating Visualization...")
    print("="*50)
    api.visualize_inference(
        trajectory, intermediates, instruction,
        output_path="inference_analysis.png", show_plot=False
    )
    
    # Print final stats
    print("\n" + "="*50)
    print("Final Performance Summary")
    print("="*50)
    final_stats = api.get_performance_stats()
    if final_stats:
        print(f"Total inferences: {final_stats['total_inferences']}")
        print(f"Mean inference time: {final_stats['mean_inference_time_ms']:.2f} ms")
        print(f"Inference speed: {final_stats['mean_fps']:.2f} Hz")
        print(f"Std deviation: {final_stats['std_inference_time_ms']:.2f} ms")
        print(f"\nComponent breakdown:")
        for comp, comp_stats in final_stats['components'].items():
            print(f"  {comp:20s}: {comp_stats['mean_ms']:6.2f} ms ({comp_stats['percentage']:5.1f}%)")
    
    print("\n" + "="*50)
    print("Inference completed successfully!")
    print("="*50)
    print(f"Visualization saved to: inference_analysis.png")
    print(f"Trajectory shape: {trajectory.shape}")

if __name__ == '__main__':
    main()

