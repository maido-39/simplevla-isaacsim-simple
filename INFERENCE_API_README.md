# VLA Inference API Documentation

## Overview

The `inference_api.py` provides a comprehensive API for running inference with the trained Simple VLA model, including:
- **Visualization** of intermediate outputs from each model component (ViT, LLM, Diffusion)
- **Performance benchmarking** with detailed timing breakdown
- **Easy-to-use API** for integration with other code
- **Inference speed measurement** in Hz

## Features

### 1. Intermediate Output Visualization
- Input images (ego and top views)
- Vision encoder outputs
- Language model hidden states
- Projected features
- Diffusion model features
- Final predicted trajectory (2D and 3D plots)
- Component timing breakdown

### 2. Performance Metrics
- Mean inference time (ms)
- Inference speed (Hz)
- Standard deviation
- Component-wise timing breakdown:
  - Vision encoding
  - Language encoding
  - Feature projection
  - Diffusion denoising

### 3. API Interface
Clean Python API that can be easily integrated into other projects.

## Usage

### Basic Usage

```python
from inference_api import VLAInferenceAPI, load_images_from_paths

# Initialize API
api = VLAInferenceAPI(
    checkpoint_path="./checkpoints/best.pth",
    device='cuda'  # or 'cpu'
)

# Load images
ego_images, top_images = load_images_from_paths(
    ego_image_paths=['path/to/ego0.jpg', 'path/to/ego1.jpg', ...],
    top_image_paths=['path/to/top0.jpg', 'path/to/top1.jpg', ...]
)

# Run inference
trajectory, intermediates = api.predict(
    ego_images, top_images, 
    instruction="Move forward",
    capture_intermediates=True
)

# Visualize results
api.visualize_inference(
    trajectory, intermediates, "Move forward",
    output_path="inference_analysis.png"
)

# Get performance stats
stats = api.get_performance_stats()
print(f"Inference speed: {stats['mean_fps']:.2f} Hz")
```

### Command Line Usage

```bash
# Basic inference with visualization
python inference_api.py \
    --checkpoint ./checkpoints/best.pth \
    --ego_images path/to/ego0.jpg path/to/ego1.jpg ... \
    --top_images path/to/top0.jpg path/to/top1.jpg ... \
    --instruction "Move forward" \
    --viz_output inference_analysis.png

# With benchmarking
python inference_api.py \
    --checkpoint ./checkpoints/best.pth \
    --ego_images path/to/ego0.jpg ... \
    --top_images path/to/top0.jpg ... \
    --instruction "Move forward" \
    --benchmark \
    --num_runs 20
```

### Test Script

A test script is provided for quick testing:

```bash
python test_inference.py
```

This will:
1. Load sample images from the dataset
2. Run inference
3. Perform benchmarking (10 runs)
4. Generate visualization
5. Print performance summary

## API Reference

### `VLAInferenceAPI`

Main class for inference.

#### `__init__(checkpoint_path, device='cuda', history_length=5, trajectory_length=16)`
Initialize the inference API.

**Parameters:**
- `checkpoint_path`: Path to model checkpoint (.pth file)
- `device`: Device to use ('cuda' or 'cpu')
- `history_length`: Number of historical frames (default: 5)
- `trajectory_length`: Length of predicted trajectory (default: 16)

#### `predict(ego_images, top_images, instruction, capture_intermediates=True)`
Run inference and predict trajectory.

**Parameters:**
- `ego_images`: Tensor of shape [H, C, H_img, W_img] or [B, H, C, H_img, W_img]
- `top_images`: Tensor of shape [H, C, H_img, W_img] or [B, H, C, H_img, W_img]
- `instruction`: Instruction string
- `capture_intermediates`: Whether to capture intermediate outputs

**Returns:**
- `trajectory`: numpy array of shape [T, 3] with predicted (x, y, z) positions
- `intermediates`: Dictionary containing intermediate outputs

#### `benchmark(ego_images, top_images, instruction, num_runs=10, warmup_runs=2)`
Benchmark inference performance.

**Parameters:**
- `ego_images`: Input ego images
- `top_images`: Input top images
- `instruction`: Instruction string
- `num_runs`: Number of benchmark runs
- `warmup_runs`: Number of warmup runs

**Returns:**
- Dictionary with performance statistics

#### `get_performance_stats()`
Get performance statistics from all inference runs.

**Returns:**
- Dictionary with:
  - `mean_inference_time_ms`: Mean inference time in milliseconds
  - `mean_fps`: Mean inference speed in Hz
  - `std_inference_time_ms`: Standard deviation
  - `components`: Component-wise timing breakdown

#### `visualize_inference(trajectory, intermediates, instruction, output_path=None, show_plot=False)`
Generate comprehensive visualization of inference results.

**Parameters:**
- `trajectory`: Predicted trajectory array
- `intermediates`: Dictionary of intermediate outputs
- `instruction`: Instruction text
- `output_path`: Path to save visualization (optional)
- `show_plot`: Whether to display plot (default: False)

## Performance Notes

Current performance (on test system):
- **Mean inference time**: ~5.6 seconds
- **Inference speed**: ~0.18 Hz
- **Component breakdown**:
  - Vision encoding: ~6.4%
  - Language encoding: ~4.2%
  - Feature projection: ~0.0%
  - Diffusion: ~89.4%

The diffusion model takes the majority of time because it runs 1000 denoising steps. To improve speed:
1. Reduce `num_timesteps` in `DiffusionActionHead` (currently 1000)
2. Use fewer inference steps (DDIM scheduler with fewer steps)
3. Use a smaller UNet architecture

## Integration Example

```python
# Example: Integrate into a robot control loop
from inference_api import VLAInferenceAPI
import torch

# Initialize once
api = VLAInferenceAPI("./checkpoints/best.pth", device='cuda')

# In control loop
def get_next_action(ego_images, top_images, instruction):
    """Get next action from VLA model."""
    trajectory, _ = api.predict(
        ego_images, top_images, instruction,
        capture_intermediates=False  # Faster for real-time
    )
    # Return first step of trajectory
    return trajectory[0]  # [x, y, z]

# Use in loop
while True:
    action = get_next_action(current_ego_images, current_top_images, "Move forward")
    robot.execute_action(action)
```

## Visualization Output

The visualization includes:
1. **Input Images**: Last ego and top camera images, combined view
2. **Language Features**: Bar plot of language model hidden states
3. **Projected Features**: Bar plot of projected features
4. **Diffusion Features**: Bar plot of diffusion input features
5. **Trajectory Plots**: X, Y, Z positions over time
6. **3D Trajectory**: 3D visualization of the predicted path
7. **Component Timing**: Bar chart of time spent in each component
8. **Performance Stats**: Text summary of inference performance

## Files

- `inference_api.py`: Main API implementation
- `test_inference.py`: Test script with example usage
- `inference.py`: Original simple inference script (still available)

## Requirements

- torch >= 2.0.0
- transformers >= 4.35.0
- matplotlib >= 3.5.0
- numpy >= 1.24.0
- pillow >= 10.0.0

All dependencies are listed in `requirements.txt` and `environment.yml`.

