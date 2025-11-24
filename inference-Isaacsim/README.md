# VLA Inference Integration for Isaac Sim

This directory contains integration code for running VLA (Vision-Language-Action) model inference in Isaac Sim simulations.

## Files

- `vla_controller.py`: VLA-based controller that replaces keyboard control
- `spot_demo_vla.py`: Simple demo with VLA inference (similar to `spot_demo.py`)
- `quadruped_example_vla.py`: Full environment demo with VLA inference (similar to `quadruped_example.py`)

## Overview

The VLA controller:
1. Maintains a history of camera images (ego + top view)
2. Runs inference at a specified frequency (default: 2 Hz)
3. Converts trajectory predictions to velocity commands [vx, vy, yaw]
4. Applies smoothing and filtering for stable control

## Requirements

- Isaac Sim installed and configured
- Trained VLA model checkpoint
- PyTorch with CUDA support (recommended)
- All dependencies from `simplevla-isaacsim-simple/requirements.txt`

### Important: Installing Dependencies in Isaac Sim

**Isaac Sim uses its own Python environment**, so you need to install dependencies there:

```bash
# Find Isaac Sim's Python (run in Isaac Sim terminal)
which python

# Install dependencies using Isaac Sim's Python
/path/to/isaac_sim/python.sh -m pip install peft transformers diffusers torch torchvision pillow numpy matplotlib
```

**Required packages:**
- `peft` - **Required** if your model uses LoRA (most checkpoints do)
- `transformers` - For PaliGemma model
- `diffusers` - For diffusion action head
- `torch`, `torchvision` - PyTorch (usually pre-installed)
- `pillow`, `numpy` - Image processing
- `matplotlib` - For visualization (optional)

See `INSTALL_DEPENDENCIES.md` for detailed installation instructions.

## Usage

### Simple Demo (spot_demo_vla.py)

Basic demo with ground plane and a box:

```bash
cd simplevla-isaacsim-simple/inference-Isaacsim
python spot_demo_vla.py \
    --checkpoint ../checkpoints/best.pth \
    --instruction "Navigate to the goal"
```

### Full Environment Demo (quadruped_example_vla.py)

Full environment with randomization, gates, boxes, and visualization:

```bash
cd simplevla-isaacsim-simple/inference-Isaacsim
python quadruped_example_vla.py \
    --checkpoint ../checkpoints/best.pth \
    --instruction "Push the box to the goal" \
    --experiment-name "VLA_TEST" \
    --loglevel INFO
```

## Command-Line Arguments

### spot_demo_vla.py

- `--checkpoint`: Path to VLA model checkpoint (required)
- `--instruction`: Instruction text for the task (default: "Navigate to the goal")

### quadruped_example_vla.py

- `--checkpoint`: Path to VLA model checkpoint (required)
- `--instruction`: Instruction text for the task (default: "Navigate to the goal")
- `--config`: Path to JSON config file (optional)
- `--experiment-name`: Name of the experiment (default: "VLA_TEST")
- `--loglevel`: Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)

## VLA Controller Parameters

The `VLAController` can be configured with the following parameters:

- `checkpoint_path`: Path to model checkpoint
- `instruction`: Instruction text for the task
- `max_vx`, `max_vy`, `max_yaw`: Maximum velocities (m/s, m/s, rad/s)
- `inference_freq`: Frequency to run inference (Hz, default: 2.0)
- `update_dt`: Command update period (seconds, default: 0.02 for 50Hz)
- `device`: Device to run inference on ('cuda' or 'cpu')
- `history_length`: Number of image frames in history (default: 5)
- `trajectory_length`: Length of predicted trajectory (default: 16)
- `trajectory_to_velocity_scale`: Scale factor for trajectory to velocity conversion (default: 1.0)
- `smoothing_alpha`: Smoothing factor for velocity commands (0-1, default: 0.7)
- `use_first_step_only`: If True, use only first trajectory step (default: True)
- `num_steps_for_velocity`: Number of trajectory steps to average for velocity (default: 3)

## How It Works

1. **Image History**: The controller maintains a rolling history of camera images (ego + top view)
2. **Inference**: At the specified frequency, the controller runs inference on the image history
3. **Trajectory to Velocity**: The predicted trajectory is converted to velocity commands:
   - Position differences are converted to velocities
   - Yaw rate is calculated from the direction of movement
4. **Smoothing**: Velocity commands are smoothed using exponential moving average
5. **Robot Control**: Commands are applied to the robot at 50Hz

## Performance

- Inference runs at 2 Hz by default (configurable)
- Command updates happen at 50 Hz for smooth control
- Camera images are captured at rendering rate (50 Hz)
- Performance stats are logged at the end of simulation

## Notes

- The controller maintains compatibility with the `KeyboardController` interface
- Camera images are automatically resized to 224x224 for the model
- The model expects images in [0, 1] range (RGB format)
- Trajectory predictions are in (x, y, z) position format
- The controller applies safety decay if no inference occurs for 2+ seconds

## Troubleshooting

1. **ModuleNotFoundError: No module named 'peft'**
   - **Solution**: Install `peft` in your conda environment or Isaac Sim's Python
   - For conda: `conda activate isc-pak && pip install peft`
   - For Isaac Sim: `/path/to/isaac_sim/python.sh -m pip install peft`
   - See `INSTALL_DEPENDENCIES.md` for detailed instructions

2. **ImportError: cannot import name 'cached_download' from 'huggingface_hub'**
   - **Solution**: Version mismatch between `diffusers` and `huggingface_hub`
   - **If you have robomimic**: Use `pip install diffusers==0.11.1 huggingface_hub==0.23.4`
   - **If you don't need robomimic**: Use `pip install 'diffusers>=0.21.0' 'huggingface_hub>=0.16.0,<0.20.0'`
   - See `FIX_ROBOMIMIC_CONFLICT.md` for detailed instructions on handling robomimic conflicts

3. **CUDA out of memory**: Reduce `inference_freq` or use CPU device
4. **Slow inference**: Check GPU availability, reduce `history_length` or `trajectory_length`
5. **Unstable control**: Adjust `smoothing_alpha` or `trajectory_to_velocity_scale`
6. **Camera not working**: Ensure Isaac Sim is properly initialized and cameras are set up
7. **Import errors**: Make sure all dependencies are installed in the correct Python environment

## Example Instructions

- "Navigate to the goal"
- "Push the box to the goal"
- "Move forward and avoid obstacles"
- "Reach the blue marker"
- "Go through the gate"

