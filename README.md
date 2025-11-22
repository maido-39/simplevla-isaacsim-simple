# Simple VLA Model for Isaac Sim

A simple Vision-Language-Action (VLA) model for robot control tasks, trained on custom Isaac Sim demonstration data.

## Architecture

- **Input**: 
  - History of 5 frames from ego and top cameras
  - Robot positions, orientations, timestamps, and delta time (dt) from CSV
  - Language instructions from task-specific instruction files
  
- **Model**: 
  - PaliGemma (with LoRA fine-tuning) for vision-language encoding
  - Diffusion-based action head for trajectory generation
  
- **Output**: 
  - Future trajectory (robot positions) as a list

## Dataset Structure

The dataset should be organized as follows:
```
20251121_Proto_Data/
├── *_Instruction.txt (instruction files for each person)
├── YYMMDD_HHMMSS-person_task/
│   ├── camera/
│   │   ├── ego/ (JPG images)
│   │   └── top/ (JPG images)
│   └── data.csv (with columns: timestamp, frame_num, robot_pos_x/y/z, robot_orient_w/x/y/z, ...)
```

## Installation

### Using Conda (Recommended)

**Option 1: Using environment.yml (Easiest)**

```bash
# Create and activate conda environment from file
conda env create -f environment.yml
conda activate simplevla
```

**Option 2: Manual setup**

```bash
# Create a new conda environment
conda create -n simplevla python=3.10 -y

# Activate the environment
conda activate simplevla

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

**Note:** Adjust `pytorch-cuda=11.8` to match your CUDA version (e.g., `11.7`, `12.1`, etc.) or remove it for CPU-only installation.

### Hugging Face Authentication

The PaliGemma model is a **gated model** on Hugging Face and requires authentication:

1. **Request Access**: Visit [PaliGemma model page](https://huggingface.co/google/paligemma-3b-mix-224) and click "Agree and access repository"

2. **Login to Hugging Face**:
   ```bash
   # Install huggingface-hub if not already installed
   pip install huggingface-hub
   
   # Login with your Hugging Face token
   huggingface-cli login
   ```
   
   You can get your token from: https://huggingface.co/settings/tokens

3. **Alternative: Use token directly**:
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   ```

### Model Cache Directory

By default, Hugging Face models are downloaded and cached to:
- **Linux/Mac**: `~/.cache/huggingface/hub/`
- **Windows**: `C:\Users\<username>\.cache\huggingface\hub\`

You can customize the cache directory using environment variables:
```bash
# Option 1: Set HF_HOME (sets base directory for all HF files)
export HF_HOME=/path/to/custom/cache

# Option 2: Set TRANSFORMERS_CACHE (only for transformers models)
export TRANSFORMERS_CACHE=/path/to/custom/cache
```

The model code will automatically print the cache directory location when loading.

### Alternative: Using pip only

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py \
    --data_root /direction-to-data/ \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-4 \
    --history_length 5 \
    --trajectory_length 16 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32
```

### Arguments

- `--data_root`: Path to the Proto_Data directory
- `--output_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--batch_size`: Batch size (default: 4)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--history_length`: Number of historical frames (default: 5)
- `--trajectory_length`: Length of output trajectory (default: 16)
- `--task_type`: Filter by task type ('box', 'moving', 'gate', or None for all)
- `--use_lora`: Use LoRA for PaliGemma fine-tuning
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--val_split`: Validation split ratio (default: 0.2)
- `--resume`: Path to checkpoint to resume from
- `--save_interval`: Save checkpoint every N epochs (default: 1, saves every epoch)
- `--log_dir`: Directory for TensorBoard logs (default: output_dir/runs)

## Data Format

### CSV Format
The `data.csv` file should contain:
- `timestamp`: Timestamp for each frame
- `frame_num`: Frame number
- `robot_pos_x`, `robot_pos_y`, `robot_pos_z`: Robot position
- `robot_orient_w`, `robot_orient_x`, `robot_orient_y`, `robot_orient_z`: Robot orientation (quaternion)

The dataset automatically calculates `dt` (delta time) from consecutive timestamps for temporal information.

### Instruction Files
Instruction files should follow the format:
```
Task: Moving
Description: ...
(1) ...
(2) ...

Task: Box
Description: ...
(1) ...
...

Task: Gate
Description: ...
(1) ...
...
```

## Model Components

### Dataset (`dataset.py`)
- `VLADataset`: Loads images, robot states, timestamps, and instructions
- Supports history length and trajectory length configuration
- Automatically calculates delta time (dt) from timestamps

### Model (`model.py`)
- `SimpleVLA`: Main VLA model
- `DiffusionActionHead`: Diffusion-based trajectory generator
- PaliGemma integration with LoRA support

### Training (`train.py`)
- Training loop with validation
- Checkpoint saving (latest, best, periodic)
- Config saving for reproducibility

## Tasks

The model supports three tasks:
1. **Box**: Push box to target location
2. **Moving**: Move to target location
3. **Gate**: Pass through gate and reach target

Each task has demonstrations from 4 different people with corresponding instructions.

## Monitoring and Logging

### TensorBoard

Training metrics are automatically logged to TensorBoard. To view:

```bash
tensorboard --logdir ./checkpoints/runs
```

Then open your browser to `http://localhost:6006`

TensorBoard logs include:
- `Train/Loss`: Training loss per batch
- `Train/LearningRate`: Learning rate per batch
- `Val/Loss`: Validation loss per epoch

### Training Logs

All terminal output is automatically saved to a log file:
- Location: `{output_dir}/training_{timestamp}.log`
- Contains all print statements, errors, and training progress
- Saved with timestamp for easy identification

### Checkpoints

Checkpoints are saved in the `output_dir` directory:
- `latest.pth`: Latest checkpoint (updated every epoch)
- `best.pth`: Best model based on validation loss
- `checkpoint_epoch_{N}.pth`: Checkpoints saved at specified interval (default: every epoch)

Use `--save_interval N` to save checkpoints every N epochs (useful for long training runs).

## Inference

After training, you can use the model for inference to predict trajectories from images and instructions.

### Basic Usage

```bash
python inference.py \
    --checkpoint ./checkpoints/best.pth \
    --ego_images path/to/ego1.jpg path/to/ego2.jpg path/to/ego3.jpg path/to/ego4.jpg path/to/ego5.jpg \
    --top_images path/to/top1.jpg path/to/top2.jpg path/to/top3.jpg path/to/top4.jpg path/to/top5.jpg \
    --instruction "Push the box to the target location" \
    --output trajectory.npy \
    --visualize
```

### Arguments

- `--checkpoint`: Path to model checkpoint file (e.g., `best.pth`, `latest.pth`, or `checkpoint_epoch_N.pth`)
- `--ego_images`: List of paths to ego camera images (must match history_length, default: 5)
- `--top_images`: List of paths to top camera images (must match history_length, default: 5)
- `--instruction`: Instruction text describing the task
- `--output`: (Optional) Path to save predicted trajectory as numpy array (.npy)
- `--visualize`: (Optional) Display trajectory visualization
- `--viz_output`: (Optional) Path to save trajectory visualization image
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

### Example

```bash
# Predict trajectory and save results
python inference.py \
    --checkpoint ./checkpoints/best.pth \
    --ego_images \
        data/ego_frame0.jpg \
        data/ego_frame1.jpg \
        data/ego_frame2.jpg \
        data/ego_frame3.jpg \
        data/ego_frame4.jpg \
    --top_images \
        data/top_frame0.jpg \
        data/top_frame1.jpg \
        data/top_frame2.jpg \
        data/top_frame3.jpg \
        data/top_frame4.jpg \
    --instruction "Move to the target location" \
    --output predicted_trajectory.npy \
    --visualize \
    --viz_output trajectory_plot.png
```

### Output

The inference script outputs:
- **Console**: Predicted trajectory coordinates (x, y, z) for each time step
- **Numpy file** (if `--output` specified): `[T, 3]` array of predicted positions
- **Visualization** (if `--visualize` or `--viz_output` specified): Plot showing x, y, z positions over time

### Using in Python Code

```python
from inference import load_model_from_checkpoint, load_images_from_paths, predict_trajectory
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model_from_checkpoint('./checkpoints/best.pth', device)

# Load images
ego_images, top_images = load_images_from_paths(
    ['ego1.jpg', 'ego2.jpg', 'ego3.jpg', 'ego4.jpg', 'ego5.jpg'],
    ['top1.jpg', 'top2.jpg', 'top3.jpg', 'top4.jpg', 'top5.jpg']
)

# Predict trajectory
instruction = "Push the box to the target location"
trajectory = predict_trajectory(model, ego_images, top_images, instruction, device)

print(f"Predicted trajectory shape: {trajectory.shape}")
print(f"First position: x={trajectory[0, 0]:.4f}, y={trajectory[0, 1]:.4f}, z={trajectory[0, 2]:.4f}")
```

## Output

The model outputs:
- `robot_positions`: [B, T, 3] - Predicted robot positions (x, y, z) as a list forming the trajectory
- Temporal information (timestamps, dt) is included in the dataset for potential future use

