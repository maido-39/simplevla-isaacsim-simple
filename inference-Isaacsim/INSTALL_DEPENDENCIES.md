# Installing Dependencies for Isaac Sim VLA Inference

When running VLA inference in Isaac Sim, you need to ensure all dependencies are installed in Isaac Sim's Python environment.

## Finding Isaac Sim's Python

Isaac Sim uses its own Python environment. To find it:

1. **In Isaac Sim terminal/console**, run:
   ```bash
   which python
   ```
   or
   ```bash
   python --version
   ```

2. **Common locations:**
   - Linux: `~/.local/share/ov/pkg/isaac_sim-*/python.sh`
   - Or check: `~/.local/share/ov/pkg/isaac_sim-*/exts/omni.isaac.kit/python/bin/python3`

## Installing Dependencies

Once you know Isaac Sim's Python path, install the required packages:

```bash
# Method 1: Using Isaac Sim's Python directly
/path/to/isaac_sim/python.sh -m pip install peft transformers diffusers torch torchvision pillow numpy matplotlib

# Method 2: If python.sh is in PATH
isaac-sim-python -m pip install peft transformers diffusers torch torchvision pillow numpy matplotlib

# Method 3: Using the Python executable directly
/path/to/isaac_sim/exts/omni.isaac.kit/python/bin/python3 -m pip install peft transformers diffusers torch torchvision pillow numpy matplotlib
```

## Required Packages

The following packages are required:

- `peft` - For LoRA support (required if model uses LoRA)
- `transformers` - For PaliGemma model
- `diffusers` - For diffusion action head
- `torch` - PyTorch (usually pre-installed in Isaac Sim)
- `torchvision` - PyTorch vision utilities
- `pillow` - Image processing
- `numpy` - Numerical operations
- `matplotlib` - For visualization (optional, only for inference analysis)

## Verification

To verify installation, run in Isaac Sim's Python:

```python
import peft
import transformers
import diffusers
print("All dependencies installed successfully!")
```

## Troubleshooting

### ModuleNotFoundError: No module named 'peft'

This means `peft` is not installed in Isaac Sim's Python environment. Follow the installation steps above.

### CUDA/GPU Issues

If you encounter CUDA errors, ensure:
1. Isaac Sim is using the correct CUDA version
2. PyTorch in Isaac Sim matches the CUDA version
3. GPU drivers are up to date

### Import Errors

If you get import errors for other modules:
1. Check if they're installed: `python -m pip list | grep <module_name>`
2. Install missing modules using Isaac Sim's Python
3. Restart Isaac Sim after installation

## Alternative: Use Environment Variables

You can also set `PYTHONPATH` to include your local packages:

```bash
export PYTHONPATH=/path/to/your/packages:$PYTHONPATH
```

But this is less reliable than installing directly in Isaac Sim's environment.

