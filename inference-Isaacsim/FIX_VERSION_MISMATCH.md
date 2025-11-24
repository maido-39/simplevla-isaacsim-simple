# Fixing Version Mismatch Issues

## Problem: huggingface_hub / diffusers Version Mismatch

If you see errors like:
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

This is due to a version incompatibility between `diffusers` and `huggingface_hub`.

## Solution

### For Conda Environment (isc-pak)

```bash
# Activate your conda environment
conda activate isc-pak

# Option 1: Install compatible versions
pip install 'diffusers>=0.21.0' 'huggingface_hub>=0.16.0,<0.20.0'

# Option 2: Upgrade both to latest compatible versions
pip install --upgrade diffusers huggingface_hub

# Option 3: If Option 2 doesn't work, try downgrading huggingface_hub
pip install 'huggingface_hub==0.19.4' 'diffusers>=0.21.0'
```

### For Isaac Sim's Python Environment

If you're running in Isaac Sim, you need to install in Isaac Sim's Python:

```bash
# Find Isaac Sim's Python
which python  # in Isaac Sim terminal

# Install compatible versions
/path/to/isaac_sim/python.sh -m pip install 'diffusers>=0.21.0' 'huggingface_hub>=0.16.0,<0.20.0'
```

## Verify Installation

```python
import diffusers
import huggingface_hub
print(f"diffusers version: {diffusers.__version__}")
print(f"huggingface_hub version: {huggingface_hub.__version__}")
```

## Recommended Versions

- `diffusers`: >= 0.21.0, < 0.30.0
- `huggingface_hub`: >= 0.16.0, < 0.20.0

## Why This Happens

The `cached_download` function was deprecated and removed in newer versions of `huggingface_hub` (>= 0.20.0), but older versions of `diffusers` (< 0.21.0) still try to use it.

## Alternative: Use Specific Versions

If you continue to have issues, pin to known working versions:

```bash
pip install diffusers==0.21.4 huggingface_hub==0.19.4
```

## Check Current Versions

```bash
pip list | grep -E "(diffusers|huggingface-hub)"
```

