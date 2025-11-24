# Pi-Zero Architecture Analysis - Quick Summary

## Model Structure Overview

### Pi-Zero Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
│  - Images: [B, H, W, 3] (multiple camera views)            │
│  - Language: Tokenized prompt [B, S_lang]                    │
│  - State: [B, action_dim] (robot state)                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              VISION ENCODER (SigLIP ViT)                     │
│  - Variant: So400m/14                                        │
│  - Input: [B, H, W, 3]                                      │
│  - Output: [B, S_img, 2048] (image tokens)                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│         PREFIX: Vision + Language Expert (Gemma 2B)         │
│  - Processes: Image tokens + Language tokens               │
│  - Architecture: Gemma 2B (2048 width, 18 layers)          │
│  - Output: [B, S_prefix, 2048] (hidden states)              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│         SUFFIX: Action Expert (Gemma 300M)                  │
│  - Processes: Noisy actions + Timestep + State              │
│  - Architecture: Gemma 300M (1024 width, 18 layers)         │
│  - Input: [B, action_horizon, action_dim] → [B, 50, 1024]   │
│  - Output: [B, 50, 1024] → [B, 50, 32] (actions)           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              FLOW MATCHING PROCESS                           │
│  - Method: Continuous flow (t ∈ [0, 1])                      │
│  - Velocity: v_t = noise - actions                           │
│  - Integration: x_{t+dt} = x_t + dt * v_t                   │
│  - Steps: ~10 inference steps                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT                                    │
│  - Actions: [B, action_horizon, action_dim]                │
│  - Default: [B, 50, 32]                                      │
└─────────────────────────────────────────────────────────────┘
```

## Key Components Input/Output Dimensions

### 1. Vision Encoder (SigLIP)
```
Input:  [B, H, W, 3]           # Multiple camera images
Output: [B, S_img, 2048]       # Image tokens (S_img varies by image size)
```

### 2. PaliGemma (Vision+Language Expert)
```
Input:  
  - Image tokens: [B, S_img, 2048]
  - Language tokens: [B, S_lang, 2048]
Output: [B, S_prefix, 2048]    # Prefix hidden states
```

### 3. Action Input Processing
```
Input:  [B, 50, 32]             # Noisy actions (action_horizon=50, action_dim=32)
        ↓ action_in_proj
Output: [B, 50, 1024]           # Action tokens
```

### 4. Time Embedding
```
Input:  [B]                     # Timestep scalar (t ∈ [0, 1])
        ↓ posemb_sincos
Output: [B, 1024]              # Time embedding
        ↓ time_mlp (for pi05) or action_time_mlp (for pi0)
Output: [B, 1024]              # Processed time features
```

### 5. Action Expert (Gemma 300M)
```
Input:  [B, 50, 1024]           # Action tokens + time conditioning
Output: [B, 50, 1024]           # Action hidden states
        ↓ action_out_proj
Output: [B, 50, 32]             # Predicted velocity v_t
```

### 6. Flow Matching Integration
```
Input:  x_t [B, 50, 32]         # Current noisy actions
        v_t [B, 50, 32]         # Predicted velocity
        dt = -1.0 / num_steps    # Step size (negative, going from t=1 to t=0)
Output: x_{t+dt} = x_t + dt * v_t
```

## Flow Matching Details

### Training Loss
```python
# Sample timestep from Beta distribution
time = Beta(1.5, 1) * 0.999 + 0.001  # t ∈ [0.001, 0.999]

# Interpolate between noise and actions
x_t = t * noise + (1 - t) * actions

# Target velocity
u_t = noise - actions

# Model predicts velocity
v_t = model(x_t, time, observation)

# Loss: MSE between predicted and true velocity
loss = mean((v_t - u_t)^2)
```

### Inference (Sampling)
```python
# Start from noise
x_t = noise  # t = 1.0

# Euler integration steps
for step in range(num_steps):  # typically 10 steps
    dt = -1.0 / num_steps  # negative step (going backwards in time)
    v_t = model(x_t, t, observation)
    x_t = x_t + dt * v_t
    t = t + dt  # t goes from 1.0 to 0.0

# Final actions
actions = x_t  # when t ≈ 0
```

## Attention Masking Strategy

Pi-Zero uses sophisticated attention masks to control information flow:

```
[Image Tokens] ──┐
                 ├──→ Full attention between each other
[Language Tokens]┘

[State Token] ──→ Isolated (images/language cannot attend to it)

[Action Tokens] ──→ Isolated from prefix
                  └──→ Causal attention within actions (first token isolated)
```

This creates a structure:
- **Prefix**: Images + Language (full attention)
- **Suffix**: State + Actions (isolated from prefix, causal within)

## Comparison with Our Model

| Feature | Pi-Zero | Our Model |
|---------|---------|-----------|
| **Action Method** | Flow Matching | Diffusion |
| **Inference Steps** | ~10 | 20 |
| **Action Horizon** | 50 | 16 |
| **State Input** | Yes | No |
| **History** | Single timestep | 5 timesteps |
| **Architecture** | Dual-expert LLM | Single encoder + UNet |
| **Action Processing** | Tokens in LLM | Separate UNet head |

## Key Insights

1. **Flow Matching is Simpler**: Just predict velocity and integrate
2. **Dual-Expert Design**: Separate models for vision/lang vs actions
3. **Token-Based Actions**: Actions are processed as language model tokens
4. **Attention Masking**: Sophisticated control over information flow
5. **State Integration**: Explicit robot state as input
6. **KV Caching**: Reuses prefix computation during inference

## Recommended Improvements for Our Model

### Quick Wins
1. ✅ Add state input (robot position/orientation)
2. ✅ Increase action horizon to 32-50
3. ✅ Use sine-cosine time embedding

### Medium Effort
4. ⚠️ Try flow matching instead of diffusion
5. ⚠️ Process ego/top images separately

### Major Refactoring
6. 🔄 Consider dual-expert architecture (if needed)

