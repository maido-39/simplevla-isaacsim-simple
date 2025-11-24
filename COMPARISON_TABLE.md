# Pi-Zero vs Simple VLA - Quick Comparison Table

## Architecture Comparison

| Component | Pi-Zero | Simple VLA | Difference |
|-----------|---------|------------|------------|
| **Vision Encoder** | SigLIP ViT (400M) | PaliGemma built-in ViT | Separate vs integrated |
| **Language Model** | Gemma 2B (vision/lang) | PaliGemma 3B | Different sizes |
| **Action Expert** | Gemma 300M (separate) | UNet (1.5M) | LLM vs diffusion head |
| **Action Method** | Flow Matching | Diffusion (DDIM) | Different paradigms |
| **State Input** | ✅ Yes (as token) | ❌ No | Missing in our model |
| **Temporal History** | ❌ No | ✅ Yes (5 steps) | Our advantage |
| **Action Horizon** | 50 steps | 16 steps | Pi-Zero longer |
| **Action Dimension** | 32 (typical) | 3 (x,y,z) | Different spaces |

## Input/Output Dimensions

### Pi-Zero

| Stage | Input Shape | Output Shape | Notes |
|-------|-------------|--------------|-------|
| Vision | `[B, H, W, 3]` | `[B, S_img, 2048]` | Multiple views |
| Language | `[B, S_lang]` | `[B, S_lang, 2048]` | Tokenized |
| Prefix | `[B, S_prefix, 2048]` | `[B, S_prefix, 2048]` | Gemma 2B |
| Action Input | `[B, 50, 32]` | `[B, 50, 1024]` | Projection |
| Action Expert | `[B, 50, 1024]` | `[B, 50, 1024]` | Gemma 300M |
| Action Output | `[B, 50, 1024]` | `[B, 50, 32]` | Projection |
| **Final** | - | `[B, 50, 32]` | Actions |

### Simple VLA

| Stage | Input Shape | Output Shape | Notes |
|-------|-------------|--------------|-------|
| Vision | `[B, 5, 3, 224, 224]` | `[B, 2048]` | History + mean pool |
| Language | Text strings | `[B, 2048]` | Combined with vision |
| Features | `[B, 2048]` | `[B, 2048]` | Projection |
| Action Head | `[B, 2048]` | `[B, 256]` | Input projection |
| UNet | `[B, 3, 16, 1]` | `[B, 3, 16, 1]` | Diffusion |
| **Final** | - | `[B, 16, 3]` | Actions |

## Performance Comparison

| Metric | Pi-Zero | Simple VLA (Current) | Simple VLA (Optimized) |
|--------|---------|---------------------|------------------------|
| **Inference Speed** | ~10-20 Hz | ~0.18 Hz | ~2-3 Hz |
| **Inference Steps** | ~10 | 20 | 20 (could be 10) |
| **Training Steps** | Continuous | 500 | 500 |
| **Model Size (Action)** | ~300M | ~1.5M | ~1.5M |
| **Total Model Size** | ~2.3B | ~3B + 1.5M | ~3B + 1.5M |
| **Memory (Action Head)** | ~1.2 GB | ~6 MB | ~6 MB |

## Key Architectural Differences

### 1. Action Generation

**Pi-Zero (Flow Matching)**:
```
Noise → [Flow Steps] → Actions
  t=1      t=0.9...0.1    t=0
  ↓         ↓              ↓
Predict velocity v_t = noise - actions
Integrate: x_{t+dt} = x_t + dt * v_t
```

**Simple VLA (Diffusion)**:
```
Noise → [Denoising Steps] → Actions
  t=500    t=480...20        t=0
  ↓         ↓                 ↓
Predict noise ε_t
Denoise: x_{t-1} = scheduler.step(ε_t, t, x_t)
```

### 2. Model Architecture

**Pi-Zero**:
```
[Images] → SigLIP → [Tokens]
[Language] → Embed → [Tokens]
[State] → Proj → [Token]
[Actions] → Proj → [Tokens]
         ↓
    [Gemma 2B + 300M]
    (Unified attention)
         ↓
    [Actions]
```

**Simple VLA**:
```
[Images + Language] → PaliGemma → [Features]
                              ↓
                         [UNet Head]
                              ↓
                         [Actions]
```

### 3. Attention Structure

**Pi-Zero**:
- Prefix (images + language): Full attention
- Suffix (state + actions): Isolated from prefix
- Actions: Causal within actions

**Simple VLA**:
- Vision + Language: Processed together
- Actions: Separate head (no attention connection)

## Recommended Improvements

### ✅ High Priority (Easy)

1. **Add State Input**
   - Current: Only images + language
   - Add: Robot state (position, orientation)
   - Impact: Better context for navigation

2. **Increase Action Horizon**
   - Current: 16 steps
   - Target: 32-50 steps
   - Impact: Longer planning horizon

3. **Better Time Embedding**
   - Current: Diffusion timestep
   - Target: Sine-cosine positional encoding
   - Impact: Better temporal understanding

### ⚠️ Medium Priority (Moderate Effort)

4. **Flow Matching**
   - Current: Diffusion (20 steps)
   - Target: Flow matching (10 steps)
   - Impact: 2x faster inference

5. **Separate Image Processing**
   - Current: Concatenate ego+top
   - Target: Process separately then combine
   - Impact: Better spatial understanding

### 🔄 Low Priority (Major Refactoring)

6. **Dual-Expert Architecture**
   - Current: Single PaliGemma
   - Target: Separate action expert
   - Impact: Better action modeling (but complex)

## Task-Specific Notes

### Our Task (Spot Navigation)
- ✅ Temporal history (5 steps) - important for navigation
- ✅ Simpler action space (3D position)
- ✅ Multiple views (ego + top)
- ❌ Missing state input

### Pi-Zero Task (Manipulation)
- ✅ State input
- ✅ Longer horizon (50 steps)
- ✅ Complex action space (32-dim)
- ❌ No temporal history

**Conclusion**: Our architecture is appropriate, but adding state input and increasing horizon would help.

## Implementation Priority

1. **Week 1**: Add state input, increase horizon to 32
2. **Week 2**: Implement flow matching
3. **Week 3+**: Consider dual-expert if needed

