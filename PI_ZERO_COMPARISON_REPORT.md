# Pi-Zero vs Simple VLA Model Comparison Report

## Executive Summary

This report analyzes the architecture of [Physical Intelligence's Pi-Zero model](https://github.com/Physical-Intelligence/openpi) and compares it with our Simple VLA model. While both are Vision-Language-Action (VLA) models, they use fundamentally different approaches for action generation.

**Key Finding**: Pi-Zero uses **Flow Matching** with a **dual-expert language model architecture**, while our model uses **Diffusion** with a **separate UNet-based action head**.

## 1. Pi-Zero Model Architecture Analysis

### 1.1 Overall Architecture

```
Input (Images + Language + State)
    ↓
[Vision Encoder: SigLIP ViT]
    ↓
[PaliGemma LLM (Gemma 2B) - Vision+Language Expert]
    ↓
[Action Expert (Gemma 300M) - Action Generation]
    ↓
Output (Action Trajectory)
```

### 1.2 Component Breakdown

#### **1. Vision Encoder (SigLIP)**
- **Input**: Images `[B, H, W, 3]` (multiple camera views)
- **Output**: Image tokens `[B, S_img, D]` where `D = paligemma_config.width`
- **Architecture**: 
  - Variant: `So400m/14` (SigLIP 400M parameters)
  - Pool type: `none` (returns all tokens, not pooled)
  - Output dimension: Matches PaliGemma width (2048 for Gemma 2B)
- **Purpose**: Encodes multiple camera views into token sequences

#### **2. PaliGemma (Vision+Language Expert)**
- **Input**: 
  - Image tokens from SigLIP: `[B, S_img, 2048]`
  - Tokenized prompt: `[B, S_lang, 2048]`
- **Output**: Hidden states `[B, S_prefix, 2048]`
- **Architecture**:
  - Base: Gemma 2B
  - Width: 2048
  - Depth: 18 layers
  - MLP dim: 16,384
  - Num heads: 8
  - Head dim: 256
- **Purpose**: Processes vision and language inputs (prefix tokens)
- **LoRA**: Optional (gemma_2b_lora variant)

#### **3. Action Expert (Smaller Gemma)**
- **Input**: 
  - Noisy actions: `[B, action_horizon, action_dim]`
  - Timestep embedding: `[B, 1024]` (for flow matching)
  - State (optional): `[B, action_dim]`
- **Output**: Action tokens `[B, action_horizon, 1024]`
- **Architecture**:
  - Base: Gemma 300M
  - Width: 1024
  - Depth: 18 layers
  - MLP dim: 4,096
  - Num heads: 8
  - Head dim: 256
- **Purpose**: Generates actions conditioned on vision/language context
- **LoRA**: Optional (gemma_300m_lora variant)

#### **4. Action Processing Layers**
- **Action Input Projection**: `Linear(action_dim → action_expert_width)`
  - Input: `[B, action_horizon, action_dim]` (e.g., 32)
  - Output: `[B, action_horizon, 1024]`
  
- **Time Embedding**: Sine-cosine positional encoding
  - Input: Timestep `[B]` (flow matching time, range [0, 1])
  - Output: `[B, 1024]`
  - Min period: 4e-3, Max period: 4.0
  
- **Time MLP** (for pi05 with adaRMS):
  - `time_mlp_in`: `Linear(1024 → 1024)`
  - `time_mlp_out`: `Linear(1024 → 1024)`
  - Activation: Swish
  
- **Action-Time MLP** (for pi0 without adaRMS):
  - `action_time_mlp_in`: `Linear(2*1024 → 1024)` (concatenates action + time)
  - `action_time_mlp_out`: `Linear(1024 → 1024)`
  - Activation: Swish
  
- **Action Output Projection**: `Linear(action_expert_width → action_dim)`
  - Input: `[B, action_horizon, 1024]`
  - Output: `[B, action_horizon, action_dim]`

#### **5. Flow Matching Process**
- **Method**: Continuous Flow Matching (not discrete diffusion)
- **Timestep**: Continuous `t ∈ [0, 1]` (t=1 is noise, t=0 is target)
- **Noise Schedule**: Beta distribution `Beta(1.5, 1)` scaled to [0.001, 0.999]
- **Interpolation**: `x_t = t * noise + (1-t) * actions`
- **Velocity Prediction**: Model predicts `v_t = noise - actions`
- **Loss**: MSE between predicted and true velocity
- **Inference Steps**: Typically 10 steps (much fewer than diffusion)

### 1.3 Input/Output Dimensions

| Component | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| **SigLIP ViT** | `[B, H, W, 3]` | `[B, S_img, 2048]` | Multiple images → token sequences |
| **PaliGemma Embed** | `[B, S_lang]` (tokenized) | `[B, S_lang, 2048]` | Language tokens |
| **Prefix Tokens** | Image + Language tokens | `[B, S_prefix, 2048]` | Concatenated |
| **Action Input Proj** | `[B, 50, 32]` | `[B, 50, 1024]` | action_horizon=50, action_dim=32 |
| **Time Embedding** | `[B]` (scalar) | `[B, 1024]` | Sine-cosine encoding |
| **Action Expert** | `[B, 50, 1024]` | `[B, 50, 1024]` | Gemma 300M |
| **Action Output Proj** | `[B, 50, 1024]` | `[B, 50, 32]` | Final actions |

### 1.4 Attention Masking Strategy

Pi-Zero uses sophisticated attention masking:
- **Prefix tokens** (images + language): Full attention between each other
- **State token** (if present): Image/language cannot attend to it
- **Action tokens**: Image/language/state cannot attend to actions
- **Action tokens**: First action token is isolated, subsequent tokens can attend to previous actions

This creates a causal structure: `[Images + Language] → [State] → [Actions]`

## 2. Simple VLA Model Architecture Analysis

### 2.1 Overall Architecture

```
Input (Ego Images + Top Images + Language)
    ↓
[PaliGemma (Gemma 3B) - Vision+Language Encoder]
    ↓
[Feature Projection]
    ↓
[Diffusion Action Head (UNet)]
    ↓
Output (Action Trajectory)
```

### 2.2 Component Breakdown

#### **1. PaliGemma (Vision+Language Encoder)**
- **Input**: 
  - Combined images (ego + top): `[B, H, C, 224, 224]`
  - Instructions: List of strings
- **Output**: Hidden states `[B, hidden_size]` (mean pooled)
- **Architecture**:
  - Base: PaliGemma 3B Mix 224
  - Hidden size: 2048
  - Uses LoRA for fine-tuning
- **Purpose**: Encodes vision and language into a single feature vector

#### **2. Feature Projection**
- **Input**: `[B, 2048]` (from PaliGemma)
- **Output**: `[B, 2048]` (bfloat16)
- **Architecture**: `Linear(2048 → 2048)`
- **Purpose**: Project features (currently identity-like, could be used for dimension adjustment)

#### **3. Diffusion Action Head**
- **Input**: Features `[B, 2048]`
- **Output**: Trajectory `[B, trajectory_length, action_dim]`
- **Architecture**:
  - **Input Projection**: `Linear(2048 → 256)`
  - **UNet**: 
    - Sample size: 16 (trajectory length)
    - In/Out channels: 3 (action_dim)
    - Block channels: (32, 64) - optimized
    - Cross attention dim: 256
    - Attention head dim: 2
    - Down/Up blocks: 2 each
  - **Scheduler**: DDIM (20 inference steps) or DDPM (500 training steps)
- **Purpose**: Generates action trajectory via diffusion

### 2.3 Input/Output Dimensions

| Component | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| **PaliGemma** | Images `[B, 5, 3, 224, 224]` + Text | `[B, 2048]` | Mean pooled hidden states |
| **Feature Proj** | `[B, 2048]` | `[B, 2048]` | Identity-like projection |
| **Action Head Input Proj** | `[B, 2048]` | `[B, 256]` | Projects to UNet hidden dim |
| **UNet Conditioning** | `[B, 256]` | `[B, 16, 256]` | Expanded to trajectory length |
| **UNet** | `[B, 3, 16, 1]` (noisy actions) | `[B, 3, 16, 1]` | Denoised actions |
| **Final Output** | `[B, 3, 16, 1]` | `[B, 16, 3]` | Reshaped trajectory |

## 3. Detailed Comparison Table

| Aspect | Pi-Zero | Simple VLA | Notes |
|--------|---------|------------|-------|
| **Action Generation Method** | Flow Matching | Diffusion (DDIM/DDPM) | Flow matching is faster, diffusion is more established |
| **Vision Encoder** | SigLIP ViT (400M) | PaliGemma's built-in ViT | Both use ViT, but Pi-Zero has separate encoder |
| **Language Model** | Dual Expert: Gemma 2B (vision/lang) + Gemma 300M (action) | Single: PaliGemma 3B | Pi-Zero uses specialized action expert |
| **Action Processing** | Actions as tokens in LLM | Separate UNet diffusion head | Pi-Zero integrates actions into language model |
| **State Input** | Included as suffix token | Not used | Pi-Zero explicitly models robot state |
| **Time Conditioning** | Sine-cosine embedding + MLP | Diffusion timestep in UNet | Different approaches to time |
| **Attention Structure** | Sophisticated masking (prefix/suffix) | Not applicable (separate head) | Pi-Zero uses attention to connect vision/lang/actions |
| **Action Horizon** | 50 steps | 16 steps | Pi-Zero predicts longer horizons |
| **Action Dimension** | 32 (typical) | 3 (x, y, z) | Different action spaces |
| **Inference Steps** | ~10 (flow matching) | 20 (DDIM) | Both much faster than 1000-step diffusion |
| **Training Steps** | Continuous flow (t ∈ [0,1]) | 500 discrete timesteps | Flow matching is continuous |
| **Model Size (Action Head)** | ~300M (Gemma 300M) | ~1.5M (UNet) | Pi-Zero action expert is much larger |
| **Total Model Size** | ~2.3B (2B + 300M) | ~3B (PaliGemma) + 1.5M (UNet) | Similar total size, different distribution |
| **LoRA Usage** | Optional on both experts | Applied to PaliGemma | Both use LoRA for efficiency |
| **History Length** | Single timestep images | 5 timesteps (history) | Our model uses temporal history |
| **Image Processing** | Multiple camera views separately | Combined ego+top views | Different multi-view strategies |

## 4. Key Architectural Differences

### 4.1 Action Generation Paradigm

**Pi-Zero (Flow Matching)**:
- Continuous flow from noise to actions
- Actions processed as tokens in language model
- Direct integration with vision/language through attention
- Faster inference (~10 steps)

**Simple VLA (Diffusion)**:
- Discrete diffusion process
- Actions generated by separate UNet
- Vision/language features condition UNet via cross-attention
- Slower but more established method

### 4.2 Model Architecture

**Pi-Zero**:
- **Dual-expert design**: Separate models for vision/language vs actions
- **Unified attention**: Everything in one transformer with masking
- **Token-based actions**: Actions are language model tokens
- **State integration**: Robot state as explicit input

**Simple VLA**:
- **Single encoder**: PaliGemma handles vision+language
- **Separate head**: Diffusion UNet for actions
- **Feature-based**: Actions from feature vectors, not tokens
- **No explicit state**: Only images and language

### 4.3 Temporal Modeling

**Pi-Zero**:
- Single timestep images
- Action horizon: 50 steps
- No explicit history

**Simple VLA**:
- History length: 5 timesteps
- Action horizon: 16 steps
- Explicit temporal history in images

## 5. Recommended Improvements for Simple VLA

Based on the Pi-Zero architecture analysis, here are recommended improvements:

### 5.1 High Priority

#### **1. Add Robot State Input**
- **Current**: Only images and language
- **Recommendation**: Add robot state (position, orientation, etc.) as explicit input
- **Implementation**: 
  - Add state projection layer: `Linear(state_dim → hidden_dim)`
  - Concatenate with vision/language features
  - Or use as separate conditioning in UNet

#### **2. Consider Flow Matching Instead of Diffusion**
- **Current**: Diffusion with 20 inference steps (~0.3-0.5s)
- **Recommendation**: Try Flow Matching (could be 5-10 steps, ~0.1-0.2s)
- **Benefits**: 
  - Faster inference
  - Continuous timesteps (smoother)
  - Simpler implementation
- **Trade-off**: Less established, may need retraining

#### **3. Increase Action Horizon**
- **Current**: 16 steps
- **Recommendation**: Increase to 32-50 steps (like Pi-Zero)
- **Benefit**: Longer planning horizon for complex tasks

### 5.2 Medium Priority

#### **4. Dual-Expert Architecture (Optional)**
- **Current**: Single PaliGemma for everything
- **Recommendation**: Consider smaller action-specific model
- **Implementation**: 
  - Keep PaliGemma for vision/language
  - Add smaller transformer (e.g., Gemma 300M) for actions
  - Use attention to connect them
- **Trade-off**: More complex, but potentially better action modeling

#### **5. Improve Time Conditioning**
- **Current**: Diffusion timestep in UNet
- **Recommendation**: Use sine-cosine positional encoding (like Pi-Zero)
- **Implementation**: 
  - Replace timestep embedding with `posemb_sincos`
  - Add MLP for time processing
  - Better temporal understanding

#### **6. Better Multi-View Image Processing**
- **Current**: Concatenate ego+top images
- **Recommendation**: Process separately then combine (like Pi-Zero)
- **Implementation**:
  - Process ego and top images separately through vision encoder
  - Combine token sequences
  - Better spatial understanding

### 5.3 Low Priority / Future Work

#### **7. Attention Masking for Better Control**
- **Current**: Not applicable (separate head)
- **Recommendation**: If moving to unified architecture, implement attention masking
- **Benefit**: Better control over what attends to what

#### **8. Adaptive Normalization (adaRMS)**
- **Current**: Standard normalization
- **Recommendation**: Consider adaRMS for time conditioning (like Pi05)
- **Benefit**: Better integration of temporal information

#### **9. KV Caching for Inference**
- **Current**: Full forward pass each time
- **Recommendation**: Implement KV cache for prefix (vision/language)
- **Benefit**: Faster inference when only actions change

## 6. Implementation Recommendations

### 6.1 Quick Wins (Easy to Implement)

1. **Add State Input** (1-2 hours)
   ```python
   # Add to model
   self.state_proj = nn.Linear(state_dim, hidden_size)
   # Concatenate with features before action head
   ```

2. **Increase Action Horizon** (5 minutes)
   ```python
   trajectory_length: int = 32  # or 50
   ```

3. **Better Time Embedding** (2-3 hours)
   ```python
   # Replace diffusion timestep with sine-cosine
   time_emb = posemb_sincos(timestep, hidden_dim, 4e-3, 4.0)
   ```

### 6.2 Medium Effort (1-2 days)

4. **Flow Matching** (1-2 days)
   - Replace diffusion scheduler with flow matching
   - Implement continuous timestep sampling
   - Update loss function

5. **Separate Image Processing** (4-6 hours)
   - Process ego and top images separately
   - Combine token sequences
   - Update attention mechanism

### 6.3 Major Refactoring (1-2 weeks)

6. **Dual-Expert Architecture** (1-2 weeks)
   - Add action expert transformer
   - Implement unified attention
   - Redesign model architecture

## 7. Task-Specific Considerations

### Our Task (Spot Robot Navigation)
- **Action space**: 3D position (x, y, z) - simpler than Pi-Zero's 32-dim
- **Temporal history**: We use 5 timesteps - important for navigation
- **Multiple views**: Ego + top view - different from Pi-Zero's wrist/base cameras
- **Tasks**: Box, moving, gate - simpler manipulation tasks

### Pi-Zero Task (Manipulation)
- **Action space**: 32-dim (joint positions, gripper, etc.)
- **Single timestep**: No explicit history
- **Multiple views**: Base + wrist cameras
- **Tasks**: Complex manipulation (ALOHA, DROID datasets)

**Conclusion**: Our architecture is appropriate for our task, but could benefit from state input and flow matching for speed.

## 8. Performance Comparison

| Metric | Pi-Zero | Simple VLA (Current) | Simple VLA (Optimized) |
|--------|---------|---------------------|------------------------|
| **Inference Speed** | ~10-20 Hz | ~0.18 Hz | ~2-3 Hz (estimated) |
| **Inference Steps** | ~10 | 20 | 20 (could be 10 with flow matching) |
| **Model Size** | ~2.3B | ~3B + 1.5M | ~3B + 1.5M |
| **Action Horizon** | 50 | 16 | 16 (could be 32-50) |
| **State Input** | Yes | No | Could add |
| **Temporal History** | No | Yes (5 steps) | Yes (5 steps) |

## 9. Conclusion

### Strengths of Our Model
1. ✅ **Temporal history**: 5 timesteps provide better context
2. ✅ **Simpler architecture**: Easier to understand and debug
3. ✅ **Smaller action head**: More memory efficient
4. ✅ **Task-appropriate**: Good for navigation tasks

### Strengths of Pi-Zero
1. ✅ **Faster inference**: Flow matching is faster
2. ✅ **State integration**: Explicit robot state modeling
3. ✅ **Longer horizon**: 50 steps vs 16
4. ✅ **Unified architecture**: Everything in one transformer

### Recommended Next Steps
1. **Immediate**: Add state input, increase action horizon to 32
2. **Short-term**: Implement flow matching for faster inference
3. **Long-term**: Consider dual-expert if action modeling needs improvement

The architectures are fundamentally different but both valid. Our model is well-suited for our navigation task, and the recommended improvements would bring it closer to Pi-Zero's performance while maintaining its simplicity.

