import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

# Try to import peft - required for LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft module not available. LoRA functionality will be disabled.")
    print("To install: pip install peft")

# Try to import diffusers - handle version compatibility issues
# Note: robomimic requires diffusers==0.11.1 and huggingface_hub==0.23.4
# We need to work with these versions if robomimic is installed
try:
    from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    DIFFUSERS_AVAILABLE = False
    error_msg = str(e)
    print(f"Warning: diffusers module not available or has compatibility issues: {e}")
    
    # Check if it's the cached_download issue
    if "cached_download" in error_msg or "cannot import name" in error_msg:
        print("\nThis is likely a version mismatch between diffusers and huggingface_hub.")
        print("If you have robomimic installed, it requires specific versions:")
        print("  - diffusers==0.11.1")
        print("  - huggingface_hub==0.23.4")
        print("\nTry installing these exact versions:")
        print("  pip install diffusers==0.11.1 huggingface_hub==0.23.4")
        print("\nNote: This may show dependency warnings, but should work.")
        print("See FIX_ROBOMIMIC_CONFLICT.md for more options.")
    else:
        print("Try: pip install diffusers huggingface_hub")
    
    # Create dummy classes for type hints
    UNet2DConditionModel = None
    DDPMScheduler = None
    DDIMScheduler = None

from typing import Optional, Tuple
import math
import os


class DiffusionActionHead(nn.Module):
    """
    Diffusion-based action head for trajectory generation.
    Predicts future robot positions using diffusion process.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int = 3,  # x, y, z position
        hidden_dim: int = 256,
        num_timesteps: int = 1000,
        trajectory_length: int = 16,
        num_inference_steps: int = 50,  # Number of inference steps (can be much less than num_timesteps)
        use_ddim: bool = True,  # Use DDIM for faster inference
        unet_channels: tuple = (32, 64)  # Further reduced UNet channels
    ):
        super().__init__()
        
        # Check if diffusers is available
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers module is required for DiffusionActionHead but is not available.\n"
                "This is likely due to a version mismatch between diffusers and huggingface_hub.\n"
                "Please install compatible versions:\n"
                "  pip install 'diffusers>=0.21.0' 'huggingface_hub>=0.16.0,<0.20.0'\n"
                "Or try:\n"
                "  pip install --upgrade diffusers huggingface_hub"
            )
        
        self.action_dim = action_dim
        self.trajectory_length = trajectory_length
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        self.use_ddim = use_ddim
        
        # Create UNet for diffusion
        # Use hidden_dim for cross_attention_dim since we'll project features to hidden_dim
        # Further reduced size: smaller channels and fewer blocks
        self.unet = UNet2DConditionModel(
            sample_size=trajectory_length,
            in_channels=action_dim,
            out_channels=action_dim,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=unet_channels,  # Further reduced: (32, 64) instead of (64, 128)
            cross_attention_dim=hidden_dim,  # Match the projected dimension
            attention_head_dim=2,  # Further reduced from 4
        )
        
        # Scheduler for diffusion
        # Use DDIM for faster inference with fewer steps
        if use_ddim:
            self.scheduler = DDIMScheduler(
                num_train_timesteps=num_timesteps,
                beta_schedule="linear",
                prediction_type="epsilon"
            )
        else:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_timesteps,
                beta_schedule="linear",
                prediction_type="epsilon"
            )
        
        # Projection layer to match UNet input
        # Use float32 for UNet (diffusers UNet works better with float32)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, feature_dim] - features from PaliGemma
            actions: [B, T, action_dim] - ground truth actions (for training)
            timesteps: [B] - diffusion timesteps (for training)
        
        Returns:
            actions: [B, T, action_dim] - predicted actions
        """
        batch_size = features.shape[0]
        
        # Convert features to float32 for UNet (UNet works better with float32)
        features = features.to(torch.float32)
        
        # Get device for UNet (may be different from features device if action_head is on CPU)
        unet_device = next(self.unet.parameters()).device
        
        # Project features
        hidden = self.input_proj(features)  # [B, hidden_dim]
        
        # Move to UNet device if needed
        if hidden.device != unet_device:
            hidden = hidden.to(unet_device)
        
        # Expand for UNet conditioning
        # UNet expects encoder_hidden_states as [B, seq_len, hidden_dim]
        # We need to expand to match the trajectory length
        hidden = hidden.unsqueeze(1)  # [B, 1, hidden_dim]
        hidden = hidden.expand(-1, self.trajectory_length, -1)  # [B, T, hidden_dim]
        
        if actions is not None:
            # Training/Validation: apply diffusion process when ground truth is available
            # Get UNet device
            unet_device = next(self.unet.parameters()).device
            
            # Reshape actions for UNet: [B, T, action_dim] -> [B, action_dim, T, 1]
            actions_reshaped = actions.permute(0, 2, 1).unsqueeze(-1)  # [B, action_dim, T, 1]
            
            # Move to UNet device if needed
            if actions_reshaped.device != unet_device:
                actions_reshaped = actions_reshaped.to(unet_device)
            
            # Sample noise
            noise = torch.randn_like(actions_reshaped)
            
            # Sample random timesteps
            if timesteps is None:
                timesteps = torch.randint(
                    0, self.scheduler.num_train_timesteps,
                    (batch_size,), device=unet_device
                )
            
            # Ensure timesteps are on correct device
            timesteps = timesteps.to(unet_device)
            
            # Add noise to actions
            noisy_actions = self.scheduler.add_noise(actions_reshaped, noise, timesteps)
            
            # Predict noise
            noise_pred = self.unet(
                noisy_actions,
                timesteps,
                encoder_hidden_states=hidden
            ).sample
            
            return noise_pred, noise, timesteps
        else:
            # Inference: denoising process
            # Get UNet device
            unet_device = next(self.unet.parameters()).device
            
            # Start with random noise
            actions_shape = (batch_size, self.action_dim, self.trajectory_length, 1)
            actions = torch.randn(actions_shape, device=unet_device, dtype=torch.float32)
            
            # Denoise step by step
            # Use fewer inference steps for faster generation
            # Set timesteps with num_inference_steps (much less than num_timesteps)
            self.scheduler.set_timesteps(self.num_inference_steps, device=unet_device)
            
            # Get timesteps (should already be on device from set_timesteps)
            timesteps = self.scheduler.timesteps
            
            for t in timesteps:
                # Ensure t is on the correct device
                t = t.to(unet_device)
                # Predict noise
                noise_pred = self.unet(
                    actions,
                    t.unsqueeze(0).expand(batch_size).to(unet_device),
                    encoder_hidden_states=hidden
                ).sample
                
                # Denoise - ensure all inputs are on same device
                step_output = self.scheduler.step(noise_pred, t.to(unet_device), actions)
                actions = step_output.prev_sample
            
            # Reshape back: [B, action_dim, T, 1] -> [B, T, action_dim]
            actions = actions.squeeze(-1).permute(0, 2, 1)
            
            # Move back to features device if needed
            if actions.device != features.device:
                actions = actions.to(features.device)
            
            return actions


class SimpleVLA(nn.Module):
    """
    Simple Vision-Language-Action model.
    Architecture: Input -> PaliGemma -> Diffusion Action Head -> Trajectory
    """
    
    def __init__(
        self,
        paligemma_model_name: str = "google/paligemma-3b-mix-224",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        trajectory_length: int = 16,
        action_dim: int = 3,
        history_length: int = 5
    ):
        super().__init__()
        self.trajectory_length = trajectory_length
        self.action_dim = action_dim
        self.history_length = history_length
        
        # Load PaliGemma model
        print(f"Loading PaliGemma model: {paligemma_model_name}")
        
        # Use project directory for model cache
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "models_cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Model cache directory: {cache_dir}")
        print(f"Model will be downloaded/cached to: {cache_dir}/hub/")
        
        # Get Hugging Face token from environment or use default
        token = os.getenv("HF_TOKEN", None)
        if token is None:
            # Try to get from huggingface_hub
            try:
                from huggingface_hub import HfFolder
                token = HfFolder.get_token()
            except:
                pass
        
        # Check if model already exists in cache
        model_name_safe = paligemma_model_name.replace('/', '--')
        model_cache_path = os.path.join(cache_dir, "hub", f"models--{model_name_safe}")
        
        if os.path.exists(model_cache_path):
            print(f"Found cached model at {model_cache_path}, will use cached version")
        else:
            print("Model not found in cache, will download from Hugging Face")
        
        # Load processor and model with token if available
        # Hugging Face will automatically use cached files if they exist
        # Note: device_map="auto" handles device placement automatically, so we don't need to call .to(device) later
        if token:
            print("Using Hugging Face token for authentication")
            self.processor = PaliGemmaProcessor.from_pretrained(
                paligemma_model_name,
                token=token,
                cache_dir=cache_dir
            )
            self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
                paligemma_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token,
                cache_dir=cache_dir
            )
        else:
            print("Warning: No HF_TOKEN found. Make sure you're logged in with 'huggingface-cli login'")
            self.processor = PaliGemmaProcessor.from_pretrained(
                paligemma_model_name,
                cache_dir=cache_dir
            )
            self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
                paligemma_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=cache_dir
            )
        
        # Store device info - device_map="auto" may place model on multiple devices
        # Get the primary device (usually the first CUDA device)
        self.device = next(self.paligemma.parameters()).device
        print(f"PaliGemma model placed on device: {self.device}")
        
        # Apply LoRA if requested
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "LoRA requested but 'peft' module is not available. "
                    "Please install it with: pip install peft\n"
                    "Or set use_lora=False if you don't need LoRA support."
                )
            print("Applying LoRA to PaliGemma...")
            # PaliGemma is based on Gemma (causal LM), so use CAUSAL_LM task type
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            self.paligemma = get_peft_model(self.paligemma, lora_config)
            self.paligemma.print_trainable_parameters()
            
            # Enable gradient checkpointing to save memory
            if hasattr(self.paligemma, 'gradient_checkpointing_enable'):
                self.paligemma.gradient_checkpointing_enable()
            elif hasattr(self.paligemma.base_model, 'gradient_checkpointing_enable'):
                self.paligemma.base_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled to save memory")
        
        # Get feature dimension from PaliGemma
        # PaliGemma uses Gemma language model with vision encoder
        # We'll extract features from the language model's hidden states
        hidden_size = self.paligemma.config.text_config.hidden_size
        
        # Get device from PaliGemma (may be on CUDA if device_map="auto" placed it there)
        model_device = next(self.paligemma.parameters()).device
        
        # Feature projection to combine vision and language
        # Use bfloat16 to match PaliGemma model dtype
        # Create on CPU first, then move to device (to avoid OOM during initialization)
        self.feature_proj = nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16)
        
        # Diffusion action head
        # Create on CPU first, then move to device (to avoid OOM during initialization)
        # Use optimized settings: fewer timesteps, fewer inference steps, smaller UNet
        self.action_head = DiffusionActionHead(
            input_dim=hidden_size,
            action_dim=action_dim,
            trajectory_length=trajectory_length,
            num_timesteps=500,  # Reduced from 1000 for faster training
            num_inference_steps=20,  # Only 20 steps for inference (much faster!)
            use_ddim=True,  # Use DDIM for faster inference
            unet_channels=(32, 64)  # Smaller UNet: (32, 64) instead of (64, 128)
        )
        
        # Move to device after creation (this is more memory efficient)
        try:
            self.feature_proj = self.feature_proj.to(model_device)
            self.action_head = self.action_head.to(model_device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Warning: Could not move action_head to {model_device} due to OOM.")
                print("Action head will remain on CPU (slower but will work)")
                # Keep on CPU if OOM
                pass
            else:
                raise
    
    def encode_images_and_text(
        self,
        ego_images: torch.Tensor,
        top_images: torch.Tensor,
        instructions: list
    ) -> torch.Tensor:
        """
        Encode images and text using PaliGemma.
        
        Args:
            ego_images: [B, H, C, H_img, W_img] - history of ego images
            top_images: [B, H, C, H_img, W_img] - history of top images
            instructions: List of instruction strings
        
        Returns:
            features: [B, hidden_size] - combined features
        """
        batch_size = ego_images.shape[0]
        # Use the device from PaliGemma model
        device = self.device if hasattr(self, 'device') else ego_images.device
        
        # Process images and text with processor
        # Combine ego and top images
        combined_images = []
        for b in range(batch_size):
            # Stack ego and top images for each timestep
            image_list = []
            for h in range(self.history_length):
                # Combine ego and top images side by side
                ego_img = ego_images[b, h].cpu().permute(1, 2, 0).numpy()
                top_img = top_images[b, h].cpu().permute(1, 2, 0).numpy()
                
                # Convert to PIL Images
                from PIL import Image
                ego_pil = Image.fromarray((ego_img * 255).astype('uint8'))
                top_pil = Image.fromarray((top_img * 255).astype('uint8'))
                
                # Concatenate horizontally
                combined = Image.new('RGB', (ego_pil.width * 2, ego_pil.height))
                combined.paste(ego_pil, (0, 0))
                combined.paste(top_pil, (ego_pil.width, 0))
                combined_images.append(combined)
        
        # Process with PaliGemma processor
        # For simplicity, use the last image in history
        # In practice, you might want to process all images and aggregate
        # Add <image> token to instructions as required by PaliGemma
        instructions_with_image = [f"<image>\n{instr}" for instr in instructions]
        
        inputs = self.processor(
            text=instructions_with_image,
            images=[combined_images[-1] for _ in range(batch_size)],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Get features from PaliGemma
        # We'll use the encoder outputs
        # Note: We need gradients for training, so don't use no_grad()
        # Use torch.cuda.amp for mixed precision if available
        outputs = self.paligemma(**inputs, output_hidden_states=True)
        
        # Extract features from the last hidden state
        # PaliGemma returns hidden states from the language model
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_size]
            # Use mean pooling over sequence length
            features = hidden_states.mean(dim=1)  # [B, hidden_size]
        else:
            # Fallback: use logits or other outputs
            if hasattr(outputs, 'logits'):
                features = outputs.logits.mean(dim=1)
            else:
                # Create dummy features
                features = torch.zeros(
                    batch_size,
                    self.paligemma.config.text_config.hidden_size,
                    device=device,
                    dtype=torch.bfloat16
                )
        
        # Ensure features are in bfloat16
        if features.dtype != torch.bfloat16:
            features = features.to(torch.bfloat16)
        
        # Project features
        features = self.feature_proj(features)
        
        return features
    
    def forward(
        self,
        ego_images: torch.Tensor,
        top_images: torch.Tensor,
        instructions: list,
        robot_positions: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            ego_images: [B, H, C, H_img, W_img]
            top_images: [B, H, C, H_img, W_img]
            instructions: List of instruction strings
            robot_positions: [B, T, action_dim] - ground truth (for training)
            timesteps: [B] - diffusion timesteps (for training)
        
        Returns:
            predicted_positions: [B, T, action_dim] - predicted trajectory
        """
        # Encode images and text
        features = self.encode_images_and_text(ego_images, top_images, instructions)
        
        # Generate trajectory using diffusion head
        # If robot_positions are provided, compute training loss (for both train and eval)
        if robot_positions is not None:
            noise_pred, noise, timesteps = self.action_head(
                features, robot_positions, timesteps
            )
            return noise_pred, noise, timesteps
        else:
            # Pure inference - no ground truth
            predicted_positions = self.action_head(features)
            return predicted_positions

