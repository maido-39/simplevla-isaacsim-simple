import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from diffusers import UNet2DConditionModel, DDPMScheduler
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
        trajectory_length: int = 16
    ):
        super().__init__()
        self.action_dim = action_dim
        self.trajectory_length = trajectory_length
        self.num_timesteps = num_timesteps
        
        # Create UNet for diffusion
        # Use hidden_dim for cross_attention_dim since we'll project features to hidden_dim
        self.unet = UNet2DConditionModel(
            sample_size=trajectory_length,
            in_channels=action_dim,
            out_channels=action_dim,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(128, 256, 512),
            cross_attention_dim=hidden_dim,  # Match the projected dimension
            attention_head_dim=8,
        )
        
        # Scheduler for diffusion
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
        
        # Project features
        hidden = self.input_proj(features)  # [B, hidden_dim]
        
        # Expand for UNet conditioning
        # UNet expects encoder_hidden_states as [B, seq_len, hidden_dim]
        # We need to expand to match the trajectory length
        hidden = hidden.unsqueeze(1)  # [B, 1, hidden_dim]
        hidden = hidden.expand(-1, self.trajectory_length, -1)  # [B, T, hidden_dim]
        
        if self.training and actions is not None:
            # Training: apply diffusion process
            # Reshape actions for UNet: [B, T, action_dim] -> [B, action_dim, T, 1]
            actions_reshaped = actions.permute(0, 2, 1).unsqueeze(-1)  # [B, action_dim, T, 1]
            
            # Sample noise
            noise = torch.randn_like(actions_reshaped)
            
            # Sample random timesteps
            if timesteps is None:
                timesteps = torch.randint(
                    0, self.scheduler.num_train_timesteps,
                    (batch_size,), device=features.device
                )
            
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
            # Start with random noise
            actions_shape = (batch_size, self.action_dim, self.trajectory_length, 1)
            actions = torch.randn(actions_shape, device=features.device)
            
            # Denoise step by step
            self.scheduler.set_timesteps(self.num_timesteps)
            
            for t in self.scheduler.timesteps:
                # Predict noise
                noise_pred = self.unet(
                    actions,
                    t.unsqueeze(0).expand(batch_size),
                    encoder_hidden_states=hidden
                ).sample
                
                # Denoise
                actions = self.scheduler.step(noise_pred, t, actions).prev_sample
            
            # Reshape back: [B, action_dim, T, 1] -> [B, T, action_dim]
            actions = actions.squeeze(-1).permute(0, 2, 1)
            
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
        
        # Apply LoRA if requested
        if use_lora:
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
        
        # Feature projection to combine vision and language
        # Use bfloat16 to match PaliGemma model dtype
        self.feature_proj = nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16)
        
        # Diffusion action head
        self.action_head = DiffusionActionHead(
            input_dim=hidden_size,
            action_dim=action_dim,
            trajectory_length=trajectory_length
        )
    
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
        device = ego_images.device
        
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
        if self.training and robot_positions is not None:
            noise_pred, noise, timesteps = self.action_head(
                features, robot_positions, timesteps
            )
            return noise_pred, noise, timesteps
        else:
            predicted_positions = self.action_head(features)
            return predicted_positions

