import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys
import argparse
from dataset import VLADataset, collate_fn
from model import SimpleVLA
import json
from datetime import datetime
import logging


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer=None, global_step=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Clear cache before each batch to prevent OOM
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Move to device
        ego_images = batch['ego_images'].to(device)
        top_images = batch['top_images'].to(device)
        robot_positions = batch['robot_positions'].to(device)
        instructions = batch['instructions']
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            noise_pred, noise, timesteps = model(
                ego_images=ego_images,
                top_images=top_images,
                instructions=instructions,
                robot_positions=robot_positions
            )
            
            # Compute loss (MSE between predicted and actual noise)
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            current_step = global_step + batch_idx
            
            # Log to tensorboard
            if writer is not None:
                writer.add_scalar('Train/Loss', loss.item(), current_step)
                # Get learning rate from optimizer
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Train/LearningRate', current_lr, current_step)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / num_batches})
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM at batch {batch_idx}, clearing cache and skipping batch")
                torch.cuda.empty_cache()
                continue
            else:
                raise
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step + len(dataloader)


def validate(model, dataloader, device, epoch, writer=None):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Clear cache before validation
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            try:
                ego_images = batch['ego_images'].to(device)
                top_images = batch['top_images'].to(device)
                robot_positions = batch['robot_positions'].to(device)
                instructions = batch['instructions']
                
                # Forward pass - model will compute loss when robot_positions are provided
                noise_pred, noise, timesteps = model(
                    ego_images=ego_images,
                    top_images=top_images,
                    instructions=instructions,
                    robot_positions=robot_positions
                )
                
                # Compute loss
                loss = nn.functional.mse_loss(noise_pred, noise)
                total_loss += loss.item()
                num_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM at validation batch {batch_idx}, clearing cache and skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
    
    # Clear cache after validation
    torch.cuda.empty_cache()
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Simple VLA Model')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Proto_Data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--history_length', type=int, default=5,
                        help='History length')
    parser.add_argument('--trajectory_length', type=int, default=16,
                        help='Trajectory length')
    parser.add_argument('--task_type', type=str, default=None,
                        choices=['box', 'moving', 'gate', None],
                        help='Task type to train on (None for all)')
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Use LoRA for PaliGemma')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs (default: 1)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for tensorboard logs (default: output_dir/runs)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging to file
    log_file = os.path.join(args.output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Training log will be saved to: {log_file}")
    
    # Setup TensorBoard
    if args.log_dir is None:
        log_dir = os.path.join(args.output_dir, 'runs')
    else:
        log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    logger.info(f"To view TensorBoard, run: tensorboard --logdir {log_dir}")
    
    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    config['log_file'] = log_file
    config['tensorboard_log_dir'] = log_dir
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset
    logger.info("Loading dataset...")
    full_dataset = VLADataset(
        data_root=args.data_root,
        history_length=args.history_length,
        trajectory_length=args.trajectory_length,
        task_type=args.task_type
    )
    
    # Split dataset
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Use smaller batch size for validation to avoid OOM
    # Limit validation to first 50 samples to save memory
    val_subset_size = min(50, len(val_dataset))
    val_subset = torch.utils.data.Subset(val_dataset, list(range(val_subset_size)))
    val_batch_size = 1  # Always use batch_size=1 for validation
    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # No workers to save memory
        pin_memory=False  # Disable pin_memory to save memory
    )
    logger.info(f"Validation subset size: {val_subset_size} (out of {len(val_dataset)} total validation samples)")
    
    # Create model
    logger.info("Creating model...")
    model = SimpleVLA(
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        trajectory_length=args.trajectory_length,
        history_length=args.history_length
    )
    
    # Note: PaliGemma uses device_map="auto" which handles device placement automatically
    # The action_head and feature_proj are already on the correct device (created in __init__)
    model_device = next(model.paligemma.parameters()).device
    logger.info(f"Model device: {model_device}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', start_epoch * len(train_loader))
        logger.info(f"Resuming from epoch {start_epoch}, global_step {global_step}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, writer, global_step
        )
        
        # Validate
        val_loss = validate(model, val_loader, device, epoch, writer)
        
        logger.info(f"\nTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pth'))
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, 'best.pth'))
            logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoints at specified interval
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Close tensorboard writer
    writer.close()
    
    logger.info("\nTraining completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Training log saved to: {log_file}")
    logger.info(f"TensorBoard logs saved to: {log_dir}")


if __name__ == '__main__':
    main()

