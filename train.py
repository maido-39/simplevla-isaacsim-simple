import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import argparse
from dataset import VLADataset, collate_fn
from model import SimpleVLA
import json
from datetime import datetime


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        ego_images = batch['ego_images'].to(device)
        top_images = batch['top_images'].to(device)
        robot_positions = batch['robot_positions'].to(device)
        instructions = batch['instructions']
        
        # Forward pass
        optimizer.zero_grad()
        
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
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / num_batches})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            ego_images = batch['ego_images'].to(device)
            top_images = batch['top_images'].to(device)
            robot_positions = batch['robot_positions'].to(device)
            instructions = batch['instructions']
            
            # Forward pass
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
    
    return total_loss / num_batches if num_batches > 0 else 0.0


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
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
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
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = SimpleVLA(
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        trajectory_length=args.trajectory_length,
        history_length=args.history_length
    )
    model = model.to(device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
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
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
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
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

