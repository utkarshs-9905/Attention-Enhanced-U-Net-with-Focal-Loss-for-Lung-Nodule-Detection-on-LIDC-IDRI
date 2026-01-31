import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import argparse
from tqdm import tqdm
import shutil
import yaml

from model_unet import UNet3D_CBAM
from losses import CombinedLoss, dice_loss
from dataset_png import get_png_loaders
from utils import save_checkpoint, load_checkpoint, EarlyStopping

def train_epoch(model, loader, optimizer, criterion, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    for i, (inputs, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}")):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        dice = 1 - dice_loss(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_dice += dice.item() * inputs.size(0)
        
        # Log training progress
        if i % 10 == 0 and writer is not None:
            step = epoch * len(loader) + i
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/dice', dice.item(), step)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_dice = running_dice / len(loader.dataset)
    
    return epoch_loss, epoch_dice

def validate(model, loader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validation"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss and metrics
            loss = criterion(outputs, targets)
            dice = 1 - dice_loss(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_dice += dice.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_dice = running_dice / len(loader.dataset)
    
    # Log validation results
    if writer is not None:
        writer.add_scalar('val/loss', epoch_loss, epoch)
        writer.add_scalar('val/dice', epoch_dice, epoch)
    
    return epoch_loss, epoch_dice

def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D U-Net with CBAM for Lung Nodule Segmentation')
    parser.add_argument('--data-dir', type=str, default='data/LIDC-IDRI', help='Path to the dataset directory')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patch-size', type=int, default=64, help='Size of 3D patches')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/segmentation', 
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='runs/segmentation', 
                        help='Directory to save logs for TensorBoard')
    parser.add_argument('--resume', type=str, default='', 
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--config', type=str, default='configs/segmentation.yaml', 
                        help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config file if provided
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override command line arguments with config file
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Set up device and CUDA optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    else:
        print('CUDA not available. Using CPU')
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up data loaders
    data_loaders = get_png_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.patch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model, loss, and optimizer
    model = UNet3D_CBAM(n_channels=1, n_classes=1)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        torch.cuda.empty_cache() 
    
    criterion = CombinedLoss(alpha=0.7, beta=0.3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15, verbose=True)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        print('-' * 10)
        
        # Train for one epoch
        train_loss, train_dice = train_epoch(
            model, data_loaders['train'], optimizer, criterion, device, epoch, writer
        )
        
        # Validate
        val_loss, val_dice = validate(
            model, data_loaders['val'], criterion, device, epoch, writer
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'\nTrain Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}')
        
        # Save checkpoint
        is_best = val_dice > best_dice
        best_dice = max(val_dice, best_dice)
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'best_dice': best_dice,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=args.checkpoint_dir)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Load best model
    best_model_path = os.path.join(args.checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Loaded best model from epoch {checkpoint["epoch"]} with Dice: {checkpoint["best_dice"]:.4f}')
    
    writer.close()
    
    return model

if __name__ == '__main__':
    model = main()
