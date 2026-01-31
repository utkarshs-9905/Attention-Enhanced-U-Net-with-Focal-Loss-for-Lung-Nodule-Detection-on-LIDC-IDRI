import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm

from model_classifier import NoduleClassifier
from losses import FocalLoss
from utils import save_checkpoint, EarlyStopping


# Training Epoch
def train_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in tqdm(loader, desc=f"Epoch {epoch + 1} [Train]"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(2)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)

    if writer:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)

    return epoch_loss, epoch_acc


# Validation Epoch
def validate(model, loader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if inputs.dim() == 4:
                inputs = inputs.unsqueeze(2)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)

    if writer:
        writer.add_scalar('Val/Loss', epoch_loss, epoch)
        writer.add_scalar('Val/Accuracy', epoch_acc, epoch)

    return epoch_loss, epoch_acc, all_preds, all_labels


# Data Loader
def get_classification_data_loaders(data_dir, batch_size=16, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    total_size = len(dataset)
    val_size = int(0.15 * total_size)
    test_size = int(0.15 * total_size)
    train_size = total_size - val_size - test_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    loaders = {
        'train': torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers, pin_memory=True),
        'val': torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                           num_workers=num_workers, pin_memory=True),
        'test': torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=True),
    }
    return loaders


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='classification_data')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/classification')
    parser.add_argument('--log-dir', type=str, default='runs/classification')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--use-focal', action='store_true')
    return parser.parse_args()


# Main Training 
def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # CUDA optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Data loaders
    data_loaders = get_classification_data_loaders(args.data_dir, args.batch_size, args.num_workers)

    # Model setup
    model = NoduleClassifier(num_classes=args.num_classes, use_pretrained=True).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    # Loss function
    criterion = FocalLoss(alpha=0.25, gamma=2.0) if args.use_focal else nn.CrossEntropyLoss()

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    best_acc = 0.0
    start_epoch = 0

    # Resume from checkpoint
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint.get('best_acc', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training Loop 
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print('-' * 40)

        train_loss, train_acc = train_epoch(model, data_loaders['train'], criterion, optimizer, device, epoch, writer)
        val_loss, val_acc, _, _ = validate(model, data_loaders['val'], criterion, device, epoch, writer)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=args.checkpoint_dir)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(" Early stopping triggered.")
            break

    # Load best model
    best_model_path = os.path.join(args.checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f" Loaded best model (Val Acc = {checkpoint['best_acc']:.4f})")

    writer.close()


if __name__ == '__main__':
    main()