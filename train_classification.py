# clean single-copy training script
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


def train_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(2)

        expected_in = None
        try:
            conv1 = model.module.model.conv1 if hasattr(model, 'module') else model.model.conv1
            expected_in = conv1.in_channels
        except Exception:
            expected_in = None

        if inputs.dim() == 5 and expected_in is not None:
            # Avoid swapping when depth==1 (2D inputs converted to 3D by unsqueeze)
            if inputs.shape[1] != expected_in and inputs.shape[2] == expected_in and inputs.shape[2] > 1:
                inputs = inputs.permute(0, 2, 1, 3, 4)
            if inputs.shape[1] == 1 and expected_in == 3:
                inputs = inputs.repeat(1, 3, 1, 1, 1)

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
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if inputs.dim() == 4:
                inputs = inputs.unsqueeze(2)

            expected_in = None
            try:
                conv1 = model.module.model.conv1 if hasattr(model, 'module') else model.model.conv1
                expected_in = conv1.in_channels
            except Exception:
                expected_in = None

            if inputs.dim() == 5 and expected_in is not None:
                if inputs.shape[1] != expected_in and inputs.shape[2] == expected_in and inputs.shape[2] > 1:
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                if inputs.shape[1] == 1 and expected_in == 3:
                    inputs = inputs.repeat(1, 3, 1, 1, 1)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    return epoch_loss, epoch_acc, all_preds, all_labels


def get_classification_data_loaders(data_dir, batch_size=16, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    total_size = len(dataset)
    val_size = int(0.15 * total_size)
    test_size = int(0.15 * total_size)
    train_size = total_size - val_size - test_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    loaders = {
        "train": torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        "val": torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        "test": torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
    }
    return loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 for Lung Nodule Classification")
    parser.add_argument("--data-dir", type=str, default="classification_data", help="Path to classification dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/classification")
    parser.add_argument("--log-dir", type=str, default="runs/classification")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--use-focal", action="store_true", help="Use focal loss instead of CE loss")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f" Using GPU: {gpu_name}")
            print(f" GPU Memory Available: {total_mem:.2f} GB")
        except Exception:
            pass
    else:
        print(" CUDA not available — using CPU")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    data_loaders = get_classification_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = NoduleClassifier(num_classes=args.num_classes, use_pretrained=True).to(device)
    if torch.cuda.device_count() > 1:
        print(f" Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    torch.cuda.empty_cache()

    criterion = FocalLoss(alpha=0.25, gamma=2.0) if args.use_focal else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15, verbose=True)

    start_epoch, best_acc = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume)
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f" Resumed training from epoch {start_epoch}")

    writer = SummaryWriter(log_dir=args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, data_loaders["train"], criterion, optimizer, device, epoch, writer)
        val_loss, val_acc, _, _ = validate(model, data_loaders["val"], criterion, device, epoch, writer)

        scheduler.step(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best, checkpoint_dir=args.checkpoint_dir)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(" Early stopping triggered.")
            break

    best_model_path = os.path.join(args.checkpoint_dir, "model_best.pth.tar")
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path)
        model.load_state_dict(ckpt["state_dict"])
        print(f" Loaded best model (epoch {ckpt['epoch']}) with Acc: {ckpt['best_acc']:.4f}")

    test_loss, test_acc, _, _ = validate(model, data_loaders["test"], criterion, device, epoch, writer)
    print(f"\n Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
import yaml

from model_classifier import NoduleClassifier
from losses import FocalLoss
from utils import save_checkpoint, EarlyStopping


# ------------------------------
# Training and validation loops
# ------------------------------
def train_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Debug: print shape for first batch to trace channel ordering problems
        if epoch == 0 and loader.batch_size and hasattr(loader, '__iter__'):
            # print only once per run for first batch
            try:
                print(f"[train] input shape before permute: {tuple(inputs.shape)}")
            except Exception:
                pass

        if inputs.shape[1] != 3:
            inputs = inputs.permute(0, 2, 1, 3, 4)
            if epoch == 0:
                try:
                    print(f"[train] input shape after permute: {tuple(inputs.shape)}")
                except Exception:
                    pass

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
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("train/acc", epoch_acc, epoch)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
        writer.add_scalar("val/loss", epoch_loss, epoch)
        writer.add_scalar("val/acc", epoch_acc, epoch)
    return epoch_loss, epoch_acc, all_preds, all_labels


# ------------------------------
# Dataset Loader
# ------------------------------
def get_classification_data_loaders(data_dir, batch_size=16, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    total_size = len(dataset)
    val_size = int(0.15 * total_size)
    test_size = int(0.15 * total_size)
    train_size = total_size - val_size - test_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    loaders = {
        "train": torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders


# ------------------------------
# Argument Parsing
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 for Lung Nodule Classification")
    parser.add_argument("--data-dir", type=str, default="classification_data", help="Path to classification dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/classification")
    parser.add_argument("--log-dir", type=str, default="runs/classification")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--use-focal", action="store_true", help="Use focal loss instead of CE loss")
    return parser.parse_args()


# ------------------------------
# Main Training Function
# ------------------------------
def main():
    args = parse_args()

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" Using GPU: {gpu_name}")
        print(f" GPU Memory Available: {total_mem:.2f} GB")
    else:
        print(" CUDA not available — using CPU")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # --- Load data ---
    data_loaders = get_classification_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # --- Model ---
    model = NoduleClassifier(num_classes=args.num_classes, use_pretrained=True).to(device)
    if torch.cuda.device_count() > 1:
        print(f" Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    torch.cuda.empty_cache()

    # --- Loss & optimizer ---
    criterion = FocalLoss(alpha=0.25, gamma=2.0) if args.use_focal else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15, verbose=True)

    # --- Checkpoint resume ---
    start_epoch, best_acc = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume)
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f" Resumed training from epoch {start_epoch}")

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=args.log_dir)

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, data_loaders["train"], criterion, optimizer, device, epoch, writer)
        val_loss, val_acc, _, _ = validate(model, data_loaders["val"], criterion, device, epoch, writer)

        scheduler.step(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best, checkpoint_dir=args.checkpoint_dir)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(" Early stopping triggered.")
            break

    # --- Load best model & test ---
    best_model_path = os.path.join(args.checkpoint_dir, "model_best.pth.tar")
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path)
        model.load_state_dict(ckpt["state_dict"])
        print(f" Loaded best model (epoch {ckpt['epoch']}) with Acc: {ckpt['best_acc']:.4f}")

    test_loss, test_acc, _, _ = validate(model, data_loaders["test"], criterion, device, epoch, writer)
    print(f"\n Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
