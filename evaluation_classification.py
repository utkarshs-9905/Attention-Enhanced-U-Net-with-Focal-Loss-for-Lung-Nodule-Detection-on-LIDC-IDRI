import os
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from tqdm import tqdm
from model_classifier import NoduleClassifier


def get_test_loader(data_dir, batch_size=8, num_workers=4):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    total_size = len(dataset)
    test_size = int(0.15 * total_size)
    val_size = int(0.15 * total_size)
    train_size = total_size - val_size - test_size

    _, _, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return test_loader


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(loader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)

        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(2)  # Ensure 5D: [N, C, D, H, W]

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    avg_acc = running_corrects.double() / len(loader.dataset)
    return avg_loss, avg_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification model")
    parser.add_argument('--data-dir', default='classification_data', help='Path to classification data (ImageFolder)')
    parser.add_argument('--checkpoint', default='checkpoints/classification/model_best.pth.tar', help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    print("Loading test data...")
    test_loader = get_test_loader(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load model
    print("Loading trained classifier...")
    model = NoduleClassifier(num_classes=2, use_pretrained=True).to(device)

    # Load checkpoint (robust to map_location and different dict shapes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Handle DataParallel 'module.' prefixes
    def _strip_module_prefix(sd):
        if any(k.startswith('module.') for k in sd.keys()):
            return {k.replace('module.', ''): v for k, v in sd.items()}
        return sd

    state_dict = _strip_module_prefix(state_dict)

    # Load state dict permissively (allow missing/mismatched keys)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        # Fallback: try filtering by exact key and shape match
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

    model.eval()

    criterion = nn.CrossEntropyLoss()

    # Evaluate
    print("Evaluating model...")
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n Test Loss: {test_loss:.4f}")
    
    print(f" Test Accuracy: {float(test_acc):.4f}")

    print("\n Classification Report:")
    print(classification_report(labels, preds, target_names=["Benign", "Malignant"]))

    print("\n Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()