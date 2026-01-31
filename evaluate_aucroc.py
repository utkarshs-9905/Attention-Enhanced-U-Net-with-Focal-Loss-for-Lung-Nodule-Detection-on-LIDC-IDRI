import os
import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_classifier import NoduleClassifier


# Load Data

def get_test_loader(data_dir, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, dataset.classes


# Evaluate

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 4:
                inputs = inputs.unsqueeze(2)  # [N, C, 1, H, W]
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of malignant
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


# ROC Curve

def plot_roc(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Lung Nodule Classification")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" ROC curve saved at: {save_path}")
    return roc_auc


# Confusion Matrix

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Confusion matrix saved at: {save_path}")


# Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    checkpoint_path = "checkpoints/classification/model_best.pth.tar"
    test_dir = "classification_data" 
    os.makedirs("evaluation_results", exist_ok=True)

    # Load model
    model = NoduleClassifier(num_classes=2, use_pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(" Loaded trained model")

    # Data
    test_loader, class_names = get_test_loader(test_dir, batch_size=8)
    print(f" Found {len(test_loader.dataset)} test images in {class_names}")

    # Evaluate
    y_true, y_probs = evaluate(model, test_loader, device)
    y_pred = (y_probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = plot_roc(y_true, y_probs, "evaluation_results/roc_curve.png")
    plot_confusion_matrix(y_true, y_pred, class_names, "evaluation_results/confusion_matrix.png")

    print("\n Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC-ROC:   {roc_auc:.4f}")


if __name__ == "__main__":
    main()