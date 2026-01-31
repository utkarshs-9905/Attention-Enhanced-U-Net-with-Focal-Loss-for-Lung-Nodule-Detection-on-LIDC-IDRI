import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_png import get_png_loaders
from model_unet import UNet3D_CBAM

# Metrics

def dice_coefficient(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Evaluation

def evaluate_model(model, loader, device):
    model.eval()
    dice_scores, iou_scores = [], []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            dice = dice_coefficient(preds, masks)
            iou = iou_score(preds, masks)

            dice_scores.append(dice.item())
            iou_scores.append(iou.item())

    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    print(f"\n Evaluation Results:\nDice Score: {mean_dice:.4f} | IoU: {mean_iou:.4f}")
    return mean_dice, mean_iou

# Visualization

def visualize_predictions(model, loader, device, num_samples=3):
    model.eval()
    samples_shown = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            for i, (img, mask) in enumerate(zip(images, masks)):
                plt.subplot(num_samples, 3, i * 3 + 1)
                plt.imshow(np.squeeze(img), cmap='gray')  
                plt.title('Image')
                plt.axis('off')
                plt.subplot(num_samples, 3, i * 3 + 2)
                plt.imshow(np.squeeze(mask), cmap='gray')  
                plt.title('Ground Truth')
                plt.axis('off')
                plt.subplot(num_samples, 3, i * 3 + 3)
                plt.imshow(np.squeeze(preds[i]), cmap='gray')  
                plt.title('Prediction')
                plt.axis('off')

                plt.tight_layout()
                plt.show()

                samples_shown += 1
                if samples_shown >= num_samples:
                    return

# Main

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = r"data/LIDC-IDRI"
    model_path = r"checkpoints/segmentation/model_best.pth.tar"

    # Load data
    loaders = get_png_loaders(data_dir, batch_size=1, img_size=256)
    test_loader = loaders["test"]

    # Load model
    model = UNet3D_CBAM(n_channels=1, n_classes=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    print(f"Loaded model from: {model_path}")

    # Evaluate
    dice, iou = evaluate_model(model, test_loader, device)

    # Visualize
    visualize_predictions(model, test_loader, device, num_samples=3)