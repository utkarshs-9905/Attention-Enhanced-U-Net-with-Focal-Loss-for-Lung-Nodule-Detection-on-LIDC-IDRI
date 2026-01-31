import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from model_classifier import NoduleClassifier
from PIL import Image
from tqdm import tqdm


# GradCAM helper class

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        gradients = self.gradients[0]       # [C, D, H, W]
        activations = self.activations[0]   # [C, D, H, W]

        weights = gradients.mean(dim=(1, 2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=0)   # [D, H, W]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()

        mid_slice = cam.shape[0] // 2
        return cam[mid_slice, :, :]

# Visualization helper

def show_cam_on_image(img, cam, output_path):
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = np.float32(img) / 255.0
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    cam_result = np.float32(heatmap) * 0.4 + np.float32(img) * 0.6

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, np.uint8(255 * cam_result))

# Main script

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    checkpoint_path = "checkpoints/classification/model_best.pth.tar"
    data_dir = "classification_data"
    output_root = "gradcam_results_all"

    # Load model
    model = NoduleClassifier(num_classes=2, use_pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Fix depth mismatch if needed
    if "model.conv1.weight" in state_dict:
        w = state_dict["model.conv1.weight"]
        model_w = model.model.conv1.weight
        if w.ndim == 5 and model_w.ndim == 5 and w.shape[2] != model_w.shape[2]:
            expected_depth = model_w.shape[2]
            if w.shape[2] > expected_depth:
                mid = w.shape[2] // 2
                start = mid - expected_depth // 2
                end = start + expected_depth
                print(f" Cropping conv1 weight depth {w.shape[2]} → {expected_depth}")
                state_dict["model.conv1.weight"] = w[:, :, start:end, :, :]
            else:
                pad = (expected_depth - w.shape[2]) // 2
                print(f" Padding conv1 weight depth {w.shape[2]} → {expected_depth}")
                state_dict["model.conv1.weight"] = F.pad(w, (0, 0, 0, 0, pad, pad))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f" Model loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")

    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Target layer for Grad-CAM
    target_layer = model.model.layer4[-1].conv3 if hasattr(model.model.layer4[-1], "conv3") else model.model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    # Process both classes
    for cls in ["benign", "malignant"]:
        input_dir = os.path.join(data_dir, cls)
        output_dir = os.path.join(output_root, cls)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n Processing {cls} images...")
        for img_name in tqdm(os.listdir(input_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_gradcam.png")

            # Load and preprocess
            img = cv2.imread(img_path)
            if img is None:
                print(f" Skipping unreadable image: {img_name}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            input_tensor = transform(img_pil).unsqueeze(0).unsqueeze(2).to(device)

            # Generate Grad-CAM
            cam = gradcam.generate_cam(input_tensor)
            show_cam_on_image(img_rgb, cam, output_path)

        print(f" Done processing {cls}. Results saved in {output_dir}")

    print(f"\n All Grad-CAM visualizations saved under: {output_root}")


if __name__ == "__main__":
    main()