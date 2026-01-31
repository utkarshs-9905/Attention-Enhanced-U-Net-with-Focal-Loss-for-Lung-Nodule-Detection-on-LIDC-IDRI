import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. Load dataset and merge multiple radiologist masks

def load_png_dataset(data_dir, test_size=0.15, val_size=0.15, random_state=42, 
                    img_size=256, patch_size=64, num_workers=0):

    image_paths = []
    mask_paths = []

    # Traverse through all LIDC-IDRI-xxxx folders
    for case in sorted(os.listdir(data_dir)):
        case_path = os.path.join(data_dir, case)
        if not os.path.isdir(case_path):
            continue

        # Go inside each nodule folder (nodule-0, nodule-1, etc.)
        for nodule_folder in os.listdir(case_path):
            nodule_path = os.path.join(case_path, nodule_folder)
            if not os.path.isdir(nodule_path):
                continue

            images_dir = os.path.join(nodule_path, "images")
            if not os.path.exists(images_dir):
                continue

            # Find all mask-* folders
            mask_dirs = [
                os.path.join(nodule_path, d)
                for d in os.listdir(nodule_path)
                if d.startswith("mask-")
            ]

            # Loop through each image in the images folder
            for file in sorted(os.listdir(images_dir)):
                if not file.endswith(".png"):
                    continue

                img_path = os.path.join(images_dir, file)

                # Combine all masks (logical OR operation)
                combined_mask = None
                for m_dir in mask_dirs:
                    mask_file = os.path.join(m_dir, file)
                    if os.path.exists(mask_file):
                        mask_img = np.array(Image.open(mask_file).convert("L"))
                        if combined_mask is None:
                            combined_mask = mask_img
                        else:
                            combined_mask = np.maximum(combined_mask, mask_img)

                # Save combined mask temporarily
                if combined_mask is not None:
                    tmp_mask_path = os.path.join(nodule_path, f"combined_{file}")
                    Image.fromarray(combined_mask).save(tmp_mask_path)
                    image_paths.append(img_path)
                    mask_paths.append(tmp_mask_path)

    print(f"Found: {len(image_paths)} image-mask pairs")

    if len(image_paths) == 0:
        raise ValueError("No image-mask pairs found. Check dataset path or folder structure.")

    # Split dataset into train, val, test
    img_train_val, img_test, mask_train_val, mask_test = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=random_state
    )
    val_size_adj = val_size / (1 - test_size)
    img_train, img_val, mask_train, mask_val = train_test_split(
        img_train_val, mask_train_val, test_size=val_size_adj, random_state=random_state
    )

    return {
        "train": (img_train, mask_train),
        "val": (img_val, mask_val),
        "test": (img_test, mask_test),
    }


# 2. PyTorch Dataset Class

class LIDCIDRPNGDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256, patch_size=64):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.patch_size = patch_size
        self.transform_img = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.transform_img(image)
        mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()  # binarize mask

        # The model expects 3D volumes (Conv3d) with shape [C, D, H, W].
        # Current images are 2D (C, H, W). Add a singleton depth dimension
        # so each sample becomes a 3D volume with depth=1. DataLoader will
        # batch them to [N, C, D, H, W]. This keeps the code compatible
        # with the 3D UNet implementation.
        image = image.unsqueeze(1)  # (C, 1, H, W)
        mask = mask.unsqueeze(1)    # (C, 1, H, W)

        return image, mask


# 3. Dataloader Function

def get_png_loaders(data_dir, batch_size=4, img_size=256, patch_size=64, num_workers=0):
    datasets = load_png_dataset(data_dir, patch_size=patch_size)

    pin_memory = torch.cuda.is_available()
    loaders = {
        split: DataLoader(
            LIDCIDRPNGDataset(imgs, masks, img_size=img_size, patch_size=patch_size),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split, (imgs, masks) in datasets.items()
    }
    return loaders