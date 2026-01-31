from dataset_png import get_png_loaders
import matplotlib.pyplot as plt

data_dir = r"C:\Users\Utkarsh\OneDrive\Desktop\ml\ml\attention_unet_lung_nodule\data\LIDC-IDRI"  

# Load dataset
print(f"Searching in: {data_dir}")
loaders = get_png_loaders(data_dir, batch_size=2, img_size=256)

# Print dataset sizes
dataset_sizes = {split: len(loader.dataset) for split, loader in loaders.items()}
print("Dataset sizes:", dataset_sizes)

# Check one batch from the training loader
train_loader = loaders["train"]
for imgs, masks in train_loader:
    print("Image batch shape:", imgs.shape)
    print("Mask batch shape:", masks.shape)
    
    # Visualize the first image and its mask
    img = imgs[0][0].numpy()
    mask = masks[0][0].numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("CT Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Merged Mask")

    plt.tight_layout()
    plt.show()
    break