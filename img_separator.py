import os
from PIL import Image
from tqdm import tqdm  
import shutil

# Input and output folder paths
input_folder = "classification_data"  
output_folder_benign = os.path.join(input_folder, "benign")
output_folder_malignant = os.path.join(input_folder, "malignant")

os.makedirs(output_folder_benign, exist_ok=True)
os.makedirs(output_folder_malignant, exist_ok=True)

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for filename in tqdm(os.listdir(input_folder), desc="Sorting images"):
    if filename.lower().endswith(valid_extensions):
        file_path = os.path.join(input_folder, filename)
        try:
            with Image.open(file_path) as img:
                width, height = img.size

                # Check dimensions and move to appropriate folder
                if width < 20 or height < 20:
                    shutil.move(file_path, os.path.join(output_folder_benign, filename))
                else:
                    shutil.move(file_path, os.path.join(output_folder_malignant, filename))
        except Exception as e:
            print(f"Skipping {filename}: {e}")

print(" Sorting complete!")
print(f"Images <20px in width or height → {output_folder_benign}")
print(f"Images ≥20px in width and height → {output_folder_malignant}")