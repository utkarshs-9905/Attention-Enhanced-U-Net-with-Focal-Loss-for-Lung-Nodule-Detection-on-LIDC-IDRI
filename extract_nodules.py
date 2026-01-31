import os
import cv2
import numpy as np

#PATH SETTINGS
data_root = r"C:\Users\Utkarsh\OneDrive\Desktop\ml\ml\attention_unet_lung_nodule\data\LIDC-IDRI"
mask_root = r"C:\Users\Utkarsh\OneDrive\Desktop\ml\ml\attention_unet_lung_nodule\evaluation_results\visualizations"
output_dir = r"C:\Users\Utkarsh\OneDrive\Desktop\ml\ml\attention_unet_lung_nodule\classification_data"

os.makedirs(output_dir, exist_ok=True)

print(f"🔍 Searching recursively in: {data_root}")
count = 0

#RECURSIVE SEARCH
for root, dirs, files in os.walk(data_root):
    if "images" in root.lower():  
        image_files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg'))]
        
        for img_name in image_files:
            img_path = os.path.join(root, img_name)
            
            nodule_dir = os.path.dirname(root)
            mask_dirs = [os.path.join(nodule_dir, d) for d in os.listdir(nodule_dir) if d.lower().startswith('mask')]
            found_mask = None
            combined_mask = None

            for m_dir in mask_dirs:
                candidate = os.path.join(m_dir, img_name)
                if os.path.exists(candidate):
                    m = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    if combined_mask is None:
                        combined_mask = m
                    else:
                        combined_mask = np.maximum(combined_mask, m)
                    found_mask = True

            if combined_mask is None:
                mask_path = os.path.join(mask_root, img_name.replace("slice", "mask"))
                if os.path.exists(mask_path):
                    combined_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if combined_mask is None:
                print(f"[SKIP] No mask found for image: {img_path}")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = combined_mask
            
            if img is None or mask is None:
                print(f"[WARN] Could not read {img_path} or {mask_path}")
                continue
            
            # Threshold mask and find contours
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for j, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < 50:  
                    continue
                
                nodule_crop = img[y:y+h, x:x+w]
                
                patient_id = os.path.basename(os.path.dirname(os.path.dirname(root)))  # e.g., LIDC-IDRI-0412
                save_name = f"{patient_id}_nodule{j}.png"
                save_path = os.path.join(output_dir, save_name)
                
                cv2.imwrite(save_path, nodule_crop)
                count += 1

print(f"\n Done! Saved {count} nodule crops in: {output_dir}")