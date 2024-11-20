import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from augmentation import CLAHEAugmentation, EqualizeHistAugmentation

def apply_clahe(image):
    return CLAHEAugmentation.apply_clahe(image)

def apply_equalize_hist(image):
    return EqualizeHistAugmentation.apply_equalize_hist(image)

def visualize_and_save(image_path, save_dir, use_clahe=False, use_eh=False):
    image = Image.open(image_path).convert("RGB")  
    image_np = np.array(image)

    if use_clahe:
        clahe_image_np = apply_clahe(image_np)
        clahe_image = Image.fromarray(clahe_image_np)

    if use_eh:
        eh_image_np = apply_equalize_hist(image_np)
        eh_image = Image.fromarray(eh_image_np)

    original_dir = os.path.join(save_dir, 'Original')
    clahe_dir = os.path.join(save_dir, 'CLAHE') if use_clahe else None
    eh_dir = os.path.join(save_dir, 'EqualizeHist') if use_eh else None

    os.makedirs(original_dir, exist_ok=True)
    if use_clahe:
        os.makedirs(clahe_dir, exist_ok=True)
    if use_eh:
        os.makedirs(eh_dir, exist_ok=True)

    filename = os.path.basename(image_path)

    image.save(os.path.join(original_dir, filename))

    if use_clahe:
        clahe_image.save(os.path.join(clahe_dir, filename))

    if use_eh:
        eh_image.save(os.path.join(eh_dir, filename))

    fig, axes = plt.subplots(1, 2 + int(use_clahe) + int(use_eh), figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    if use_clahe:
        axes[1].imshow(clahe_image)
        axes[1].set_title("CLAHE Applied Image")
        axes[1].axis("off")

    if use_eh:
        axes[2].imshow(eh_image)
        axes[2].set_title("EqualizeHist Applied Image")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"comparison_{filename}.png"))
    plt.close()

    print(f"Processed and saved images for: {image_path}")

def process_images_in_folder(image_folder, save_dir, use_clahe=False, use_eh=False):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                visualize_and_save(image_path, save_dir, use_clahe=use_clahe, use_eh=use_eh)

if __name__ == "__main__":
    image_folder = "/data/ephemeral/home/data/train/DCM"  
    save_dir = "visualize_output"  

    use_clahe = True 
    use_eh = True     

    process_images_in_folder(image_folder, save_dir, use_clahe=use_clahe, use_eh=use_eh)