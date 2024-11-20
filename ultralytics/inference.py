from ultralytics import YOLO
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from argparse import ArgumentParser

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_all_image_paths(root_dir, file_extension='png'):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(f".{file_extension}"):
                image_paths.append(os.path.join(dirpath, file))
    return sorted(image_paths)

def test_model(args):
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}.pt")
    model = YOLO(checkpoint_path)

    image_root = os.path.abspath(args.image_dir)
    image_paths = get_all_image_paths(image_root)

    results = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        # YOLO Inference
        pred_results = model.predict(source=image_path, save=False, conf=0.5)
        if pred_results[0].masks is None or len(pred_results[0].masks.data) == 0:
            print(f"No valid masks detected for image: {image_path}")
            continue

        masks = pred_results[0].masks.data.cpu().numpy()

        # Ensure all 29 classes are included
        for cls_idx in range(len(CLASSES)):
            if cls_idx < len(masks):
                mask = masks[cls_idx]
                resized_mask = cv2.resize(mask, (2048, 2048), interpolation=cv2.INTER_NEAREST)
                binary_mask = (resized_mask > 0.5).astype(np.uint8)
                rle = encode_mask_to_rle(binary_mask)
            else:
                # If mask does not exist for this class, use an empty RLE
                rle = "0"

            # Append result
            results.append({
                "image_name": os.path.basename(image_path),
                "class": CLASSES[cls_idx],
                "rle": rle
            })

    # Save results to a CSV file with correct header
    output_csv = os.path.join(args.output_dir, "output.csv")
    df = pd.DataFrame(results, columns=["image_name", "class", "rle"])  # Ensure correct column order
    df.to_csv(output_csv, index=False, header=True)  # Include headers explicitly
    print(f"Results saved to {output_csv}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory path")
    parser.add_argument("--checkpoint_file", type=str, default="best", help="Checkpoint file name without extension")
    parser.add_argument("--image_dir", type=str, default="/data/ephemeral/home/data/test", help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output CSV")
    args = parser.parse_args()

    test_model(args)