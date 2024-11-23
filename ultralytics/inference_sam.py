from ultralytics import YOLO, SAM
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from glob import glob

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def inference_yolo():
    model = YOLO("/data/ephemeral/home/bohyun/ultralytics/best.pt").cuda()
    infer_images = sorted(glob("/data/ephemeral/home/data/test/*/*/*.png"))
    
    results_yolo = []
    for infer_image in tqdm(infer_images):
        result = model.predict(infer_image, imgsz=2048)[0]
        results_yolo.append((infer_image, result))
    return results_yolo

def inference_sam2():
    model = SAM("sam2.1_b.pt").cuda()
    infer_images = sorted(glob("/data/ephemeral/home/data/test/*/*/*.png"))
    
    results_sam2 = []
    for infer_image in tqdm(infer_images):
        result = model(infer_image)
        results_sam2.append((infer_image, result))
    return results_sam2

def process_results(results, method="YOLO"):
    rles = []
    filename_and_class = []
    
    for infer_image, result in tqdm(results):
        img_name = os.path.basename(infer_image)
        
        if method == "YOLO":
            boxes = result.boxes.data.cpu().numpy()
            classes = boxes[:, 5].astype(np.uint8).tolist()
            masks = result.masks.xy
            
            for c, mask_pts in zip(classes, masks):
                empty_mask = np.zeros((2048, 2048), dtype=np.uint8)
                pts = [[int(x[0]), int(x[1])] for x in mask_pts]
                cv2.fillPoly(empty_mask, [np.array(pts)], 1)
                rle = encode_mask_to_rle(empty_mask)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}_{img_name}")
        elif method == "SAM":
            masks = result.masks  # Assuming SAM outputs masks directly
            for idx, mask in enumerate(masks):
                rle = encode_mask_to_rle(mask)
                rles.append(rle)
                filename_and_class.append(f"SAM_{img_name}_{idx}")
    
    return filename_and_class, rles

def save_results(filename_and_class, rles, method):
    classes, filenames = zip(*[x.split("_") for x in filename_and_class])
    df = pd.DataFrame({
        "image_name": filenames,
        "class": classes,
        "rle": rles,
    })
    
    output_dir = './result'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{method}_output.csv"), index=False)

if __name__ == "__main__":
    print("Running YOLO Inference...")
    yolo_results = inference_yolo()
    filename_and_class_yolo, rles_yolo = process_results(yolo_results, method="YOLO")
    save_results(filename_and_class_yolo, rles_yolo, method="YOLO")
    
    print("Running SAM 2 Inference...")
    sam2_results = inference_sam2()
    filename_and_class_sam2, rles_sam2 = process_results(sam2_results, method="SAM")
    save_results(filename_and_class_sam2, rles_sam2, method="SAM")