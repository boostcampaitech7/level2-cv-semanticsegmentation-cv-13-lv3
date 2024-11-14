# torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xraydataset import XRayDataset
from utils import get_sorted_files_by_type, label2rgb

from constants import TRAIN_DATA_DIR, CLASSES, PALETTE

from argparse import ArgumentParser

import albumentations as A

import os

import numpy as np

import matplotlib.pyplot as plt
import shutil

from PIL import Image, ImageDraw
import cv2

import wandb

def ready_for_visualize(image, label):
    lbl = label.numpy().astype(np.uint8)
    img = image.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)  # [C, H, W] -> [H, W, C]로 변환   

    return img, lbl

def draw_outline(image, label, is_binary = False):

    draw = ImageDraw.Draw(image)

    for i, class_label in enumerate(label):
        if class_label.max() > 0:  # Only process if the class is present in the image
            color = PALETTE[i] if not is_binary else 1

            # Find the points for the outline
            contours, _ = cv2.findContours(class_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw each contour as a polygon
            for contour in contours:
                pts = [(int(point[0][0]), int(point[0][1])) for point in contour]
                draw.polygon(pts, outline=color)

    return image, label
    

# 모델 학습과 검증을 수행하는 함수
def visual_dataset_wandb(visual_loader):

    wandb.init()
    
    # Define your class groups
    class_groups = [
        [1, 4, 8, 12, 16, 20, 22, 26],  # finger-1, finger-4, finger-8, etc.
        [2, 5, 9, 13, 17, 23, 24, 29],  # finger-2, finger-5, finger-9, etc.
        [3, 6, 10, 14, 18, 21, 27, 28],  # finger-11, finger-15
        [11, 19, 25],  # finger-19
        [7, 15]
    ]

    class_group_label = [
        'Trapezium, Capitate, Triquetrum',
        'Hamate, Scaphoid, Ulna',
        'Trapezoid, Pisiform, Radius',
        '11, 19, Lunate',
        '7, 15'
    ]

    class_labels = [{} for _ in range(len(class_groups))]
    for idx, class_group in enumerate(class_groups):
        for class_idx in class_group:
           class_labels[idx][class_idx]=CLASSES[class_idx-1]

    table = wandb.Table(columns=["Image"])

    # Create the class label dictionary to map IDs to group names
    for image_names, images, labels in visual_loader:
            for image_name, image, label in zip(image_names, images, labels):
                img, lbl = ready_for_visualize(image, label)

                # outline_lbl = Image.fromarray(np.zeros(label.shape[1:], dtype=np.uint8))
                # lbl = draw_outline(outline_lbl, lbl, is_binary=True)*255

                # Initialize the mask array
                combined_mask = np.zeros(shape=(len(class_groups), args.input_size, args.input_size), dtype=np.uint8)  # Shape: (H, W)
                # Assign unique IDs to each group
                for group_id, group_classes in enumerate(class_groups):
                    for class_index in group_classes:
                        # Set the pixels in the mask for the current class group
                        combined_mask[group_id][lbl[class_index - 1] == 1] = class_index  # lbl is 0-indexed, so subtract 1

                masks = dict()
                for i, mask in enumerate(combined_mask):
                    masks[class_group_label[i]] = dict(mask_data=mask,class_labels=class_labels[i])
                masked_image = wandb.Image(img, masks=masks, caption=image_name)             
                table.add_data(masked_image)
                wandb.log({"image with GT mask":masked_image})
    wandb.log({"random_field": table})
    return

def visual_dataset(visual_loader):
    save_dir = 'visualize/'

    if os.path.exists(save_dir):  
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)    

    for image_names, images, labels in visual_loader:
        for image_name, image, label in zip(image_names, images, labels):
            img, lbl = ready_for_visualize(image, label)

            img = draw_outline(img, lbl)

            img.save(os.path.join(save_dir, image_name))

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--wandb", action="store_true", help="upload image into wandb")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
    label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')

    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')

    visualize_dataset = XRayDataset(image_files=np.array(pngs), label_files=jsons, transforms=A.Resize(args.input_size, args.input_size))

    visual_loader = DataLoader(
        dataset=visualize_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    if args.wandb:
        visual_dataset_wandb(visual_loader)
    else:
        visual_dataset(visual_loader)