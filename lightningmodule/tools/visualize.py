import sys

sys.path.append('../')

# torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xraydataset import XRayDataset, split_data
from utils import get_sorted_files_by_type, label2rgb, decode_rle_to_mask

from constants import TRAIN_DATA_DIR, CLASSES, PALETTE

from argparse import ArgumentParser

import albumentations as A

import os

import numpy as np

import shutil

from PIL import Image, ImageDraw
import cv2

import wandb

import pandas as pd

import datetime

from augmentation import load_transforms
from omegaconf import OmegaConf
from argparse import ArgumentParser

from augmentation import CopyPasteDataset

def create_pred_mask_dict(csv_path, input_size):
    df = pd.read_csv(csv_path)

    mask_dict = dict()

    # 그룹화하여 처리
    grouped = df.groupby('image_name')
    for image_name, group in grouped:
        print(f'creating mask for {image_name}...')
        masks = dict()
        for _, row in group.iterrows():
            classname = row['class']
            rle = row['rle']
            if isinstance(rle, str):
                mask = decode_rle_to_mask(rle, 2048, 2048)
                mask_resized = np.array(Image.fromarray(mask).resize((input_size, input_size)))
                masks[classname]=mask_resized

        mask_dict[image_name] = masks
    print('mask creation from csv is done')
    return mask_dict

def ready_for_visualize(image, label):
    lbl = label.numpy().astype(np.uint8)
    img = image.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)  # [C, H, W] -> [H, W, C]로 변환   

    return img, lbl

# 모델 학습과 검증을 수행하는 함수
def visualize_compare(args, visual_loader, mask_dict):

    csv_compare = False if mask_dict is None else True

    time = datetime.datetime.now().strftime('%m-%d_%H:%M')
    run_name = args.name + '_' + time
    project_name = 'visualize'
    wandb.init(project=project_name, name=run_name)
    
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

    class_labels = []
    # if csv_compare:
    class_labels_cmp = [{1:"GT", 2:"Pred", 3:"Overlap"} for _ in range(len(class_groups))] 
    # else:
    class_labels = [{} for _ in range(len(class_groups))]
    for idx, class_group in enumerate(class_groups):
        for class_idx in class_group:
            class_labels[idx][class_idx]=CLASSES[class_idx-1]      

    # Create the class label dictionary to map IDs to group names
    for image_names, images, labels in visual_loader:
            for image_name, image, label in zip(image_names, images, labels):

                print(f"Uploading {image_name}...")

                img, gt = ready_for_visualize(image, label)

                pred = None
                if csv_compare:
                    pred = mask_dict.get(image_name, None)

                # Initialize the mask array
                if args.gt:
                    combined_mask_gt = np.zeros(shape=(len(class_groups), args.input_size, args.input_size), dtype=np.uint8)  # Shape: (H, W)
                combined_mask_pred = np.zeros(shape=(len(class_groups), args.input_size, args.input_size), dtype=np.uint8)  # Shape: (H, W)
                combined_mask_cmp = np.zeros(shape=(len(class_groups), args.input_size, args.input_size), dtype=np.uint8)  # Shape: (H, W)
                # Assign unique IDs to each group
                for group_id, group_classes in enumerate(class_groups):
                    for class_index in group_classes:

                        gt_mask = gt[class_index-1]
                        if pred is not None:                        
                            pred_mask = pred.get(CLASSES[class_index-1], None)
                            # 세 가지 경우를 모두 고려하여 combined_mask 값 설정
                            combined_mask_cmp[group_id][(gt_mask == 1) & (pred_mask == 0)] = 1  # gt만 있는 영역
                            combined_mask_cmp[group_id][(gt_mask == 0) & (pred_mask == 1)] = 2  # pred만 있는 영역
                            combined_mask_cmp[group_id][(gt_mask == 1) & (pred_mask == 1)] = 3  # 겹치는 영역

                            combined_mask_pred[group_id][pred_mask == 1] = class_index  # lbl is 0-indexed, so subtract 1
                        if args.gt:
                            combined_mask_gt[group_id][gt_mask == 1] = class_index  # lbl is 0-indexed, so subtract 1
                if args.gt:   
                    masks_gt = dict()
                    for i, mask in enumerate(combined_mask_gt):
                        masks_gt[class_group_label[i]] = dict(mask_data=mask,class_labels=class_labels[i])
                    masked_image_gt = wandb.Image(img, masks=masks_gt, caption=image_name)      
                    wandb.log({"GT Mask":masked_image_gt})

                masks_pred = dict()
                for i, mask in enumerate(combined_mask_pred):
                    masks_pred[class_group_label[i]] = dict(mask_data=mask,class_labels=class_labels[i])
                masked_image_pred = wandb.Image(img, masks=masks_pred, caption=image_name)  
                wandb.log({"Pred Mask":masked_image_pred})    

                masks_cmp = dict()
                for i, mask in enumerate(combined_mask_cmp):
                    masks_cmp[class_group_label[i]] = dict(mask_data=mask,class_labels=class_labels_cmp[i])
                masked_image_cmp = wandb.Image(img, masks=masks_cmp, caption=image_name)      
                wandb.log({"Mask Compare":masked_image_cmp})

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

    return image

def visualize_dataset(visual_loader, augmentation=False):
    save_dir = 'visualize/'

    if os.path.exists(save_dir):  
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)    

    for image_names, images, labels in visual_loader:
        for image_name, image, label in zip(image_names, images, labels):
            
            lbl = label.numpy().astype(np.uint8)
            img = image.permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            
            for i, class_label in enumerate(lbl):
                img[class_label == 1] = PALETTE[i]
                
            img = Image.fromarray(img)  # [C, H, W] -> [H, W, C]로 변환   
            img.save(os.path.join(save_dir, image_name))

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--gt", action="store_true", help="upload gt")
    parser.add_argument("--local", action="store_true", help="save aug into local")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--name", type=str, default="compare_mask")
    parser.add_argument("--augmentation", action="store_true", help="augmentation visualize")
    parser.add_argument("--config", type=str, default="../configs/base_config.yaml")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
    label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')

    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')

    image_files, label_files = None, None

    if not args.gt:
        _, valid_files = split_data(pngs, jsons)
        image_files, label_files = valid_files['filenames'], valid_files['labelnames']
    else: # 모든 train에 대한 업로드를 진행
        image_files, label_files = np.array(pngs), jsons

    transforms = A.Resize(args.input_size, args.input_size)
    if args.augmentation:
        print("Augmentation Visualize")
        with open(args.config, 'r') as f:
            cfg = OmegaConf.load(f)
        transforms = load_transforms(cfg)

    visual_dataset = XRayDataset(
        image_files=image_files, 
        label_files=label_files, 
        transforms=transforms, 
        use_cp=True, 
        cp_args=['Trapezium', 'Capitate', 'Lunate', 'Scaphoid', 'finger-1', 'finger-16', 'Pisiform', 'Hamate', 'Triquetrum'])
                               
    visual_loader = DataLoader(
        dataset=visual_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    if args.local:
        visualize_dataset(visual_loader, args.augmentation)
    else:
        mask_dict = None
        if args.csv is not None:
            mask_dict = create_pred_mask_dict(args.csv, args.input_size)

        visualize_compare(args, visual_loader, mask_dict)

if __name__ == '__main__':
    main()