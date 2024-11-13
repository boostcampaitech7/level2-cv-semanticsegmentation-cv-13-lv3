import os

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt

from train import train
from test import test

from xraydataset import XRayDataset, split_data
from utils import label2rgb, set_seed, get_sorted_files_by_type

from constants import CLASSES, TRAIN_DATA_DIR, TEST_DATA_DIR

from argparse import ArgumentParser
import numpy as np
import pandas as pd

import albumentations as A

import segmentation_models_pytorch as smp

# model 불러오기
# 출력 label 수 정의 (classes=29)


def do_train(args):
    # model = models.segmentation.fcn_resnet50(pretrained=True)
    # # output class 개수를 dataset에 맞도록 수정합니다.
    # model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

    model = smp.Unet(
        encoder_name="resnet50", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=len(CLASSES),                     # model output channels (number of classes in your dataset)
    )

    image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
    label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')

    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')

    train_files, valid_files = split_data(pngs, jsons)

    train_dataset = XRayDataset(image_files=train_files['filenames'], label_files=train_files['labelnames'], transforms=A.Resize(args.input_size, args.input_size))
    valid_dataset = XRayDataset(image_files=valid_files['filenames'], label_files=valid_files['labelnames'], transforms=A.Resize(args.input_size, args.input_size))

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # Loss function을 정의합니다.
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer를 정의합니다.
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)

    train(model, train_loader, valid_loader, criterion, optimizer, args.max_epoch, args.valid_interval, args.checkpoint_dir)


def do_test(args):
    model = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_file))

    image_root = os.path.join(TEST_DATA_DIR, 'DCM')
    pngs = get_sorted_files_by_type(image_root, 'png')
    
    test_dataset = XRayDataset(image_files=np.array(pngs), transforms=A.Resize(args.input_size, args.input_size))

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    rles, filename_and_class = test(model, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.to_csv("output.csv", index=False)


def main(args):
    # 시드를 설정합니다.
    set_seed(args.seed)

    if args.train:
        do_train(args)
        
    elif args.test:
        do_test(args)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--checkpoint_dir', type=str,default="./checkpoints")
    parser.add_argument('--checkpoint_file', type=str,default="fcn_resnet50_best_model.pt")
    
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument("--train", action="store_true", help="train")
    parser.add_argument("--test", action="store_true", help="test")

    parser.add_argument("--input_size", type=int, default=512)

    # parser.add_argument("--amp", action="store_true", help="mixed precision")
 
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--valid_interval', type=int, default=5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)