import os
from glob import glob
import albumentations as A
from ultralytics import YOLO
from torch.utils.data import DataLoader
from xraydataset import XRayDataset
from lightning.pytorch.loggers import WandbLogger
from utils.Gsheet import Gsheet_param


def get_sorted_files_by_type(root_dir, file_extension):
    return sorted(glob(os.path.join(root_dir, "**", f"*.{file_extension}"), recursive=True))


def split_data(image_files, label_files, split_ratio=0.8):
    num_train = int(len(image_files) * split_ratio)
    train_data = {
        'filenames': image_files[:num_train],
        'labelnames': label_files[:num_train]
    }
    valid_data = {
        'filenames': image_files[num_train:],
        'labelnames': label_files[num_train:]
    }
    return train_data, valid_data


if __name__ == "__main__":
    DATA_ROOT = "/data/ephemeral/home/data"
    train_image_root = os.path.join(DATA_ROOT, "yolo_train/images")
    train_label_root = os.path.join(DATA_ROOT, "yolo_train/labels")

    pngs = get_sorted_files_by_type(train_image_root, "png")
    txts = get_sorted_files_by_type(train_label_root, "txt")

    train_files, valid_files = split_data(pngs, txts, split_ratio=0.8)

    train_dataset = XRayDataset(
        image_files=train_files['filenames'],
        label_files=train_files['labelnames'],
        transforms=A.Resize(640, 640)
    )

    valid_dataset = XRayDataset(
        image_files=valid_files['filenames'],
        label_files=valid_files['labelnames'],
        transforms=A.Resize(640, 640)
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4)

    wandb_logger = WandbLogger(project="YOLO_Segmentation", name="T6030_yolo_seg_640")

    model = YOLO("yolov8x-seg.pt")
    model.train(
        data="cfg/data.yaml",
        epochs=50,
        imgsz=2048,
        batch=2,
        optimizer="AdamW",
        device=0,
        workers=4,
        project="Segmentation",
        name="T6030_yolo_seg_640",
    )

    gsheet_config = {
        "data_root": DATA_ROOT,
        "epochs": 50,
        "image_size": 2048,
        "batch_size": 2,
        "optimizer": "AdamW",
        "model": "yolov8x-seg.pt",
        "project_name": "YOLO_Segmentation",
        "run_name": "T6030_yolo_seg_640"
    }
    Gsheet_param(gsheet_config)

    print("Training complete and parameters logged to Google Sheets.")