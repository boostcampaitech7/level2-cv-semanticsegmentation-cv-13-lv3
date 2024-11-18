import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from data_utils import convert_coco_to_yolo, split_dataset
from ultralytics import YOLO
from utils.Gsheet import Gsheet_param
from argparse import ArgumentParser
from omegaconf import OmegaConf

# Define paths
original_image_dir = "/data/ephemeral/home/data/train"  
train_split_dir = "/data/ephemeral/home/data/train_split" 
val_split_dir = "/data/ephemeral/home/data/val_split"     
train_json_path = "/data/ephemeral/home/data/train.json" 
data_yaml_path = "./cfg/data.yaml"  
model_path = "yolo11x-seg.pt"  

valid_classes = list(range(30))

print("Converting COCO data to YOLO format...")
convert_yolo(train_json_path, original_image_dir, valid_classes)
print("Conversion to YOLO format completed.")

print("Splitting dataset into train and val...")
split_dataset(original_image_dir, train_split_dir, val_split_dir)
print("Dataset split completed.")

wandb.init(project="Ultralytics")

model = YOLO(model_path)

add_wandb_callback(model)

results = model.train(
    data=data_yaml_path,  
    epochs=100,
    imgsz=512,
    batch=2,
    amp=True, 
)

def filter_wandb_predictions(predictions, valid_classes):
    filtered_predictions = {}
    for image_id, pred_string in predictions.items():
        filtered_pred = []
        pred_list = pred_string.strip().split(" ")
        for i in range(0, len(pred_list), 6):
            values = pred_list[i:i+6]
            if len(values) != 6:
                continue
            cls = int(values[0])
            if cls in valid_classes:
                filtered_pred.append(" ".join(values))
        filtered_predictions[image_id] = " ".join(filtered_pred)
    return filtered_predictions

def filter_ground_truth(ground_truth, valid_classes):
    filtered_ground_truth = {}
    for image_id, gt_string in ground_truth.items():
        filtered_gt = []
        gt_list = gt_string.strip().split(" ")
        for i in range(0, len(gt_list), 6):
            values = gt_list[i:i+6]
            if len(values) != 6:
                continue
            cls = int(values[0])
            if cls in valid_classes:
                filtered_gt.append(" ".join(values))
        filtered_ground_truth[image_id] = " ".join(filtered_gt)
    return filtered_ground_truth

if hasattr(results, 'pred'):
    filtered_predictions = filter_wandb_predictions(results.pred, valid_classes)
    wandb.log({"filtered_predictions": filtered_predictions})

if hasattr(results, 'labels'):
    filtered_ground_truth = filter_ground_truth(results.labels, valid_classes)
    wandb.log({"filtered_ground_truth": filtered_ground_truth})

wandb.finish()

print("Training completed.")

def train_model(cfg):
    # Training code here
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    train_model(cfg)
    Gsheet_param(cfg)