import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm.auto import tqdm
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import wandb

from dataset import XRayDataset
from utils import *

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

# Function to visualize CAM with the SnapMix image
def visualize_snapmix_with_cam(image_a, image_b, cam_a, cam_b, target_class=0):
    # Resize the CAMs to match the input image size
    cam_a_resized = cv2.resize(cam_a, (image_a.shape[2], image_a.shape[1]))
    cam_b_resized = cv2.resize(cam_b, (image_b.shape[2], image_b.shape[1]))

    # Normalize the CAM for better visualization (0-255 range)
    cam_a_resized = np.maximum(cam_a_resized, 0)
    cam_b_resized = np.maximum(cam_b_resized, 0)
    cam_a_resized = (cam_a_resized / cam_a_resized.max() * 255).astype(np.uint8)
    cam_b_resized = (cam_b_resized / cam_b_resized.max() * 255).astype(np.uint8)

    # Convert to heatmaps
    heatmap_a = cv2.applyColorMap(cam_a_resized, cv2.COLORMAP_JET)
    heatmap_b = cv2.applyColorMap(cam_b_resized, cv2.COLORMAP_JET)

    # Blend the original images with the CAM heatmaps (overlay)
    blended_a = cv2.addWeighted(image_a.permute(1, 2, 0).cpu().numpy(), 0.7, heatmap_a, 0.3, 0)
    blended_b = cv2.addWeighted(image_b.permute(1, 2, 0).cpu().numpy(), 0.7, heatmap_b, 0.3, 0)

    return blended_a, blended_b

# Update the SnapMix function to include CAM visualization
def save_eda_images(dataset, save_dir, model, num_examples=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(num_examples):
        image, mask = dataset[i]
        next_image, next_mask = dataset[(i + 1) % len(dataset)]

        # Generate CAMs for both images
        cam_a = generate_cam(model, image)
        cam_b = generate_cam(model, next_image)

        # Visualize the SnapMix image with CAMs
        snapmix_image, _ = apply_cam_based_snapmix(image, mask, next_image, next_mask, model)

        # Visualize SnapMix with CAMs overlayed
        blended_a, blended_b = visualize_snapmix_with_cam(image, next_image, cam_a, cam_b)

        # Log the images with WandB
        wandb.log({
            f"Original Image {i}": wandb.Image(image),
            f"SnapMix Image {i}": wandb.Image(snapmix_image),
            f"Blended Image A (with CAM) {i}": wandb.Image(blended_a),
            f"Blended Image B (with CAM) {i}": wandb.Image(blended_b),
        })
def save_model(model, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main(seed, epochs, lr, batch_size, valid_batch_size, valid_interval, valid_thr,
         save_dir, save_name, encoder_name, encoder_weights, data_yaml_path, clahe, cp):

    # Debugging: Check if wandb is initialized correctly
    print("Initializing WandB...")
    wandb.init(project="Semseg", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "data_yaml": data_yaml_path,
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights
    })
    print("WandB Initialized.")

    try:
        image_root = "/data/ephemeral/home/data/train/DCM"
        label_root = "/data/ephemeral/home/data/train/outputs_json"
        set_seed(seed)

        save_dir_root = '/data/ephemeral/home'
        if not os.path.isdir(save_dir_root):
            os.mkdir(save_dir_root)

        save_dir = os.path.join(save_dir_root, save_dir)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        print("Creating Model...")
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=len(CLASSES))
        print("Model Created.")

        train_tf = A.Resize(1024, 1024)
        valid_tf = A.Resize(1024, 1024)

        # Create Datasets
        print("Creating Datasets...")
        train_dataset = XRayDataset(image_root=image_root, label_root=label_root, is_train=True, transforms=train_tf, clahe=clahe, copypaste=cp)
        valid_dataset = XRayDataset(image_root=image_root, label_root=label_root, is_train=False, transforms=valid_tf, clahe=clahe, copypaste=False)
        print("Datasets Created.")

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=1, drop_last=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

        # Initialize best_dice before comparison
        best_dice = 0.  # or use float('-inf') if you prefer

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            model.train()
            total_train_loss = 0
            for step, (images, masks) in enumerate(train_loader):
                images, masks = images.cuda(), masks.cuda()
                model = model.cuda()
                outputs = model(images)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss
                if (step + 1) % 25 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], Loss: {round(loss.item(),4)}')

            scheduler.step()

            # Validation and logging to wandb
            if (epoch + 1) % valid_interval == 0:
                model.eval()
                dices = []
                with torch.no_grad():
                    total_valid_loss = 0
                    for step, (images, masks) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                        images, masks = images.cuda(), masks.cuda()
                        outputs = model(images)
                        if outputs.size(-2) != masks.size(-2) or outputs.size(-1) != masks.size(-1):
                            outputs = F.interpolate(outputs, size=masks.size()[-2:], mode="bilinear")
                        loss = criterion(outputs, masks)
                        total_valid_loss += loss
                        outputs = torch.sigmoid(outputs)
                        outputs = (outputs > valid_thr).detach().cpu()
                        masks = masks.detach().cpu()
                        dice = dice_coef(outputs, masks)
                        dices.append(dice)

                dices = torch.cat(dices, 0)
                dices_per_class = torch.mean(dices, 0)
                avg_dice = torch.mean(dices_per_class).item()

                print(f"Validation Loss: {total_valid_loss.item()}, Average Dice: {avg_dice}")
                wandb.log({"Validation Loss": total_valid_loss.item(), "Average Dice": avg_dice})
                for idx, class_name in enumerate(CLASSES):
                    wandb.log({f"{class_name} Dice": dices_per_class[idx].item()})

                if best_dice < avg_dice:
                    print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {avg_dice:.4f}")
                    best_dice = avg_dice
                    save_model(model, save_dir=save_dir, file_name=save_name)

        print("Training Complete.")
        wandb.finish()

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=137, help='random seed (default: 21)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training (default: 8->2)')
    parser.add_argument('--valid_batch_size', type=int, default=2, help='input batch size for validating (default: 2)')
    parser.add_argument('--valid_interval', type=int, default=2, help='validation after {valid_interval} epochs (default: 10)')
    parser.add_argument('--valid_thr', type=float, default=.5, help='validation threshold (default: 0.5)')
    parser.add_argument('--save_dir', type=str, default="exp", help="model save directory")
    parser.add_argument('--save_name', type=str, default="best.pt", help="model save name")
    parser.add_argument('--encoder_name', type=str, default='tu-xception71', help="encoder name")
    parser.add_argument('--encoder_weights', type=str, default="imagenet", help="pre-trained weights for encoder initialization")
    parser.add_argument('--data_yaml_path', type=str, required=True, help="Path to data.yaml file for wandb logging")
    parser.add_argument('--clahe', type=bool, default=False, help='clahe augmentation')
    parser.add_argument('--cp', type=bool, default=False, help='copypaste augmentation')
    args = parser.parse_args()
    main(**args.__dict__)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.model_selection import GroupKFold
import numpy as np
import os
import cv2
import json

from PIL import Image, ImageDraw
import random

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

def do_clahe(image) :
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)) 
    image[:,:,0] = clahe.apply(image[:,:,0])
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    return image

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, is_train=True, transforms=None, clahe=False, copypaste=False, k=3):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

        jsons = {
            os.path.relpath(os.path.join(root, fname), start=label_root)
            for root, _dirs, files in os.walk(label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }

        print(f"Found {len(pngs)} .png files in {image_root}")
        print(f"Found {len(jsons)} .json files in {label_root}")

        if len(pngs) == 0 or len(jsons) == 0:
            raise ValueError("No .png or .json files found. Check if paths are correct and files are in the specified directories.")

        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

        assert len(jsons_fn_prefix) - len(pngs_fn_prefix) == 0, "Mismatch in .json and .png file counts"
        assert len(pngs_fn_prefix) - len(jsons_fn_prefix) == 0, "Mismatch in .png and .json file counts"

        pngs = sorted(pngs)
        jsons = sorted(jsons)

        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)


        groups = [os.path.dirname(fname) for fname in _filenames]

        ys = [0 for fname in _filenames]

        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        dataset_no = 2
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                if i == dataset_no:
                    continue

                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            else:
                if i != dataset_no :
                    continue
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                break

        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.image_root = image_root
        self.label_root = label_root 

        self.clahe = clahe
        self.copypaste = copypaste
        self.k = k

    def __len__(self):
        return len(self.filenames)

    def get_coord(self, polygon):
        polygon = polygon
        for i in range(len(polygon)):
            polygon[i] = tuple(polygon[i])
        polygon_np = np.array(polygon)

        max = np.max(polygon_np, axis=0)
        min = np.min(polygon_np, axis=0)

        return max, min

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)

        if self.clahe :
            image = do_clahe(image)

        image = image / 255.

        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)

        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.copypaste:
            randoms = random.choices([i for i in range(640)], k=self.k)
            for i in randoms:
                target_image = cv2.imread(os.path.join(self.image_root, self.filenames[i])) / 255.
                target_label_path = os.path.join(self.label_root, self.labelnames[i])

                with open(target_label_path, "r") as f:
                    target_annotations = json.load(f)
                target_annotations = target_annotations["annotations"]

                for ann in target_annotations:
                    target_c = ann["label"]
                    target_c = CLASS2IND[target_c]
                    if target_c == 19 or target_c == 20 or target_c == 25 or target_c == 26:
                        points = np.array(ann["points"])
                        max, min = self.get_coord(ann['points'])
                        x = random.randint(100,1800)
                        y = random.randint(100,1800)
                        alpha = random.randint(25,50)

                        bone_area_x = [i for i in range(400,1600)]
                        while x in bone_area_x:
                            x = random.randint(100,1800)
                        x -= alpha

                        img = Image.new('L', target_image.shape[:2], 0)
                        ImageDraw.Draw(img).polygon(ann['points'], outline=0, fill=1)
                        mask = np.array(img)

                        new_image = cv2.bitwise_or(target_image, target_image, mask=mask)
                        image[y:y+max[1]-min[1], x:x+max[0]-min[0], ...] = new_image[min[1]:max[1], min[0]:max[0], ...]

                        ori_label = label[..., target_c]
                        ori_label[y:y+max[1]-min[1], x:x+max[0]-min[0]] = mask[min[1]:max[1], min[0]:max[0]]
                        label[..., target_c] = ori_label

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

class XRayInferenceDataset(Dataset):
    def __init__(self, image_root, transforms=None, clahe=False):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms
        self.image_root = image_root
        self.clahe = clahe

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)

        if self.clahe :
            image = do_clahe(image)

        image = image / 255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        image = image.transpose(2, 0, 1)  

        image = torch.from_numpy(image).float()

        return image, image_name