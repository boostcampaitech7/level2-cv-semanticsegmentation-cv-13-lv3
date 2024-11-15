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