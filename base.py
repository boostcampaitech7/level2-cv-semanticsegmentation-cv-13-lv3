import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import datetime
from tqdm.auto import tqdm
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
import os

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

def save_model(model, save_dir, file_name='best_model.pt'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, file_name))

def save_eda_images(dataset, save_dir, num_examples=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(num_examples):
        image, mask = dataset[i] 

        image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        mask_np = mask.cpu().numpy()
        mask_np = np.argmax(mask_np, axis=0).astype(np.uint8)  

        cv2.imwrite(os.path.join(save_dir, f"image_{i}.png"), image_np)
        cv2.imwrite(os.path.join(save_dir, f"mask_{i}.png"), mask_np)

def main(seed, epochs, lr, batch_size, valid_batch_size, valid_interval, valid_thr,
         save_dir, save_name, encoder_name, encoder_weights, clahe, cp):
    
    image_root = "/data/ephemeral/home/train/DCM"  # Update to correct image directory
    label_root = "/data/ephemeral/home/train/outputs_json"  # Update to correct label directory

    set_seed(seed)
    
    save_dir_root = '/data/ephemeral/home'
    if not os.path.isdir(save_dir_root):
        os.mkdir(save_dir_root)
        
    save_dir = os.path.join(save_dir_root, save_dir)
    if not os.path.isdir(save_dir):                                                           
        os.mkdir(save_dir)

    ### MODEL SETTING ###
    model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=len(CLASSES)
            )

    train_tf = A.Resize(1024, 1024)
    valid_tf = A.Resize(1024, 1024)
    
    # Initialize datasets with image and label root directories
    train_dataset = XRayDataset(image_root=image_root, label_root=label_root, is_train=True, transforms=train_tf, clahe=clahe, copypaste=cp)
    valid_dataset = XRayDataset(image_root=image_root, label_root=label_root, is_train=False, transforms=valid_tf, clahe=clahe, copypaste=False)

    # EDA - Save sample augmented images
    eda_dir = '/data/ephemeral/home/eda'
    save_eda_images(train_dataset, eda_dir)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    
    # Loss function 정의
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer 정의
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6)
    
    # Scheduler 정의
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)  # T_max는 최대 에포크 수, eta_min은 최소 학습률
    
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for step, (images, masks) in enumerate(train_loader):            
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            # inference
            outputs = model(images)
            
            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{epochs}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                
        scheduler.step()        
        
        total_train_loss = total_train_loss.item()
        mean_train_loss = total_train_loss/len(train_loader)
          
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % valid_interval == 0:
            print(f'Start validation #{epoch:2d}')
            set_seed(seed)
            model.eval()

            dices = []
            with torch.no_grad():
                n_class = len(CLASSES)
                total_valid_loss = 0
                cnt = 0

                for step, (images, masks) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    images, masks = images.cuda(), masks.cuda()         
                    model = model.cuda()
                    
                    outputs = model(images)
                    if outputs.size(-2) != masks.size(-2) or outputs.size(-1) != masks.size(-1):
                        outputs = F.interpolate(outputs, size=masks.size()[-2:], mode="bilinear")
                    
                    loss = criterion(outputs, masks)
                    total_valid_loss += loss
                    cnt += 1
                    
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > valid_thr).detach().cpu()
                    masks = masks.detach().cpu()
                    
                    dice = dice_coef(outputs, masks)
                    dices.append(dice)
                    
            total_valid_loss = total_valid_loss.item()
            mean_valid_loss = total_valid_loss / len(valid_loader)
                        
            dices = torch.cat(dices, 0)
            dices_per_class = torch.mean(dices, 0)
            dice_str = [
                f"{c:<12}: {d.item():.4f}"
                for c, d in zip(CLASSES, dices_per_class)
            ]
            dice_str = "\n".join(dice_str)
            print(dice_str)
            
            avg_dice = torch.mean(dices_per_class).item()
            
            if best_dice < avg_dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {avg_dice:.4f}")
                print(f"Save model in {save_dir}")
                best_dice = avg_dice
                save_model(model, save_dir=save_dir, file_name=save_name)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=137, help='random seed (default: 21)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training (default: 8->2)')
    parser.add_argument('--valid_batch_size', type=int, default=2, help='input batch size for validating (default: 2)')
    parser.add_argument('--valid_interval', type=int, default=2, help='validation after {valid_interval} epochs (default: 10)')
    parser.add_argument('--valid_thr', type=float, default=.5, help='validation threshold (default: 0.5)')
   
    # Removed image_root and label_root from arguments
    parser.add_argument('--save_dir', type=str, default="exp", help="model save directory")
    parser.add_argument('--save_name', type=str, default="best.pt", help="model save name")
    
    parser.add_argument('--encoder_name', type=str, default='tu-xception71', help="encoder name")
    parser.add_argument('--encoder_weights', type=str, default="imagenet", help="pre-trained weights for encoder initialization")
    parser.add_argument('--clahe', type=bool, default=False, help='clahe augmentation')
    parser.add_argument('--cp', type=bool, default=False, help='copypaste augmentation')
    
    args = parser.parse_args()
    main(**args.__dict__)