import albumentations as A
import numpy as np
import cv2

def cutout(num_holes=8, max_h_size=32, max_w_size=32, always_apply=False, p=0.5):
    """Cutout augmentation"""
    return A.Cutout(num_holes=num_holes, max_h_size=max_h_size, max_w_size=max_w_size, always_apply=always_apply, p=p)

def grid_dropout(ratio=0.5, unit_size_min=32, unit_size_max=64, holes_number_x=None, holes_number_y=None, p=0.5):
    """GridDropout augmentation"""
    return A.GridDropout(ratio=ratio, unit_size_min=unit_size_min, unit_size_max=unit_size_max, 
                         holes_number_x=holes_number_x, holes_number_y=holes_number_y, p=p)

def mixup(images, labels, alpha=0.4):
    """Mixup augmentation"""
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(images))
    mixed_images = lam * images + (1 - lam) * images[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_images, mixed_labels

def cutmix(images, labels, alpha=0.4):
    """CutMix augmentation"""
    lam = np.random.beta(alpha, alpha)
    rand_index = np.random.permutation(len(images))
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.shape, lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[-2] * images.shape[-1]))
    labels = lam * labels + (1 - lam) * labels[rand_index]
    return images, labels

def snapmix(images, masks, model, alpha=0.4):
    """SnapMix augmentation using CAMs"""
    raise NotImplementedError("Implement SnapMix using CAMs.")

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2