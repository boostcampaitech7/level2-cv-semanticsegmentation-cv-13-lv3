import albumentations as A
import numpy as np
import torch
from torchvision.transforms import RandomErasing as TorchRandomErasing

class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        if not isinstance(scale, (tuple, list)) or not isinstance(ratio, (tuple, list)):
            raise TypeError("Scale and Ratio must be a sequence (tuple or list).")
        self.random_erasing = TorchRandomErasing(p=p, scale=tuple(scale), ratio=tuple(ratio), value=0)

    def __call__(self, **kwargs):
        image = kwargs["image"]
        image_tensor = torch.tensor(image).permute(2, 0, 1)  
        augmented_tensor = self.random_erasing(image_tensor)
        return {"image": augmented_tensor.permute(1, 2, 0).numpy()} 

    
def parse_transforms(transform_configs):
    transforms = []
    for transform in transform_configs:
        t_type = transform["type"]
        t_params = transform.get("params", {})
        if hasattr(A, t_type):
            transforms.append(getattr(A, t_type)(**t_params))
        elif t_type == "RandomErasing":
            transforms.append(RandomErasing(**t_params))
        else:
            raise ValueError(f"Unknown transform type: {t_type}")
    return A.Compose(transforms)

def Cutout(max_h_size=32, max_w_size=32, always_apply=False, p=0.5):
    return A.CoarseDropout(
        max_holes=8,  # Default value for max holes (you can adjust this)
        max_height=max_h_size,
        max_width=max_w_size,
        min_height=1,
        min_width=1,
        always_apply=always_apply,
        p=p
    )

def Grid_dropout(ratio=0.5, unit_size_min=32, unit_size_max=64, holes_number_x=None, holes_number_y=None, p=0.5):
    return A.GridDropout(ratio=ratio, unit_size_min=unit_size_min, unit_size_max=unit_size_max, 
                         holes_number_x=holes_number_x, holes_number_y=holes_number_y, p=p)

def Mixup(images, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(images))
    mixed_images = lam * images + (1 - lam) * images[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_images, mixed_labels

def Cutmix(images, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    rand_index = np.random.permutation(len(images))
    bbx1, bby1, bbx2, bby2 = Rand_bbox(images.shape, lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[-2] * images.shape[-1]))
    labels = lam * labels + (1 - lam) * labels[rand_index]
    return images, labels

def Snapmix(images, masks, model, alpha=0.4):
    raise NotImplementedError("Implement SnapMix using CAMs.")

def Rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2