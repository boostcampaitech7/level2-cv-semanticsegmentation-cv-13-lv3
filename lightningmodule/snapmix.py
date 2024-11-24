from torch.utils.data import Dataset
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def generate_cam(model, image, target_class=None):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0) 
        outputs = model(image)
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        grad = torch.autograd.grad(outputs[:, target_class].sum(), model.features, retain_graph=True)[0]
        cam = F.relu(grad.mean(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max()
    return cam

class SnapMixDataset(Dataset):
    def __init__(self, base_dataset, model=None, beta=1.0, probability=0.5, use_copypaste=False):
        self.base_dataset = base_dataset
        self.model = model  
        self.beta = beta
        self.probability = probability
        self.use_copypaste = use_copypaste

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        idx2 = idx
        while idx2 == idx:
            idx2 = random.randint(0, len(self.base_dataset) - 1)

        image_name1, image1, label1 = self.base_dataset[idx]
        image_name2, image2, label2 = self.base_dataset[idx2]

        if self.use_copypaste:
            image1, label1 = copypaste(image1, label1)
            image2, label2 = copypaste(image2, label2)

        if random.random() < self.probability:
            mixed_image, mixed_label = self._apply_snapmix(image1, label1, image2, label2)
        else:
            mixed_image, mixed_label = image1, label1

        return image_name1, mixed_image, mixed_label

    def _apply_snapmix(self, image1, label1, image2, label2):
        H, W = image1.shape[1:] 
        lam = np.random.beta(self.beta, self.beta)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = random.randint(0, W), random.randint(0, H)
        x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
        y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)

        mixed_image = image1.clone()
        mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]

        mixed_label = label1.clone()
        mixed_label[:, y1:y2, x1:x2] = label2[:, y1:y2, x1:x2]

        return mixed_image, mixed_label


def copypaste(image, label, alpha=0.5, beta=0.5):
    h, w = image.shape[1:]

    patch_h, patch_w = np.random.randint(h // 4, h // 2), np.random.randint(w // 4, w // 2)
    start_x, start_y = np.random.randint(0, w - patch_w), np.random.randint(0, h - patch_h)
    end_x, end_y = start_x + patch_w, start_y + patch_h

    patch = np.random.randint(0, 256, (image.shape[0], patch_h, patch_w), dtype=np.uint8)
    patch_mask = np.random.randint(0, 2, (label.shape[0], patch_h, patch_w), dtype=np.uint8)

    patch = torch.from_numpy(patch).float() / 255.0
    patch_mask = torch.from_numpy(patch_mask).float()

    overlay = image.clone()
    overlay[:, start_y:end_y, start_x:end_x] = patch
    label_overlay = label.clone()
    label_overlay[:, start_y:end_y, start_x:end_x] = patch_mask

    blended_image = alpha * image + beta * overlay
    blended_label = torch.maximum(label, label_overlay)

    return blended_image, blended_label
