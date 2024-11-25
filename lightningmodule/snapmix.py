from torch.utils.data import Dataset
import random
import numpy as np

class SnapMixDataset(Dataset):
    def __init__(self, base_dataset, beta=1.0, probability=0.5):
        self.base_dataset = base_dataset
        self.beta = beta
        self.probability = probability

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_name1, image1, label1 = self.base_dataset[idx]

        if random.random() > self.probability:
            return image_name1, image1, label1

        idx2 = random.randint(0, len(self.base_dataset) - 1)
        image_name2, image2, label2 = self.base_dataset[idx2]

        mixed_image, mixed_label = self._apply_snapmix(image1, label1, image2, label2)
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