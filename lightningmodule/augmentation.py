import cv2
import numpy as np
import albumentations as A

class CLAHEAugmentation:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def albumentations_clahe(self):
        return A.CLAHE(clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size, always_apply=True)

class EqualizeHistAugmentation:
    @staticmethod
    def apply_equalize_hist(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        min_val, max_val = np.percentile(gray, (2, 98))
        stretched = np.clip((gray - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

        equalized = cv2.equalizeHist(stretched)

        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def load_transforms(args):
    transform = [
        A.Resize(args.input_size, args.input_size),
        # A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0)
        # A.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.005, hue=0.005, p=1.0)
    ]
    
    if args.clahe:
        clahe_clip_limit = args.clahe_clip_limit
        clahe_tile_grid_size = tuple(args.clahe_tile_grid_size)
        print(f"Using CLAHE augmentation with clipLimit={clahe_clip_limit}, tileGridSize={clahe_tile_grid_size}")
        clahe_aug = CLAHEAugmentation(clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
        transform.append(clahe_aug.albumentations_clahe())
        
    transform = A.Compose(transform)
    return transform