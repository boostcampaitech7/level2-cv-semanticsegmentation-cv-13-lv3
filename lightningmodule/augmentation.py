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

class ColorJitterAugmentation(A.ImageOnlyTransform):
    def __init__(self, brightness=0.05, contrast=0.2, p=1.0):
        super().__init__(always_apply=False, p=p)
        self.brightness = brightness
        self.contrast = contrast
        
    def apply(self, image, **kwargs):
        # image는 이미 float64이고 [0,1] 범위라고 가정
        
        # Convert to uint8 for cv2
        image_uint8 = (image * 255).astype(np.uint8)
        
        
        # Brightness adjustment
        beta = int(255 * self.brightness)
        # Contrast adjustment
        contrast_factor = 1 + self.contrast
        
        image_uint8 = cv2.convertScaleAbs(image_uint8, alpha=contrast_factor, beta=beta)
        
        # Convert back to float64 [0,1] range
        image = image_uint8.astype(np.float64) / 255.0
        
        return image

def load_transforms(args):
    transform = [
        ColorJitterAugmentation(brightness=0.1, contrast=0.4, p=1.0),
        A.Resize(args.input_size, args.input_size),
    ]
    
    if args.clahe:
        clahe_clip_limit = args.clahe_clip_limit
        clahe_tile_grid_size = tuple(args.clahe_tile_grid_size)
        print(f"Using CLAHE augmentation with clipLimit={clahe_clip_limit}, tileGridSize={clahe_tile_grid_size}")
        clahe_aug = CLAHEAugmentation(clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
        transform.append(clahe_aug.albumentations_clahe())
        
    transform = A.Compose(transform)
    return transform