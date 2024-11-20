import cv2
import numpy as np
import albumentations as A

class CLAHEAugmentation:
    @staticmethod
    def apply_clahe(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab_clahe = cv2.merge((cl, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    @staticmethod
    def albumentations_clahe():
        return A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True)

class EqualizeHistAugmentation:
    @staticmethod
    def apply_equalize_hist(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        min_val, max_val = np.percentile(gray, (2, 98))
        stretched = np.clip((gray - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

        equalized = cv2.equalizeHist(stretched)

        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)