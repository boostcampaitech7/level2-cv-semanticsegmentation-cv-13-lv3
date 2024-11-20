import cv2
import numpy as np
import albumentations as A

class CLAHEEqualizer:
    @staticmethod
    def apply_clahe(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab_clahe = cv2.merge((cl, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_equalize_hist(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def albumentations_clahe():
        return A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True)

if __name__ == "__main__":
    image = cv2.imread('sample_image.jpg')  # 테스트용 이미지
    clahe_image = CLAHEEqualizer.apply_clahe(image)
    hist_image = CLAHEEqualizer.apply_equalize_hist(image)

    cv2.imshow('Original', image)
    cv2.imshow('CLAHE', clahe_image)
    cv2.imshow('Equalized Hist', hist_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
