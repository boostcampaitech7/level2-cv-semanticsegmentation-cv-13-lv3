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

class EdgeDetection(A.ImageOnlyTransform):
    def __init__(self, threshold1=100, threshold2=200, p=1.0):
        super().__init__(always_apply=False, p=p)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        
    def apply(self, image, **kwargs):
        # Convert to uint8 for cv2
        image = (image * 255).astype(np.uint8)
        
        # Canny Edge Detection 적용
        # threshold1 : 하단 임계값, threshold2 : 상단 임계값 (보통 1:2 또는 1:3 비율로 설정)
        # 임계값이 높으면 밝기 차이가 큰 edge만 검출, 낮으면 밝기 차이가 적은 edge도 검출
        edges = cv2.Canny(image, threshold1=self.threshold1, threshold2=self.threshold2)
        
        # edges to 3-channel image 변환
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # float64 [0,1] range로 변환
        edges = edges.astype(np.float64) / 255.0
        return edges

def load_transforms(args):
    transform = [
        # A.RandomScale(scale_limit=(0.9, 1.1), p=1.0),
        # X-ray에 적합한 Affine 변환
        A.Affine(
            scale=(0.9, 1.1),           # 급격한 크기 변화 방지
            translate_percent={
                'x': (-0.07, 0.07),     # x축 이동
                'y': (-0.07, 0.07)      # y축 이동
            },
            rotate=(-10, 10),           # 의료영상은 급격한 회전 지양
            shear=(-7, 7),              # 약간의 전단 변환
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            cval=0,                     # 빈 공간을 검은색으로
            fit_output=True,            # 이미지가 잘리지 않도록 조정
            p=1                         # 50% 확률로 적용
        ),
        A.Resize(args.input_size, args.input_size),
        # ColorJitter를 사용하려면 ToFloat 사용해야됨
        # A.ToFloat(max_value=255),
        # A.ColorJitter(brightness=(1.0, 1.1), contrast=(1.0, 1.4), saturation=0, hue=0, p=1.0),
        # ColorJitterAugmentation(brightness=0.1, contrast=0.4, p=1.0),
        # EdgeDetection(threshold1=50, threshold2=100, p=1.0),
        # A.HorizontalFlip(p=0.5),
    ]
    
    if args.clahe:
        clahe_clip_limit = args.clahe_clip_limit
        clahe_tile_grid_size = tuple(args.clahe_tile_grid_size)
        print(f"Using CLAHE augmentation with clipLimit={clahe_clip_limit}, tileGridSize={clahe_tile_grid_size}")
        clahe_aug = CLAHEAugmentation(clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
        transform.append(clahe_aug.albumentations_clahe())
        
    # transform.append(A.Normalize(normalization='min_max', p=1.0))
    # transform.append(A.Normalize(normalization='robust', p=1.0)) # Percentile Normalization : 매우 밝거나 어두운 영역을 좀 무시하고 정규화
    transform = A.Compose(transform)
    return transform