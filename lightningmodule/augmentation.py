import cv2
import random
import numpy as np
import albumentations as A

class GridMaskAugmentation(A.ImageOnlyTransform):
    def __init__(self, num_grid=5, grid_ratio=0.5, fill_value=0, p=0.5):
        """
        Parameters:
        - num_grid (int): 격자의 개수. 숫자가 클수록 격자가 작아짐.
        - grid_ratio (float): 가려지는 영역 비율 (0~1). 비율이 작을수록 간격이 커짐.
        - fill_value (int): 가려진 영역에 채울 값 (기본값: 0).
        - p (float): 증강 적용 확률.
        """
        super().__init__(always_apply=False, p=p)
        self.num_grid = num_grid
        self.grid_ratio = grid_ratio
        self.fill_value = fill_value

    def apply(self, image, **kwargs):
        h, w = image.shape[:2]
        grid_h = h // self.num_grid
        grid_w = w // self.num_grid
        mask = np.ones_like(image, dtype=np.uint8)

        # 격자 간 간격 생성
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                start_h = int(i * grid_h + (1 - self.grid_ratio) * grid_h / 2)
                end_h = int(start_h + self.grid_ratio * grid_h)
                start_w = int(j * grid_w + (1 - self.grid_ratio) * grid_w / 2)
                end_w = int(start_w + self.grid_ratio * grid_w)

                # 해당 격자 부분을 가림
                mask[start_h:end_h, start_w:end_w] = self.fill_value

        return image * mask

class SnapMixAugmentation(A.DualTransform):
    def __init__(self, beta=1.0, probability=0.5, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.beta = beta
        self.probability = probability

    def apply(self, image, mask=None, image2=None, mask2=None, **params):
        if random.random() > self.probability:
            return image, mask
        mixed_image, mixed_mask = snapmix(image, mask, image2, mask2, beta=self.beta)
        return mixed_image, mixed_mask

    def get_params_dependent_on_targets(self, params):
        return params

def snapmix(image1, mask1, image2, mask2, beta=1.0):
    assert image1.shape == image2.shape, "Input images must have the same dimensions"
    assert mask1.shape == mask2.shape, "Input masks must have the same dimensions"

    H, W, C = image1.shape if len(image1.shape) == 3 else (*image1.shape, 1)

    lam = np.random.beta(beta, beta)

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = random.randint(0, W)
    cy = random.randint(0, H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_image = image1.copy()
    mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

    mixed_mask = mask1.copy()
    if len(mask1.shape) == 2:
        mixed_mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
    else:
        mixed_mask[y1:y2, x1:x2, :] = mask2[y1:y2, x1:x2, :]

    return mixed_image, mixed_mask

def load_transforms(args):
    transform = [        
        A.Resize(args.input_size, args.input_size),       
        A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.5),        
        A.Affine(            
            scale=(0.9, 1.1),           # 급격한 크기 변화 방지            
            translate_percent=None,            
            rotate=(-10, 10),           # 의료영상은 급격한 회전 지양            
            shear=None,            
            interpolation=cv2.INTER_LINEAR,            
            mask_interpolation=cv2.INTER_NEAREST,            
            cval=0,                     # 빈 공간을 검은색으로            
            p=0.9                                 
            ),        
        A.HorizontalFlip(p=0.5),        
        A.Normalize(normalization='min_max', p=1.0),    
        ]

    return A.Compose(transform)