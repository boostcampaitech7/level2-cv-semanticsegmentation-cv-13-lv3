import cv2
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

    
def load_transforms(args):
    transform = [
        A.Resize(args.input_size, args.input_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Affine(
            scale=(0.9, 1.1),           # 급격한 크기 변화 방지
            translate_percent={
                'x': (-0.05, 0.05),     # x축 이동
                'y': (-0.05, 0.05)      # y축 이동
            },
            rotate=(-10, 10),           # 의료영상은 급격한 회전 지양
            shear=(-7, 7),              # 약간의 전단 변환
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            cval=0,                     # 빈 공간을 검은색으로
            p=0.9                         
        ),
        A.HorizontalFlip(p=0.5),
        A.Sharpen(
            alpha=(0.5, 0.5),
            lightness=(1.0, 1.0),
            p=0.4), 
        GridMaskAugmentation(num_grid=40, grid_ratio=0.4, fill_value=0, p=0.3),
        A.ElasticTransform(alpha=40, sigma=8, p=0.1),
        A.GridDistortion(num_steps=8, distort_limit=0.5, p=0.2), 
        A.RandomGamma(gamma_limit=(30, 200), p=0.3),
        A.Normalize(normalization='min_max', p=1.0)
    ]
    
    return A.Compose(transform)