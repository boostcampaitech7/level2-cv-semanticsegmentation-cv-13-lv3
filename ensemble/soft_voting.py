import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lightningmodule'))
from model_lightning import SegmentationModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mmsegmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SAM2-UNet'))

import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F

from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

#from model_lightning import SegmentationModel

def preprocess_images_batch(images, min_pos, max_pos, inference_size=(1024, 1024)):
    """
    이미지 배치에서 크롭, 패딩, 리사이즈를 한 번에 수행합니다.
    Args:
        images: torch.Tensor (B, C, H, W)
        min_pos: torch.Tensor (B, 2) - 각 이미지의 크롭 최소 좌표 (x, y)
        max_pos: torch.Tensor (B, 2) - 각 이미지의 크롭 최대 좌표 (x, y)
        target_size: tuple - 리사이즈 대상 크기 (H, W)
    Returns:
        resized_images: torch.Tensor (B, C, target_H, target_W)
        crop_offsets: torch.Tensor (B, 2) - 크롭 좌표 오프셋 (x, y)
        pad_offsets: torch.Tensor (B, 2) - 패딩 좌표 오프셋 (x, y)
        original_sizes: torch.Tensor (B, 2) - 크롭된 이미지의 원본 크기 (H, W)
    """
    B, C, H, W = images.shape
    min_x, min_y = min_pos[:, 0], min_pos[:, 1]
    max_x, max_y = max_pos[:, 0], max_pos[:, 1]

    crop_w, crop_h = max_x - min_x, max_y - min_y
    

    cropped_images = []
    crop_offsets = []
    original_sizes = []

    # 1. Crop Images
    for i in range(B):
        # 이미지 크기에 따라 여백 비율 조정
        width_margin = int((H - crop_w[i]) * 0.005)  
        height_margin = int((W - crop_h[i]) * 0.005)  

        cropped_x = min_x[i] - width_margin
        cropped_y = min_y[i] - height_margin

        cropped = images[i, :, 
                         cropped_y:max_y[i]+height_margin, 
                         cropped_x:max_x[i]+width_margin]  # 크롭
        cropped_images.append(cropped)
        crop_offsets.append((cropped_x, cropped_y))
        print(f'cropped offset : {(cropped_x, cropped_y)}')

    # 2. Pad to Square
    max_crop_H = max([img.shape[1] for img in cropped_images])  # 최대 높이
    max_crop_W = max([img.shape[2] for img in cropped_images])  # 최대 너비
    max_dim = max(max_crop_H, max_crop_W)  # 정사각형 기준 크기

    padded_images = []
    pad_offsets = []

    for cropped in cropped_images:
        pad_H = max_dim - cropped.shape[1]
        pad_W = max_dim - cropped.shape[2]
        # Ensure symmetric padding
        padded = F.pad(
            cropped,
            (pad_W // 2, pad_W - pad_W // 2, pad_H // 2, pad_H - pad_H // 2),  # (left, right, top, bottom)
            mode="constant",
            value=0,
        )
        padded_images.append(padded)
        original_sizes.append((padded.shape[1], padded.shape[2]))  # (H, W)
        pad_offsets.append((pad_W // 2, pad_H // 2))

    # 3. Stack Padded Images
    padded_images = torch.stack(padded_images, dim=0)  # (B, C, max_dim, max_dim)

    # 4. Resize to Target Size
    resized_images = F.interpolate(padded_images, size=inference_size, mode="bilinear", align_corners=False)

    return resized_images, torch.tensor(crop_offsets), torch.tensor(pad_offsets), torch.tensor(original_sizes)

def restore_to_original_sizes(predictions, original_sizes, crop_offsets, pad_offsets, target_size=(2048, 2048)):
    """
    모델의 예측을 원본 크기로 복원합니다.
    Args:
        predictions: torch.Tensor (B, C, target_H, target_W)
        original_sizes: torch.Tensor (B, 2)
        crop_offsets: torch.Tensor (B, 2)
        pad_offsets: torch.Tensor (B, 2)
        target_size: tuple
    Returns:
        restored_outputs: torch.Tensor (B, C, target_H, target_W)
    """
    B, C, _, _ = predictions.shape
    target_H, target_W = target_size

    restored_outputs = torch.zeros((B, C, target_H, target_W), device=predictions.device)
    for i in range(B):
        # Resize back to cropped size
        orig_H, orig_W = original_sizes[i]
        resized_output = F.interpolate(
            predictions[i].unsqueeze(0), size=(orig_H, orig_W), mode="bilinear", align_corners=False
        ).squeeze(0)

                # Determine the position in the target canvas
        crop_x, crop_y = crop_offsets[i]
        pad_x, pad_y = pad_offsets[i]

        # Adjust the position: take crop_offsets into account, subtract pad_offsets
        start_x = crop_x - pad_x
        start_y = crop_y - pad_y
        end_x = start_x + orig_W
        end_y = start_y + orig_H

        # Ensure the coordinates are valid within the canvas size
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(target_W, end_x)
        end_y = min(target_H, end_y)


        print(f'restored offset : {(start_x, end_x)}')

        # Place the resized output into the restored canvas
        restored_outputs[i, :, start_y:end_y, start_x:end_x] = resized_output[:, :end_y-start_y, :end_x-start_x]

    return restored_outputs


    def ensemble_palm_batch(self, images, crop_infos):
        """
        배치 단위로 이미지를 처리합니다.
        Args:
            images: torch.Tensor (B, C, H, W)
            crop_infos: List of dicts containing 'min' and 'max' for each image
        Returns:
            palm_outputs: torch.Tensor (B, C, H, W)
        """
        B = images.size(0)
        inference_size = (1024, 1024)

        # Crop 정보 준비
        min_pos = torch.tensor([info['min'] for info in crop_infos], device=images.device)
        max_pos = torch.tensor([info['max'] for info in crop_infos], device=images.device)

        # 1,2,3. Preprocess Images (Crop, Pad, Resize)
        resized_images, crop_offsets, pad_offsets, original_sizes = preprocess_images_batch(
            images, min_pos, max_pos, inference_size=inference_size
        )

        print(f'interpolated size (should be 1k) : {(resized_images.shape)}')

        # 4. Forward pass
        palm_outputs = self(resized_images)

        # 3. Restore to Original Sizes
        restored_outputs = restore_to_original_sizes(palm_outputs, original_sizes, crop_offsets, pad_offsets)

        return restored_outputs

class EnsembleDataset(Dataset):
    """
    Soft Voting을 위한 DataSet 클래스입니다. 이 클래스는 이미지를 로드하고 전처리하는 작업과
    구성 파일에서 지정된 변환을 적용하는 역할을 수행합니다.

    Args:
        fnames (set) : 로드할 이미지 파일 이름들의 set
        cfg (dict) : 이미지 루트 및 클래스 레이블 등 설정을 포함한 구성 객체
        tf_dict (dict) : 이미지에 적용할 Resize 변환들의 dict
    """    
    def __init__(self, fnames, cfg, tf_dict):
        self.fnames = np.array(sorted(fnames))
        self.image_root = cfg.image_root
        self.tf_dict = tf_dict
        self.ind2class = {i : v for i, v in enumerate(cfg.CLASSES)}

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, item):
        """
        지정된 인덱스에 해당하는 이미지를 로드하여 반환합니다.
        Args:
            item (int): 로드할 이미지의 index

        Returns:
            dict : "image", "image_name"을 키값으로 가지는 dict
        """        
        image_name = self.fnames[item]
        image_path = osp.join(self.image_root, image_name)
        image = cv2.imread(image_path)

        assert image is not None, f"{image_path} 해당 이미지를 찾지 못했습니다."
        
        image = image / 255.0
        return {"image" : image, "image_name" : image_name}

    def collate_fn(self, batch):
        """
        배치 데이터를 처리하는 커스텀 collate 함수입니다.

        Args:
            batch (list): __getitem__에서 반환된 데이터들의 list

        Returns:
            dict: 처리된 이미지들을 가지는 dict
            list: 이미지 이름으로 구성된 list
        """        
        images = [data['image'] for data in batch]
        image_names = [data['image_name'] for data in batch]
        inputs = {"images" : images}

        image_dict = self._apply_transforms(inputs)

        image_dict = {k : torch.from_numpy(v.transpose(0, 3, 1, 2)).float()
                      for k, v in image_dict.items()}
        
        for image_size, image_batch in image_dict.items():
            assert len(image_batch.shape) == 4, \
                f"collate_fn 내부에서 image_batch의 차원은 반드시 4차원이어야 합니다.\n \
                현재 shape : {image_batch.shape}"
            assert image_batch.shape == (len(batch), 3, image_size, image_size), \
                f"collate_fn 내부에서 image_batch의 shape은 ({len(batch)}, 3, {image_size}, {image_size})이어야 합니다.\n \
                현재 shape : {image_batch.shape}"

        return image_dict, image_names
    
    def _apply_transforms(self, inputs):
        """
        입력된 이미지에 변환을 적용합니다.

        Args:
            inputs (dict): 변환할 이미지를 포함하는 딕셔너리

        Returns:
            dict : 변환된 이미지들
        """        
        return {
            key: np.array(pipeline(**inputs)['images']) for key, pipeline in self.tf_dict.items()
        }


def encode_mask_to_rle(mask):
    # mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_models(cfg, device):
    """
    구성 파일에 지정된 경로에서 모델을 로드합니다.

    Args:
        cfg (dict): 모델 경로가 포함된 설정 객체
        device (torch.device): 모델을 로드할 장치 (CPU or GPU)

    Returns:
        dict: 처리 이미지 크기별로 모델을 그룹화한 dict
        int: 로드된 모델의 총 개수
    """    
    model_dict = {}
    model_count = 0

    print("\n======== Model Load ========")
    # inference 해야하는 이미지 크기 별로 모델 순차저장
    for key, paths in cfg.model_paths.items():
        if len(paths) == 0:
            continue
        model_dict[key] = []
        print(f"{key} image size 추론 모델 {len(paths)}개 불러오기 진행 시작")
        for path in paths:
            print(f"{osp.basename(path)} 모델을 불러오는 중입니다..", end="\t")
            model = torch.load(path).to(device)
            model.eval()
            model_dict[key].append(model)
            model_count += 1
            print("불러오기 성공!")
        print()

    print(f"모델 총 {model_count}개 불러오기 성공!\n")
    return model_dict, model_count


def save_results(cfg, filename_and_class, rles):
    """
    추론 결과를 csv 파일로 저장합니다.

    Args:
        cfg (dict): 출력 설정을 포함하는 구성 객체
        filename_and_class (list): 파일 이름과 클래스 레이블이 포함된 list
        rles (list): RLE로 인코딩된 세크멘테이션 마스크들을 가진 list
    """    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    print("\n======== Save Output ========")
    print(f"{cfg.save_dir} 폴더 내부에 {cfg.output_name}을 생성합니다..", end="\t")
    os.makedirs(cfg.save_dir, exist_ok=True)

    output_path = osp.join(cfg.save_dir, cfg.output_name)
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"{output_path}를 생성하는데 실패하였습니다.. : {e}")
        raise

    print(f"{osp.join(cfg.save_dir, cfg.output_name)} 생성 완료")



def soft_voting(cfg):
    """
    Soft Voting을 수행합니다. 여러 모델의 예측을 결합하여 최종 예측을 생성

    Args:
        cfg (dict): 설정을 포함하는 구성 객체
    """    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fnames = {
        osp.relpath(osp.join(root, fname), start=cfg.image_root)
        for root, _, files in os.walk(cfg.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }

    tf_dict = {image_size : A.Resize(height=image_size, width=image_size) 
               for image_size, paths in cfg.model_paths.items() 
               if len(paths) != 0}
    
    print("\n======== PipeLine 생성 ========")
    for k, v in tf_dict.items():
        print(f"{k} 사이즈는 {v} pipeline으로 처리됩니다.")

    dataset = EnsembleDataset(fnames, cfg, tf_dict)
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.num_workers,
                             drop_last=False,
                             collate_fn=dataset.collate_fn)

    model_dict, model_count = load_models(cfg, device)
    
    filename_and_class = []
    rles = []

    print("======== Soft Voting Start ========")
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Inference...]", disable=False) as pbar:
            for image_dict, image_names in data_loader:
                total_output = torch.zeros((cfg.batch_size, len(cfg.CLASSES), 2048, 2048)).to(device)
                for name, models in model_dict.items():
                    for model in models:
                        outputs = model(image_dict[name].to(device))
                        
                        #디버깅
                        #print(f"Debug: outputs type={type(outputs)}, outputs={outputs}")
                        
                        # outputs가 tuple인 경우 첫 번째 요소만 선택
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                        outputs = torch.sigmoid(outputs)
                        total_output += outputs
                        
                total_output /= model_count
                total_output = (total_output > cfg.threshold).detach().cpu().numpy()

                for output, image_name in zip(total_output, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{dataset.ind2class[c]}_{image_name}")
                
                pbar.update(1)

    save_results(cfg, filename_and_class, rles)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="soft_voting_setting.yaml")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)

    if cfg.root_path not in sys.path:
        sys.path.append(cfg.root_path)
    
    soft_voting(cfg)