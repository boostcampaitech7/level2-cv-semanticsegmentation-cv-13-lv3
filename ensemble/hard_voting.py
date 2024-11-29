import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
import torch

def csv_ensemble(path, save_dir, threshold, device="cuda"):
    def decode_rle_to_mask(rle, height, width):
        """
        RLE 문자열을 디코딩하여 torch.Tensor 마스크로 변환
        """
        if not isinstance(rle, str):
            return torch.zeros((height, width), device=device, dtype=torch.uint8)
        try:
            s = rle.split()
            starts, lengths = [torch.tensor(list(map(int, x)), dtype=torch.long, device=device) for x in (s[0:][::2], s[1:][::2])]
            starts -= 1
            ends = starts + lengths
            mask = torch.zeros(height * width, dtype=torch.uint8, device=device)
            for lo, hi in zip(starts, ends):
                mask[lo:hi] = 1
            return mask.view(height, width)
        except Exception as e:
            print(f"Error decoding RLE: {rle} - {e}")
            return torch.zeros((height, width), device=device, dtype=torch.uint8)

    def encode_mask_to_rle(mask):
        """
        torch.Tensor 마스크를 RLE 문자열로 변환
        """
        mask = mask.flatten().cpu().numpy()
        pixels = np.concatenate([[0], mask, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    # 폴더 내 모든 CSV 파일 경로를 가져옵니다.
    csv_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_paths:
        raise ValueError(f"No CSV files found in folder: {path}")
    
    # csv의 기본 column(column이지만 사실 row입니다.. default 8352)
    csv_column = 8352  # 기본 column 수
    csv_data = [pd.read_csv(path) for path in csv_paths]

    file_num = len(csv_data)
    filename_and_class = []
    rles = []

    print(f"앙상블할 모델 수: {file_num}, threshold: {threshold}")  # 정보 출력 추가

    for index in tqdm(range(csv_column)):
        model_rles = []
        for data in csv_data:
            rle = data.iloc[index]['rle']
            model_rles.append(decode_rle_to_mask(rle, 2048, 2048))

        # GPU에서 마스크 병합
        image = torch.zeros((2048, 2048), device=device, dtype=torch.uint8)
        for model in model_rles:
            image += model
            
        # Threshold 적용
        result_image = (image > threshold).to(torch.uint8)

        # RLE 인코딩
        rles.append(encode_mask_to_rle(result_image))
        filename_and_class.append(f"{csv_data[0].iloc[index]['class']}_{csv_data[0].iloc[index]['image_name']}")

        # GPU 메모리 최적화
        del model_rles, image, result_image
        torch.cuda.empty_cache()

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    # 기본 DataFrame의 구조 = image_name, class, rle
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    # 최종 ensemble output 저장
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    df.to_csv(save_dir, index=False)
    print("\n앙상블 종료!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./checkpoint/csv/tmp")
    parser.add_argument("--save_dir", type=str, default="./results/hard_voting_th4.csv")
    parser.add_argument("--threshold", type=int, default=4, help="Threshold for voting.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")

    args = parser.parse_args()

    csv_ensemble(args.path, args.save_dir, args.threshold, device=args.device)
