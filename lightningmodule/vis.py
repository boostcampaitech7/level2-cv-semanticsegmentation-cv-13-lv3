import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil

# CLAHE 적용 함수
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

# 시각화 및 저장 함수
def visualize_and_save(image_path, save_dir):
    # 이미지 로드
    image = Image.open(image_path)
    image_np = np.array(image)

    # CLAHE 적용
    clahe_image_np = apply_clahe(image_np)
    clahe_image = Image.fromarray(clahe_image_np)

    # 저장 경로 설정
    original_dir = os.path.join(save_dir, 'Original')
    clahe_dir = os.path.join(save_dir, 'CLAHE')

    # 폴더 생성
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(clahe_dir, exist_ok=True)

    # 파일명 추출
    filename = os.path.basename(image_path)

    # 원본 이미지 저장
    image.save(os.path.join(original_dir, filename))
    # CLAHE 이미지 저장
    clahe_image.save(os.path.join(clahe_dir, filename))

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(clahe_image)
    axes[1].set_title("CLAHE Applied Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"comparison_{filename}.png"))
    plt.close()

    print(f"Images saved to {save_dir}")

# 메인 함수
def main(image_folder, save_dir):
    # 기존 폴더 삭제 후 생성
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 이미지 처리
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(image_folder, image_file)
            visualize_and_save(image_path, save_dir)

if __name__ == "__main__":
    # 이미지가 저장된 폴더 경로
    image_folder = "your_image_folder"  # 원본 이미지가 저장된 폴더 경로 설정
    save_dir = "visualize_output"  # 결과 저장 폴더

    main(image_folder, save_dir)