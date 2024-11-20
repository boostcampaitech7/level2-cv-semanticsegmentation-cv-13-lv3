import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import matplotlib.colors as mcolors
import cv2

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def draw_outline(image, label, is_binary = False):

    draw = ImageDraw.Draw(image)

    for i, class_label in enumerate(label):
        if class_label.max() > 0:  # Only process if the class is present in the image
            color = PALETTE[i] if not is_binary else 1

            # Find the points for the outline
            contours, _ = cv2.findContours(class_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw each contour as a polygon
            for contour in contours:
                pts = [(int(point[0][0]), int(point[0][1])) for point in contour]
                if len(pts) >= 2:
                    draw.polygon(pts, outline=color)

    return image

# CSV 파일 경로와 이미지 폴더 경로
csv_path = './output.csv'
image_dir = '/data/ephemeral/home/data/test/DCM'


# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 점들~
points = list(map(int, df.iloc[0,:]['rle'].split()))
points_pairs = list(zip(points[0::2], points[1::2]))

# 이미지들~
pngs = {
    os.path.relpath(os.path.join(root, fname), start=image_dir)
    for root, _dirs, files in os.walk(image_dir)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}
pngs = sorted(list(pngs))

height = width = 512
origin_size = 2048
colors = list(mcolors.TABLEAU_COLORS.values())[:29]

def plot_csv(idx, ax):
    # 배경 이미지 불러오기 (1채널 -> RGB -> RGBA -> 크기 조정)
    background = Image.open(os.path.join(image_dir, pngs[idx])).resize((width, height))
    background = background.convert("RGB")
    #background_rgb = Image.merge("RGB", (background, background, background))
    #background_array = np.array(background_rgb, dtype=np.float32) / 255.0
    #image = np.zeros((width, height,4), dtype=np.float32)

    # 29개의 개별 마스크
    lst = df.values[(idx*29):(idx*29)+29]

    # 복사본으로 겹쳐 그릴 수 있도록 준비
    #mask_total = image.copy()

    masks = []

    for j in range(len(lst)):
        # RLE 데이터 추출 및 디코딩
        rle_data = list(map(int, lst[j][2].split()))
        mask = np.zeros(origin_size * origin_size, dtype=np.uint8)  # 원본 크기로 디코딩
        for k in range(0, len(rle_data), 2):
            start = rle_data[k]
            length = rle_data[k + 1]
            mask[start:start + length] = 1
        mask = mask.reshape((origin_size, origin_size))

        # 마스크를 512x512로 리사이즈
        mask = np.array(Image.fromarray(mask).resize((width, height), Image.NEAREST))
   
        masks.append(mask)
             
        # # 마스크 색상 설정
        # color = mcolors.to_rgba(colors[j % len(colors)], alpha=1.0)
        # color_mask = np.zeros((height, width, 4), dtype=np.float32)
        # for c in range(3):  # RGB 채널
        #     color_mask[:, :, c] = color[c] * mask
        # color_mask[:, :, 3] = color[3] * mask  # Alpha 채널
        
        # # 총합 마스크에 겹쳐 그리기
        # mask_total += color_mask

    background = draw_outline(background, masks)

    # 유효 범위로 클리핑 (0~1 사이로 조정)
    # mask_total = np.clip(mask_total, 0, 1)

    # 최종 이미지 표시
    ax.imshow(background)
    ax.axis('off')
    ax.set_title(f'{idx} {pngs[idx]}')

size=12
for idx in range(0,50):
    fig, axes = plt.subplots(1,1,figsize=[size,size])
    plot_csv(idx, axes)
    fig.savefig(str(idx) + '.png', bbox_inches="tight", pad_inches=0)