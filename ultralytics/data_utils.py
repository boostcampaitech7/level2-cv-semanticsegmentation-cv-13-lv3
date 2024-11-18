import os
import json
import shutil
from sklearn.model_selection import train_test_split

def convert_coco_to_yolo(coco_json_path, images_dir, valid_classes):
    """
    COCO 형식의 JSON 파일을 YOLO 형식으로 변환하고, 유효한 클래스만 필터링합니다.
    :param coco_json_path: COCO 형식의 JSON 파일 경로
    :param images_dir: 이미지 파일이 저장된 디렉토리 경로
    :param valid_classes: 유효한 클래스 리스트
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']

    for ann in annotations:
        image_id = ann['image_id']
        category_id = ann['category_id']

        if category_id not in valid_classes:
            continue

        image_info = images[image_id]
        image_filename = image_info['file_name']
        image_path = os.path.join(images_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            continue

        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(images_dir, label_filename)

        bbox = ann['bbox']
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        width = bbox[2]
        height = bbox[3]

        image_width = image_info['width']
        image_height = image_info['height']

        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        with open(label_path, 'a') as f:
            f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

def split_dataset(image_dir, train_dir, val_dir, test_size=0.2, random_state=42):
    """
    이미지와 대응하는 .txt 라벨 파일을 train과 val 폴더로 복사하여 분할하는 함수.
    :param image_dir: 원본 이미지와 라벨 파일이 있는 디렉토리
    :param train_dir: train 이미지와 라벨 파일이 저장될 디렉토리
    :param val_dir: val 이미지와 라벨 파일이 저장될 디렉토리
    :param test_size: 검증 세트의 비율 (기본값은 0.2)
    :param random_state: 데이터 분리 시 랜덤 시드 값
    """
    if os.path.exists(train_dir) and os.listdir(train_dir) and os.path.exists(val_dir) and os.listdir(val_dir):
        print(f"{train_dir} 및 {val_dir}에 데이터가 이미 분할되어 있습니다. 분할을 건너뜁니다.")
        return

    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(images) == 0:
        raise ValueError(f"이미지 파일이 {image_dir}에 없습니다. 경로를 확인하세요.")

    train_images, val_images = train_test_split(images, test_size=test_size, random_state=random_state)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for img in train_images:
        shutil.copy2(os.path.join(image_dir, img), os.path.join(train_dir, img))
        label_file = os.path.splitext(img)[0] + '.txt'
        label_path = os.path.join(image_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(train_dir, label_file))

    for img in val_images:
        shutil.copy2(os.path.join(image_dir, img), os.path.join(val_dir, img))
        label_file = os.path.splitext(img)[0] + '.txt'
        label_path = os.path.join(image_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(val_dir, label_file))

    print(f"Train 이미지 수: {len(train_images)}, Val 이미지 수: {len(val_images)}")