import os
from glob import glob
import json
from PIL import Image

def convert_to_yolov8_format(image_dir, label_dir, output_dir, classes):
    """
    다각형 데이터를 YOLOv8 Instance Segmentation 형식으로 변환합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    label_files = glob(os.path.join(label_dir, "*.json"))
    
    for label_file in label_files:
        with open(label_file, "r") as f:
            data = json.load(f)

        image_file = os.path.join(image_dir, data["imagePath"])
        image = Image.open(image_file)
        img_width, img_height = image.size
        
        yolov8_labels = []
        for shape in data["shapes"]:
            class_name = shape["label"]
            if class_name not in classes:
                continue

            class_id = classes.index(class_name)
            points = shape["points"]

            # 폴리곤 좌표 정규화
            x_points = [p[0] / img_width for p in points]
            y_points = [p[1] / img_height for p in points]

            # 바운딩 박스 계산
            x_min, x_max = min(x_points), max(x_points)
            y_min, y_max = min(y_points), max(y_points)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # YOLOv8 형식으로 변환
            segment = []
            for x, y in zip(x_points, y_points):
                segment.append(x)
                segment.append(y)
            yolov8_labels.append(f"{class_id} {x_center} {y_center} {width} {height} " + " ".join(map(str, segment)))

        # 변환된 라벨 저장
        output_label_file = os.path.join(output_dir, os.path.splitext(os.path.basename(label_file))[0] + ".txt")
        with open(output_label_file, "w") as f:
            f.write("\n".join(yolov8_labels))

        print(f"Converted: {label_file} -> {output_label_file}")

# 사용 예시
image_dir = "/path/to/images"  # 이미지 파일 경로
label_dir = "/path/to/annotations"  # 다각형 JSON 라벨 경로
output_dir = "/path/to/yolo_labels"  # YOLOv8 라벨 저장 경로
classes = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
convert_to_yolov8_format(image_dir, label_dir, output_dir, classes)