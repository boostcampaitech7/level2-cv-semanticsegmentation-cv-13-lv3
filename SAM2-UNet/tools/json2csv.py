import json
import pandas as pd

# JSON 파일 경로
json_file = "/data/ephemeral/home/kjh/level2-cv-semanticsegmentation-cv-13-lv3/ensemble/tools/crop_info.json"
csv_file = "./crop_data.csv"

# JSON 로드
with open(json_file, "r") as f:
    data = json.load(f)

# 데이터 변환
rows = []
for image_name, classes in data.items():
    for class_name, coords in classes.items():
        row = {
            "image_name": image_name,
            "class": class_name,
            "min_x": coords["min"][0],
            "min_y": coords["min"][1],
            "max_x": coords["max"][0],
            "max_y": coords["max"][1],
        }
        rows.append(row)

# DataFrame 생성 및 저장
df = pd.DataFrame(rows)
df.to_csv(csv_file, index=False)

print(f"CSV saved to {csv_file}")
