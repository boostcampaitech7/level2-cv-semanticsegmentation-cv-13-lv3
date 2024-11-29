import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import json
from constants import CLASSES, CLASS2IND
from utils import get_sorted_files_by_type

def segFormat_to_datu(json_path):
    with open(json_path, 'r') as f:
        # 원래 라벨 매핑 생성
        json_data = json.load(f)
        annotations = json_data["annotations"]
        labels = [{"name": CLASSES[i], "parent": "", "attributes": []} for i in range(len(CLASSES))]
        label_map = {label["name"]: idx for idx, label in enumerate(labels)}

        # items와 annotations 변환
        items = {
            "id": os.path.join(json_path.split('/')[-2], json_data["filename"].split(".")[0]),
            "annotations": [
                {
                    "id": idx,
                    "type": "polygon",
                    "attributes": {"occluded": False},  # 기본 속성
                    "group": 0,
                    "label_id": CLASS2IND[anno["label"]],
                    "points": [coord for point in anno["points"] for coord in point],
                    "z_order": 0
                }
                for idx, anno in enumerate(annotations)
            ],
            # 원본의 추가 정보 포함
            "attributes": json_data["attributes"],
            "last_workers": json_data["last_workers"],
            "attr": {"frame": 0},
            "point_cloud": {"path": ""}
        }

        # 변환된 JSON 구조 생성
        datu_json = {
            "info": {},  # info는 비어있음
            "categories": {
                "label": {"labels": labels, "attributes": ["occluded"]},
                "points": {"items": []}
            },
            "items": [items]          ################## 여러 파일에 대한 annotations 여기에 무한 append
        }

        return datu_json

if __name__ == '__main__':
    경로 = './hobbang/annotations'
    
    jsons_path = get_sorted_files_by_type('./base_processed', 'json')
    pngs_path = get_sorted_files_by_type('./base_processed', 'png')

    all_data = None
    for path in jsons_path:
        if all_data == None:
            all_data = segFormat_to_datu(path)
        else:
            json_data = segFormat_to_datu(path)
            all_data['items'].append(json_data['items'][0])
    
    os.makedirs(경로, exist_ok=False)

    with open(os.path.join(경로, './hihi.json'), "w") as file:
        json.dump(all_data, file, indent=4)