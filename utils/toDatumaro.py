import json

# segFormat to datu
def segFormat_to_datu_with_metadata(data_json):
    # 원래 라벨 매핑 생성
    annotations = data_json["annotations"]
    labels = [{"name": anno["label"], "parent": "", "attributes": []} for anno in annotations]
    label_map = {label["name"]: idx for idx, label in enumerate(labels)}
    
    # items와 annotations 변환
    items = {
        "id": data_json["filename"].split(".")[0],
        "annotations": [
            {
                "id": idx,
                "type": "polygon",
                "attributes": {"occluded": False},  # 기본 속성
                "group": 0,
                "label_id": label_map[anno["label"]],
                "points": [coord for point in anno["points"] for coord in point],
                "z_order": 0
            }
            for idx, anno in enumerate(annotations)
        ],
        # 원본의 추가 정보 포함
        "metadata": data_json["metadata"],
        "attributes": data_json["attributes"],
        "last_workers": data_json["last_workers"],
        "attr": {"frame": 0},
        "point_cloud": {"path": ""},
        "original_ids": {idx:anno["id"] for idx, anno in enumerate(annotations)}
    }
    
    # 변환된 JSON 구조 생성
    datu_json = {
        "info": {},  # info는 비어있음
        "categories": {
            "label": {"labels": labels, "attributes": ["occluded"]},
            "points": {"items": []}
        },
        "items": [items]
    }
    
    return datu_json

# 변환 및 저장
with open("utils/origin/image1661144825412.json", "r") as file:
    data_json = json.load(file)

datu_json = segFormat_to_datu_with_metadata(data_json)

with open("utils/trans/toDatu.json", "w") as file:
    json.dump(datu_json, file, indent=4)
