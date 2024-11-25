import json
import os

origin_path = 'CVAT/ID363/annotations'              # json 이름 가져오기 위한 ~~
processed_path = 'CVAT/ID363/hihi'                  # 웬만하면 default.json 혼자 있는 폴더를 만들어두는게 보기 편하입니더
name_processed = os.path.join(processed_path, 'default.json')

# datu to segFormat
def datu_to_segFormat_with_metadata(datu_json):
    items = datu_json["items"][0]
    labels = {idx: label["name"] for idx, label in enumerate(datu_json["categories"]["label"]["labels"])}
    
    # annotations 변환
    annotations = [
        {
            "id": items.get("original_ids", {idx:0 for idx in range(29)})[idx],
            "type": "poly_seg",
            "attributes": {},  # 원본 `attributes`는 복원 가능
            "points": [[int(anno["points"][i]), int(anno["points"][i+1])] for i in range(0, len(anno["points"]), 2)],
            "label": labels[anno["label_id"]]
        }
        for idx, anno in enumerate(items["annotations"])
    ]
    # 변환된 JSON 구조 생성
    data_json = {
        "annotations": annotations,
        "attributes": items.get("attributes", {}),  # 원본 `attributes` 복원
        "file_id": items["id"],
        "filename": f"{items['id']}.jpg",
        "parent_path": "",
        "last_modifier_id": "",
        "metadata": items.get("metadata", {"height": 0, "width": 0}),  # 원본 `metadata` 복원
        "last_workers": items.get("last_workers", {}),  # 원본 `last_workers` 복원
    }
    
    return data_json

# 변환 및 저장
with open(os.path.join(processed_path, "default.json"), "r") as file:
    datu_json = json.load(file)

data_json = datu_to_segFormat_with_metadata(datu_json)

with open(os.path.join(processed_path, os.listdir(os.path.join(origin_path))[0]), "w") as file:
    json.dump(data_json, file, indent=4)
