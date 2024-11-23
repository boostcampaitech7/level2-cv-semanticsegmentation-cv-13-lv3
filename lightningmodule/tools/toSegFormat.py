import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from constants import IND2CLASS
import json
from utils import get_sorted_files_by_type

def datu_to_segFormat(reference_json, item):
    annotations = []
    for i in range(len(item['annotations'])):
        cur_class = IND2CLASS[item['annotations'][i]['label_id']]
        for annot in reference_json['annotations']:
            if annot['label'] == cur_class:
                break

        points = []
        for j in range(0, len(item['annotations'][i]['points']), 2):
            points.append([int(item['annotations'][i]['points'][j]), int(item['annotations'][i]['points'][j+1])])

        annotations.append({
            'id': annot['id'],
            'type': annot['type'],
            'attributes': annot['attributes'],
            'points': points,
            'label': cur_class
        })

    data_json = {
        'annotations': annotations,
        'attributes': {},
        'file_id': reference_json['file_id'],
        'parent_path': reference_json['parent_path'],
        'last_modifier_id': reference_json['last_modifier_id'],
        'metadata': reference_json['metadata'],
        'last_workers': reference_json['last_workers'],
        'filename': item['id'].split('/')[-1]+'.png'
    }

    return data_json



if __name__ == '__main__':
    origin_path = './base_processed'
    processed_path = './from_CVAT/default.json'
    to_path = './bbangbbang'

    processed_json = None
    with open(processed_path, 'r') as f:
        processed_json = json.load(f)

    for item in processed_json['items']:
        new_path = item['id'].split('/')
        new_path[0] = 'outputs_json'
        new_path = '/'.join(new_path)
        item['id'] = new_path

        reference_path = os.path.join(origin_path, new_path)+'.json'
        reference_json = None
        new_json = None
        
        with open(reference_path, 'r') as f:
            reference_json = json.load(f)
        # print(reference_json)
        json_data = datu_to_segFormat(reference_json, item)
        
        os.makedirs(os.path.join(to_path, '/'.join(new_path.split('/')[:-1])), exist_ok=True)
        with open(os.path.join(to_path, new_path+'.json'), 'w') as f:
            json.dump(json_data, f, indent=4)