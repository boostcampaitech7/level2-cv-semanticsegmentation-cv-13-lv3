import cv2
import json
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from constants import CLASSES, PALETTE, CLASS2IND
from utils import get_sorted_files_by_type

# 경로에서 '/'로 나뉜 부분 중 변경하고싶은 부분 지정해서 바꾸기
def modify_path(path, from_name='train', to_name='hihi'):
    new_path = path.split('/')
    new_path[new_path.index(from_name)] = to_name
    return '/'.join(new_path)

# 여백 잘라내고 Radius의 절반 크기까지만큼만 포함하도록 이미지, annotation 잘라내기
def crop_background(img_path, json_path):
    img = cv2.imread(img_path)
    json_data = json.load(open(json_path, 'r'))
    radius_min_y = float('inf')
    radius_max_y = 0
    max_x=max_y=0
    min_x=min_y=float('inf')
    for annot in json_data['annotations']:
        for point in annot['points']:
            max_x = max(max_x, point[0])
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])
            if annot['label'] == 'Radius':
                radius_max_y = max(radius_max_y, point[1])
                radius_min_y = min(radius_min_y, point[1])

    # padding
    min_x -= 20
    max_x += 20
    max_y = (radius_max_y+radius_min_y)//2
    min_y -= 20

    for i in range(29):
        if json_data['annotations'][i]['label'] in ['Radius', 'Ulna']:
            for point in json_data['annotations'][i]['points']:
                if point[1] > max_y:
                    del point

    img = img[min_y:max_y, min_x:max_x] 
    
    for i, annot in enumerate(json_data['annotations']):
        points = []
        for j in range(len(annot['points'])):
            if annot['label'] in ['Radius', 'Ulna'] and annot['points'][j][1] > max_y: ################ 팔뚝 잘랐을 때 이미지 바깥의 point는 빼기
                continue
            points.append([annot['points'][j][0]-min_x, annot['points'][j][1]-min_y])
        json_data['annotations'][i]['points'] = points

    return img, json_data


if __name__ == '__main__':
    # train 경로: 문자열에 train이 있어야함
    # 상대경로는 실행시키는 터미널의 위치 기준
    train_path = '../data/train' 
    folder_name = 'hihi'

    png_path = get_sorted_files_by_type(train_path, 'png')
    json_path = get_sorted_files_by_type(train_path, 'json')

    new_png_path = []
    new_json_path = []

    # 파일 저장할 경로명 지정
    for i in range(len(png_path)):
        new_png_path.append(modify_path(png_path[i], from_name='train', to_name=folder_name))
        new_json_path.append(modify_path(json_path[i], from_name='train', to_name=folder_name))


    # 같은 이름의 폴더 있으면 일단 정지
    new_path = '/'.join(train_path.split('/')[:-1])
    assert folder_name not in os.listdir(new_path), f'폴더가 이미 존재하면 나오는 에러: {os.path.join(new_path, folder_name)}'
    os.makedirs(os.path.join(new_path, folder_name))

    for i in range(len(png_path)):
        img_id_path = '/'.join(new_png_path[i].split('/')[:-1])
        json_id_path = '/'.join(new_json_path[i].split('/')[:-1])
        os.makedirs(img_id_path, exist_ok=True)
        os.makedirs(json_id_path, exist_ok=True)

        # 이미지 자르기
        img, json_data = crop_background(png_path[i], json_path[i])
        
        # 새로운 경로에 새로운 이미지 쓰기
        cv2.imwrite(new_png_path[i], img)
        with open(new_json_path[i], 'w') as f:
            json.dump(json_data, f)