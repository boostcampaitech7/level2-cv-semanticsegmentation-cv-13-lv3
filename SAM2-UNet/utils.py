import numpy as np

import torch
import random

import os

import gspread
from gspread.exceptions import WorksheetNotFound
from gspread_formatting import *
from dotenv import dotenv_values

# 시각화를 위한 팔레트를 설정합니다.
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]


# json 파일이 위치한 경로를 값으로 줘야 합니다.
def Gsheet_param(cfg):
    # env 파일 불러오기
    env_path = "/data/ephemeral/home/env/.env"
    env = dotenv_values(env_path)

    # 서비스 연결

    gc = gspread.service_account(env['JSON_PATH'])


    # url에 따른 spread sheet 열기
    doc = gc.open_by_url(env['URL'])

    # 저장할 변수 dict 선언
    param_dict = dict()

    # User 명
    param_dict['user'] = os.path.abspath(__file__).split("/")[4]

    for idx, (key, value) in enumerate(cfg.items()):
        if idx < 4:
            pass
        else :
            param_dict[key] = str(value)
            

    # sheet에 추가하기 위해서 값들을 list로 저장
    params = [param_dict[k] for k in param_dict]

    # sheet가 없는 경우 Head Row를 구성하기 위해서 Col 명을 list로 저장
    cols = [k.capitalize() for k in param_dict]
    
    try:
        # 워크시트가 있는지 확인
        worksheet = doc.worksheet(cfg.project_name)
    except WorksheetNotFound:
        # 워크시트가 없으면 새로 생성
        worksheet = doc.add_worksheet(title=cfg.project_name, rows="1000", cols="30")
        # Col 명 추가
        worksheet.append_rows([cols])

        # Header Cell 서식 
        header_formatter = CellFormat(
            backgroundColor=Color(0.9, 0.9, 0.9),
            textFormat=TextFormat(bold=True, fontSize=12),
            horizontalAlignment='CENTER',
        )
        
        # Header의 서식을 적용할 범위
        header_range = f"A1:{chr(ord('A') + len(cols) - 1)}1"

        # Header 서식 적용
        format_cell_range(worksheet, header_range, header_formatter)

        # Header Cell의 넓이 조정
        for idx, header in enumerate(cols):
            column_letter = chr(ord('A') + idx)
            width = max(len(header)*10+20,80)
            set_column_width(worksheet, column_letter, width)

        print(f"'{cfg.project_name}' 워크시트가 생성되었습니다.")

    # 실험 인자를 작성한 worksheet
    worksheet = doc.worksheet(cfg.project_name)

    # 실험 인자 worksheet에 추가
    worksheet.append_rows([params])

    # 현재 작성하는 실험 인자들 Cell의 서식
    # 노란색으로 하이라이트
    row_formatter = CellFormat(
        backgroundColor=Color(1, 1, 0),
        textFormat=TextFormat(fontSize=12),
        horizontalAlignment="CENTER"
    )

    # 이전 작성 실험인자들 배경색 원상복구
    rollback_formatter = CellFormat(
        backgroundColor=Color(1.0, 1.0, 1.0)
    )
    
    # 마지막 줄에만 하이라이팅이 들어가야 하므로 마지막 row 저장
    last_row = len(worksheet.get_all_values())
    row_range = f"A{last_row}:{chr(ord('A') + len(cols) - 1)}{last_row}"
    rollback_range = f"A{last_row - 1}:{chr(ord('A') + len(cols) - 1)}{last_row - 1}"
    
    # 헤더셀의 서식이 초기화되는 것을 방지하기 위한 조건문
    if last_row - 1 != 1:
        format_cell_range(worksheet, rollback_range, rollback_formatter)
    
    format_cell_range(worksheet, row_range, row_formatter)


# 시각화 함수입니다. 클래스가 2개 이상인 픽셀을 고려하지는 않습니다.
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_sorted_files_by_type(path,file_type='json'):
    current_dir = os.getcwd()  # 현재 작업 디렉토리 기준으로 상대 경로 생성
    files = {
        os.path.relpath(os.path.join(root, fname), start=current_dir)
        for root, _dirs, files in os.walk(path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == '.' + file_type
    }
    files = sorted(files)

    return files

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)# RLE로 인코딩된 결과를 mask map으로 복원합니다.

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)