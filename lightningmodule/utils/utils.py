import numpy as np

import torch
import random

import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 시각화를 위한 팔레트를 설정합니다.
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

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

def calculate_confusion_matrix(y_true, y_pred, num_classes, threshold):
    """
    y_true: ground truth labels (B, C, H, W)
    y_pred: predicted labels (B, C, H, W)
    num_classes: number of classes (C)
    returns: confusion matrix of shape (num_classes, num_classes) with ratio values
    """
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_true.device)
    
    # 각 클래스별 전체 픽셀 수 계산
    total_pixels_per_class = y_true.shape[1] * y_true.shape[2]
    y_pred = (y_pred > threshold)

    # 각 클래스의 Calculate confusion matrix
    for i in range(num_classes):
        for j in range(num_classes):
            true_i = y_true[i].flatten()
            pred_j = y_pred[j].flatten()
            # intersection 계산 후 비율로 변환
            intersection = torch.sum(true_i * pred_j)
            # i번째 클래스의 전체 픽셀 수로 나누어 비율 계산
            ratio = intersection / (total_pixels_per_class + 1e-6)  # 0 나눗셈 방지
            confusion_matrix[i, j] = ratio.item()
    
    return confusion_matrix

def calculate_metrics(confusion_matrix):
    """
    Calculate metrics from confusion matrix.
    """
    # 각 클래스에 대한 TP, FP, FN 계산
    TP = confusion_matrix.diag()
    FP = confusion_matrix.sum(dim=0) - TP
    FN = confusion_matrix.sum(dim=1) - TP

    # Precision, Recall, F1 Score 계산
    precision = TP / (TP + FP + 1e-6)  # 0 나눗셈 방지
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    return precision, recall, f1_score

def save_confusion_matrix(confusion_matrix, classes):

    # Create figure and axes
    plt.figure(figsize=(15, 12))
    
    # Create heatmap using the averaged confusion matrix
    sns.heatmap(confusion_matrix.cpu().numpy(), 
                annot=True,
                fmt='.3f',  # 소수점 3자리까지 표시
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")