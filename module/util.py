import os
import numpy as np
import matplotlib.pyplot as plt

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_heatmap(preds, target):
    # Heatmap 생성 코드
    return np.abs(preds - target)

def plot_confusion_matrix(matrix, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.show()