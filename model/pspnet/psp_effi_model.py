from constants import CLASSES

import segmentation_models_pytorch as smp

def load_model():
    return smp.PSPNet(
        encoder_name="efficientnet-b7",  # Swin Transformer 백본 사용
        encoder_weights="imagenet",  # ImageNet 사전학습된 가중치 사용
        in_channels=3,              # 입력 채널
        classes=len(CLASSES),       # 클래스 개수에 따른 출력 채널
    )