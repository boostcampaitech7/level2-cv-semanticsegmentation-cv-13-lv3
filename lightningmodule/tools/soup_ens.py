import torch
import os

def load_checkpoint_weights(checkpoint_dir):
    """
    주어진 디렉토리에서 모든 체크포인트 파일 경로를 반환.
    """
    return [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt") or f.endswith(".pt")
    ]

def save_model_weights(model, save_path):
    """
    모델의 가중치를 지정된 경로에 저장.
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")