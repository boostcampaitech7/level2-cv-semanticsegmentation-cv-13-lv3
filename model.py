from constants import CLASSES
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
import segmentation_models_pytorch as smp

def load_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        ignore_mismatched_sizes=True,
        num_labels=len(CLASSES)
    )
    return model