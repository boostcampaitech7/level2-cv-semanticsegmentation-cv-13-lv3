from constants import CLASSES

import segmentation_models_pytorch as smp

def load_model():
    return smp.UPerNet(
        classes=len(CLASSES),                     # model output channels (number of classes in your dataset)
    )