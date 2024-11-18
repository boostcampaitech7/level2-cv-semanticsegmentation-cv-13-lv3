from constants import CLASSES

import segmentation_models_pytorch as smp

def load_model():
    return smp.UnetPlusPlus(
        encoder_name='resnet152',
        encoder_weights='imagenet',
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=len(CLASSES),                     # model output channels (number of classes in your dataset)
    )