from constants import CLASSES

import segmentation_models_pytorch as smp

def load_model(encoder_name, encoder_weight):
    return smp.UPerNet(
        encoder_name=encoder_name, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weight,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=len(CLASSES),                     # model output channels (number of classes in your dataset)
    )