from utils.constants import CLASSES

import segmentation_models_pytorch as smp

def load_model(architecture, encoder_name, encoder_weight):

    model_architectures = {
        "Unet": smp.Unet,
        "UperNet": smp.UPerNet,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "PSPNet": smp.PSPNet,
        "FPN": smp.FPN,
        "Linknet": smp.Linknet,
        "Pan": smp.PAN,
        "MAnet": smp.MAnet
    }
    
    if architecture not in model_architectures:
        raise ValueError(f"지원하지 않는 아키텍처입니다. 사용 가능한 아키텍처: {list(model_architectures.keys())}")


    return model_architectures[architecture](
        encoder_name=encoder_name, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weight,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=len(CLASSES),                     # model output channels (number of classes in your dataset)
    )