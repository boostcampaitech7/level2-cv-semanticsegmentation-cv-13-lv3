import torch
import torch.nn as nn
import timm

class CustomModel(nn.Module):
    def __init__(self, num_classes=29, backbone="resnet34"):
        super(CustomModel, self).__init__()
        self.base_module = timm.create_model(backbone, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.base_module(x)


def build_model(stage, backbone="resnet34", num_classes=29):
    if stage == 1:
        return CustomModel(num_classes=num_classes, backbone=backbone)
    elif stage == 2:
        return CustomModel(num_classes=num_classes, backbone=backbone)
    else:
        raise ValueError("Unsupported stage provided.")