import torch.nn.functional as F
import torch
import torch.nn as nn

class IOU_loss(nn.Module):
   def __init__(self, smooth=1):
       super().__init__()
       self.smooth = smooth
       
   def forward(self, inputs, targets):
       inputs = F.sigmoid(inputs)      
       inputs = inputs.view(-1)
       targets = targets.view(-1)
       intersection = (inputs * targets).sum()
       total = (inputs + targets).sum()
       union = total - intersection 
       IoU = (intersection + self.smooth)/(union + self.smooth)
       return 1 - IoU

class DiceLoss(nn.Module):
   def __init__(self, smooth=1.):
       super().__init__()
       self.smooth = smooth
       
   def forward(self, pred, target):
       pred = pred.contiguous()
       target = target.contiguous()   
       intersection = (pred * target).sum(dim=2).sum(dim=2)
       loss = (1 - ((2. * intersection + self.smooth) / 
               (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
       return loss.mean()

class FocalLoss(nn.Module):
   def __init__(self, alpha=.25, gamma=2):
       super().__init__() 
       self.alpha = alpha
       self.gamma = gamma
       
   def forward(self, inputs, targets):     
       BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
       BCE_EXP = torch.exp(-BCE)
       loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
       return loss

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

    
def iou_loss(inputs, targets, smooth=1):
    # inputs는 이미 sigmoid를 통과했다고 가정
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    
    IoU = (intersection + smooth)/(union + smooth)
    return 1 - IoU

class calc_iou_loss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        # BCE loss 계산
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        
        # IoU loss 계산을 위해 sigmoid 적용
        pred = F.sigmoid(inputs)
        iou = iou_loss(pred, targets)
        
        # 가중치를 적용한 최종 loss
        loss = bce * self.bce_weight + iou * (1 - self.bce_weight)
        
        return loss

class calc_dice_loss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        # BCE loss 계산
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        
        # Dice loss 계산을 위해 sigmoid 적용
        pred = F.sigmoid(inputs)
        dice = dice_loss(pred, targets)
        
        # 가중치를 적용한 최종 loss
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        
        return loss