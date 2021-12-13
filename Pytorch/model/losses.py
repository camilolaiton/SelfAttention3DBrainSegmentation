from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class Dice_and_Focal_loss(nn.Module):
    def __init__(self):
        super(Dice_and_Focal_loss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
    
    def forward(self, pred, mask):
        dice_loss = self.dice(pred, mask)
        focal_loss = self.focal(pred, mask)
        result = dice_loss + focal_loss
        return result

# class ComboLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(ComboLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         #True Positives, False Positives & False Negatives
#         intersection = (inputs * targets).sum()    
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
#         inputs = torch.clamp(inputs, eps, 1.0 - eps)       
#         out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
#         weighted_ce = out.mean(-1)
#         combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
#         return combo