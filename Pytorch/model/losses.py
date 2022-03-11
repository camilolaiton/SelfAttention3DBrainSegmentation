from torch import nn
import torch

class diceloss(torch.nn.Module):
    def init(self):
        super(diceloss, self).init()

    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class DiceLoss(nn.Module):
    """DiceLoss.

    .. seealso::
        Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. "V-net: Fully convolutional neural networks for
        volumetric medical image segmentation." 2016 fourth international conference on 3D vision (3DV). IEEE, 2016.

    Args:
        smooth (float): Value to avoid division by zero when images and predictions are empty.

    Attributes:
        smooth (float): Value to avoid division by zero when images and predictions are empty.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target, weights=None):
        loss = 0.

        if weights is not None:
            
            for c in range(len(weights)):
                iflat = prediction.reshape(-1)
                tflat = target.reshape(-1)
                intersection = (iflat * tflat).sum()

                w = weights[c]
                loss += w*(1 - ((2. * intersection + self.smooth) /
                             (iflat.sum() + tflat.sum() + self.smooth)))
        else:
            iflat = prediction.reshape(-1)
            tflat = target.reshape(-1)
            intersection = (iflat * tflat).sum()

            # if (weights is not None):
            #     intersection = torch.mean(weights * intersection)

            loss = - (2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)

        return loss

class WeightedLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        # self.name = f'Weighted {loss.name}'

    def forward(self, inputs, true, weights):
        print(inputs.shape)
        iflat = inputs.contiguous().view(-1)
        wflat = weights.contiguous().view(-1)

        loss_part = self.loss(inputs, true)
        weight_part = torch.mean(iflat * wflat)

        return loss_part + weight_part
        
# class DiceLoss(nn.Module):
#     def __init__(self, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, weights=None, smooth=1):
        
#         # #comment out if your model contains a sigmoid or equivalent activation layer
#         # inputs = torch.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         # print(inputs.shape, " ", targets.shape)

#         intersection = (inputs * targets).sum()
#         if (weights is not None):
#             # print(intersection.shape, " ", weights.shape)
#             # print(type(weights), " ", type(intersection))
#             intersection = torch.mean(weights * intersection)
                   
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         # print(dice, " ", dice.shape)
#         return 1 - dice
        # return compute_per_channel_dice(inputs, targets, weight=weights)

# ALPHA = 0.8
# GAMMA = 2

# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(FocalLoss, self).__init__()

#     def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = torch.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         #first compute binary cross-entropy 
#         BCE = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
#         BCE_EXP = torch.exp(-BCE)
#         focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
#         return focal_loss

class FocalLoss(nn.Module):
    """FocalLoss.

    .. seealso::
        Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
        Proceedings of the IEEE international conference on computer vision. 2017.

    Args:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.

    Attributes:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.
    """

    def __init__(self, gamma=2, alpha=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        input = input.clamp(self.eps, 1. - self.eps)

        cross_entropy = - (target * torch.log(input) + (1 - target) * torch.log(1 - input))  # eq1
        logpt = - cross_entropy
        pt = torch.exp(logpt)  # eq2

        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        balanced_cross_entropy = - at * logpt  # eq3

        focal_loss = balanced_cross_entropy * ((1 - pt) ** self.gamma)  # eq5

        return focal_loss.sum()

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class FocalDiceLoss(nn.Module):
    """FocalDiceLoss.

    .. seealso::
        Wong, Ken CL, et al. "3D segmentation with exponential logarithmic loss for highly unbalanced object sizes."
        International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018.

    Args:
        beta (float): Value from 0 to 1, indicating the weight of the dice loss.
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.

    Attributes:
        beta (float): Value from 0 to 1, indicating the weight of the dice loss.
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
    """
    def __init__(self, beta=1, gamma=2, alpha=0.25):
        super().__init__()
        self.beta = beta
        self.focal = FocalLoss(gamma, alpha)
        self.dice = DiceLoss()

    def forward(self, input, target, weights=None):
        dc_loss = - self.dice(input, target, weights)
        fc_loss = self.focal(input, target)

        # used to fine tune beta
        # with torch.no_grad():
        #     print('DICE loss:', dc_loss.cpu().numpy(), 'Focal loss:', fc_loss.cpu().numpy())
        #     log_dc_loss = torch.log(torch.clamp(dc_loss, 1e-7))
        #     log_fc_loss = torch.log(torch.clamp(fc_loss, 1e-7))
        #     print('Log DICE loss:', log_dc_loss.cpu().numpy(), 'Log Focal loss:', log_fc_loss.cpu().numpy())
        #     print('*'*20)

        loss = torch.log(torch.clamp(fc_loss, 1e-7)) - self.beta * torch.log(torch.clamp(dc_loss, 1e-7))

        return loss

# class Dice_and_Focal_loss(nn.Module):
#     def __init__(self):
#         super(Dice_and_Focal_loss, self).__init__()
#         self.dice = DiceLoss()
#         self.focal = FocalLoss()
    
#     def forward(self, pred, mask, weights=None):
#         dice_loss = self.dice(pred, mask, weights)
#         focal_loss = self.focal(pred, mask)
#         result = dice_loss + focal_loss
#         return result

# class ComboLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(ComboLoss, self).__init__()

    # def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        
    #     #flatten label and prediction tensors
    #     inputs = inputs.view(-1)
    #     targets = targets.view(-1)
        
    #     #True Positives, False Positives & False Negatives
    #     intersection = (inputs * targets).sum()    
    #     dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
    #     inputs = torch.clamp(inputs, eps, 1.0 - eps)       
    #     out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
    #     weighted_ce = out.mean(-1)
    #     combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
    #     return combo