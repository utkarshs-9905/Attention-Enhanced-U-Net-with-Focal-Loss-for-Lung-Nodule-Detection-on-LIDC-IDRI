import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Flatten label and prediction tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        tp = (y_pred * y_true).sum()    
        fp = ((1-y_true) * y_pred).sum()
        fn = (y_true * (1-y_pred)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        focal_tversky = (1 - tversky)**(1/self.gamma)
        
        return focal_tversky

def dice_loss(y_pred, y_true, smooth=1e-6):
    
    #Dice loss for binary segmentation
    
    y_pred = torch.sigmoid(y_pred)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=4/3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for FocalTverskyLoss
        self.beta = beta    # Weight for DiceLoss
        self.focal_tversky = FocalTverskyLoss(gamma=gamma)
        
    def forward(self, y_pred, y_true):
        ft_loss = self.focal_tversky(y_pred, y_true)
        dc_loss = dice_loss(y_pred, y_true)
        return self.alpha * ft_loss + self.beta * dc_loss
