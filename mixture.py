import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice��ʧ����
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# �Ľ���Focal Loss����
class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# max-pooled adaptive loss����
class MaxPooledAdaptiveLoss(nn.Module):
    def __init__(self, base_loss, pool_size=2):
        super(MaxPooledAdaptiveLoss, self).__init__()
        self.base_loss = base_loss
        self.pool_size = pool_size

    def forward(self, inputs, targets):
        loss = self.base_loss(inputs, targets)
        loss_pooled = F.max_pool2d(loss.unsqueeze(1), self.pool_size).squeeze(1)
        return loss_pooled.mean()

# �ۺ���ʧ����
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = dice_loss
        self.focal_loss = ImprovedFocalLoss(alpha=1, gamma=2)
        self.adaptive_loss = MaxPooledAdaptiveLoss(self.focal_loss, pool_size=2)

    def forward(self, inputs, targets):
        loss_dice = self.dice_loss(inputs, targets)
        loss_focal_adaptive = self.adaptive_loss(inputs, targets)
        return loss_dice + loss_focal_adaptive
