class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        smooth = 1
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.float32)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE)
        FocalLoss = at * (1 - pt) ** self.gamma * BCE
        return FocalLoss.mean()

class MaxPooledAdaptiveLoss(nn.Module):
    def forward(self, inputs, targets):
        loss = torch.abs(inputs - targets) # simple L1 loss for demonstration
        max_pooled_loss = F.max_pool2d(loss, kernel_size=2, stride=1)
        return torch.mean(max_pooled_loss)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.adaptive_loss = MaxPooledAdaptiveLoss()

    def forward(self, inputs, targets):
        return self.dice_loss(inputs, targets) + self.focal_loss(inputs, targets) + self.adaptive_loss(inputs, targets)
