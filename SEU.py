import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 修改DeepWaterMap模型以包含SE模块
class DeepWaterMapV2(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepWaterMapV2, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(6, 64)
        self.se1 = SELayer(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down2 = conv_block(64, 128)
        self.se2 = SELayer(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.down3 = conv_block(128, 256)
        self.se3 = SELayer(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = conv_block(256, 512)
        self.se_bottleneck = SELayer(512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up3 = conv_block(512, 256)
        self.se_up3 = SELayer(256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = conv_block(256, 128)
        self.se_up2 = SELayer(128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = conv_block(128, 64)
        self.se_up1 = SELayer(64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x1 = self.se1(x1)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        x2 = self.se2(x2)
        p2 = self.pool2(x2)

        x3 = self.down3(p2)
        x3 = self.se3(x3)
        p3 = self.pool3(x3)

        b = self.bottleneck(p3)
        b = self.se_bottleneck(b)

        u3 = self.up3(b)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.conv_up3(u3)
        u3 = self.se_up3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        u2 = self.se_up2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv_up1(u1)
        u1 = self.se_up1(u1)

        out = self.out(u1)
        return torch.sigmoid(out)

# 实例化模型和损失函数
model = DeepWaterMapV2(num_classes=1)
