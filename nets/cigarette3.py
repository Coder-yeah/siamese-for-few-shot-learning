import torch
from torch import nn


class SENet(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SENet, self).__init__()
        self.relu = nn.ReLU(True)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel//ratio, False),
            nn.ReLU(),
            nn.Linear(in_channel//ratio, in_channel, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu(x)
        b, c, h, w = x.size()
        avg = self.avg(x).view(b, c)
        fc = self.fc(avg).view(b, c, 1, 1)
        return x*fc             # 返回乘了通道权重的特征图


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.senet1 = SENet(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.senet2 = SENet(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.senet1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.senet2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        return x


def CigNet():
    m = Net()
    return m
