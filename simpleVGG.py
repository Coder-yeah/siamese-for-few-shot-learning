import torch
import torchvision.models
from torch import nn

input = torch.ones((1, 3, 224, 224))


class module(nn.Module):
    def __init__(self):
        super(module, self).__init__()
        self.features = torchvision.models.vgg16().features
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x


m = module()


# 先训练一个香烟识别网络，得出权重计算方式，最后直接计算特征向量进行比较
