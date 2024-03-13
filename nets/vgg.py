import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url


# 自己修改池化和全连接层之后定义的完整vgg16网络
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):             # 标准的vgg16架构，用来加载预训练权重
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()              # 载入初始的标准化权重

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 105, 105, 3   -> 105, 105, 64 -> 52, 52, 64
# 52, 52, 64    -> 52, 52, 128  -> 26, 26, 128
# 26, 26, 128   -> 26, 26, 256  -> 13, 13, 256
# 13, 13, 256   -> 13, 13, 512  -> 6, 6, 512
# 6, 6, 512     -> 6, 6, 512    -> 3, 3, 512
# 通过循环创建创建vgg16的特征提取层，并且添加各个nn层定义的参数
def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


# 是否加载vgg16的预训练权重
def VGG16(pretrained, in_channels, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        state_dict = torch.load('model_data/vgg16-397923af.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        # 因为需要下载完整参数，所以需要加载完整的模型，而不能在定义的时候直接删掉池化层和全连接层
    return model


if __name__ == '__main__':
    m =VGG16(pretrained=True, in_channels=3)
    print(m)
