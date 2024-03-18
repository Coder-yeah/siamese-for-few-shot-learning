import torch
import torchvision.models


# def VGG16(pretrained, in_channels, **kwargs):
from torch.nn import Identity


def Resnet18(pretrained):
    model = torchvision.models.resnet18(pretrained=True)
    # 因为需要下载完整参数，所以需要加载完整的模型，而不能在定义的时候直接删掉池化层和全连接层
    return model


if __name__ == '__main__':
    m = Resnet18(pretrained=True)
    m.avgpool = Identity()
    m.fc = Identity()
    print(m)
    input = torch.ones((1, 3, 224, 224))
    output = m(input)
    print(output.shape)
