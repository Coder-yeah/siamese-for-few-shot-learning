import torch
import torch.nn as nn
from torch.nn import Identity

from nets.vgg import VGG16
from resnet import Resnet50


# 每次通过一个尺寸为2，stride为2的池化层才会导致图片的输入尺寸发生变化
def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            # 输出尺寸计算公式，vgg16的卷积核size=3,padding=1,stride=1，并不会导致尺寸变化，因此不做记录
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width) * get_output_length(height)


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


class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):          # vgg
    # def __init__(self, pretrained=False):     # resnet
        super(Siamese, self).__init__()
        # resnet:
        # self.resnet = Resnet50(pretrained=pretrained)
        # self.resnet.avgpool = Identity()
        # self.resnet.fc = Identity()
        # resnet:
        # flat_shape = 32768

        # vgg16:
        self.vgg = VGG16(pretrained, 3)  # 主干网络vgg16，已经加载好预训练权重
        del self.vgg.avgpool
        del self.vgg.classifier
        # 最后使用的卷积核数量是512
        # vgg16卷积结束的尺寸:
        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        # vgg16
        # self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        # self.fully_connect2 = torch.nn.Linear(512, 1)

        # 增加通道注意力机制:主要是变换通道数
        # self.vgg = VGG16(pretrained, 3)
        # del self.vgg.avgpool
        # del self.vgg.classifier
        # self.vgg.features[1] = SENet(64)         # 插入注意力模块到卷积层
        # self.vgg.features[6] = SENet(128)

        # vgg+余弦距离
        self.fully_connect1 = torch.nn.Linear(512, 256)
        self.fully_connect2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        # vgg16: 3*3*512
        x1 = self.vgg.features(x1)              # 尺寸为什么会是2*512*3*3的？--->dataset代码
        x2 = self.vgg.features(x2)

        # resnet:
        # x1 = self.resnet(x1)
        # x2 = self.resnet(x2)

        # -------------------------#
        #   横向展平，相减取绝对值，取l1距离
        # -------------------------#
        # x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)
        # x = torch.abs(x1 - x2)

        # 计算dim=1的余弦相似度，进行连接
        x1 = x1.view(8, 512, 9)                         # why is 8 = 2*4（batchsize的两倍）？？
        x2 = x2.view(8, 512, 9)
        x = torch.cosine_similarity(x1, x2, dim=2)          # 返回每一个特征图的相似度度量，512维的向量，不用展平了

        # -------------------------#
        #   进行两次全连接，变尺寸进行余弦相似度计算？
        # -------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x


if __name__ == '__main__':
    input = torch.ones((2, 3, 224, 224))
    print(input.shape)
    # 创建实例，没有传播
    m = Siamese(input_shape=input.shape)
    print(m)
