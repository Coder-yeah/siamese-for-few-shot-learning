import torch
import torch.nn as nn
from torch.nn import Identity

from nets.cigarette3 import CigNet


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


class Siamese(nn.Module):
    # def __init__(self, input_shape, pretrained=False):          # vgg
    def __init__(self):     # resnet
        super(Siamese, self).__init__()
        self.feature = CigNet()
        flat_shape = 256*13*13
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)
        self._initialize_weights()

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#

        x1 = self.feature(x1)
        x2 = self.feature(x2)

        # -------------------------#
        #   横向展平，相减取绝对值，取l1距离
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)

        # -------------------------#
        #   进行两次全连接，变尺寸进行余弦相似度计算？
        # -------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
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
                # nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    m = Siamese()
    print(m)
