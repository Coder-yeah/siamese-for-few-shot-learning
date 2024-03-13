import torchvision.models


# def VGG16(pretrained, in_channels, **kwargs):
def Resnet50(pretrained):
    model = torchvision.models.resnet50(pretrained)
    # 因为需要下载完整参数，所以需要加载完整的模型，而不能在定义的时候直接删掉池化层和全连接层
    return model
