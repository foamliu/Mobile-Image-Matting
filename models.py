import torch
from config import num_classes
from torchvision import models

class DIMModel(nn.Module):
    def __init__(self, n_classes=1, in_channels=4, is_unpooling=True, pretrain=True):
        super(DIMModel, self).__init__()

        self.deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_classes)
        self.pretrain = pretrain


    def forward(self, inputs):
        x = self.deeplabv3(inputs)
        x = self.sigmoid(x)

        return x




if __name__ == '__main__':
    m = DIMModel()

