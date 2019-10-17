from torch import nn
from torchsummary import summary
from torchvision.models.segmentation import deeplabv3_resnet101

from config import device


class DIMModel(nn.Module):
    def __init__(self):
        super(DIMModel, self).__init__()
        self.model = deeplabv3_resnet101(pretrained=True, num_classes=256)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = DIMModel().to(device)
    summary(model, (4, 320, 320))
