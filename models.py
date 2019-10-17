from torch import nn
from torchsummary import summary

from config import device


class DIMModel(nn.Module):
    def __init__(self):
        super(DIMModel, self).__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    model = DIMModel().to(device)
    summary(model, (4, 320, 320))
