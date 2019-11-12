import torch

from models.deeplab import DeepLab
from torchscope import scope

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16, num_classes=1)
    model.eval()
    input = torch.rand(1, 4, 320, 320)
    output = model(input)
    print(output.size())
    scope(model, (4, 320, 320))
