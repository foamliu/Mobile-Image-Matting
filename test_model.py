import torch

from models.deeplab import DeepLab

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16, num_classes=256)
    model.eval()
    input = torch.rand(1, 3, 320, 320)
    output = model(input)
    print(output.size())
