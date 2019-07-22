import math

import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchsummary import summary

from config import device, im_size
from utils import parse_args

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f.append(x)  # [N, 256, 56, 56]
        x = self.layer2(x)
        f.append(x)  # [N, 512, 28, 28]
        x = self.layer3(x)
        f.append(x)  # [N, 1024, 14, 14]
        x = self.layer4(x)
        f.append(x)  # [N, 2048, 7, 7]

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x, f


def resnet18(args, **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(args, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(args, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(args, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(args, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class EastModel(nn.Module):
    def __init__(self, args):
        super(EastModel, self).__init__()

        if args.network == 'r18':
            self.resnet = resnet18(args)
        elif args.network == 'r34':
            self.resnet = resnet34(args)
        elif args.network == 'r50':
            self.resnet = resnet50(args)
        elif args.network == 'r101':
            self.resnet = resnet101(args)
        else:  # args.network == 'r152':
            self.resnet = resnet152(args)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=3072, out_channels=128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=640, out_channels=64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)

        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)

        self.conv8 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, images):
        _, f = self.resnet(images)

        h = f[3]  # bs 2048 w/32 h/32
        g = (self.unpool1(h))  # bs 2048 w/16 h/16
        c = self.conv1(torch.cat((g, f[2]), 1))
        c = self.bn1(c)
        c = self.relu(c)

        h = self.conv2(c)  # bs 128 w/16 h/16
        h = self.bn2(h)
        h = self.relu(h)
        g = self.unpool2(h)  # bs 128 w/8 h/8
        c = self.conv3(torch.cat((g, f[1]), 1))
        c = self.bn3(c)
        c = self.relu(c)

        h = self.conv4(c)  # bs 64 w/8 h/8
        h = self.bn4(h)
        h = self.relu(h)
        g = self.unpool3(h)  # bs 64 w/4 h/4
        c = self.conv5(torch.cat((g, f[0]), 1))
        c = self.bn5(c)
        c = self.relu(c)

        h = self.conv6(c)  # bs 32 w/4 h/4
        h = self.bn6(h)
        h = self.relu(h)
        g = self.conv7(h)  # bs 32 w/4 h/4
        g = self.bn7(g)
        g = self.relu(g)

        score = self.conv8(g)  # bs 1 w/4 h/4
        score = self.sigmoid(score)
        geo_map = self.conv9(g)
        geo_map = self.sigmoid(geo_map) * im_size
        angle_map = self.conv10(g)
        angle_map = self.sigmoid(angle_map)
        angle_map = (angle_map - 0.5) * math.pi / 2

        geo = torch.cat((geo_map, angle_map), 1)  # bs 5 w/4 w/4
        return score, geo


if __name__ == "__main__":
    args = parse_args()
    model = EastModel(args).to(device)
    summary(model, (3, 320, 320))
