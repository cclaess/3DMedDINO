from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU()
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

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, block_inplanes, n_input_channels=1, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0, num_classes=0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels, self.in_planes,
            kernel_size=(7, 7, conv1_t_size),
            stride=(2, 2, conv1_t_stride),
            padding=(3, 3, conv1_t_size // 2),
            bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if num_classes > 0 else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _downsample_basic_block(x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward_features_unet(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        if not self.no_max_pool:
            x1 = self.maxpool(x0)
        else:
            x1 = deepcopy(x0)

        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

    def forward_head(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x


def resnet10(**kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    return model


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    return model


def resnet200(**kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model


def upconv3x3x3(in_planes, out_planes, stride=1, output_padding=0):
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                              output_padding=output_padding, bias=False)


class DeconvBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.upconv1 = upconv3x3x3(in_planes, planes, stride, output_padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU()
        self.upconv2 = upconv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        out = self.upconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.upconv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNetDecoder(nn.Module):

    def __init__(self, block_inplanes, conv1_t_size=7, conv1_t_stride=1, num_classes=1000):
        super().__init__()

        self.bridge = conv1x1x1(block_inplanes[3], block_inplanes[3])
        self.uplayer4 = DeconvBlock(block_inplanes[3] * 2, block_inplanes[2], stride=2)
        self.uplayer3 = DeconvBlock(block_inplanes[2] * 2, block_inplanes[1], stride=2)
        self.uplayer2 = DeconvBlock(block_inplanes[1] * 2, block_inplanes[0], stride=2)
        self.uplayer1 = DeconvBlock(block_inplanes[0] * 2, block_inplanes[0], stride=2)

        self.upconv1 = nn.ConvTranspose3d(block_inplanes[0], num_classes, kernel_size=(7, 7, conv1_t_size),
                                          stride=(2, 2, conv1_t_stride), padding=(3, 3, conv1_t_size // 2),
                                          output_padding=(1, 1, conv1_t_stride // 2), bias=False)
        self.bn2 = nn.BatchNorm3d(num_classes)

    def forward(self, x1, x2, x3, x4):

        out = self.bridge(x4)
        out = self.uplayer4(out, x4)
        out = self.uplayer3(out, x3)
        out = self.uplayer2(out, x2)
        out = self.uplayer1(out, x1)

        out = self.upconv1(out)
        out = self.bn2(out)

        return out


def resnet10_decoder(**kwargs):
    model = UNetDecoder([i * 1 for i in get_inplanes()], **kwargs)
    return model


def resnet18_decoder(**kwargs):
    model = UNetDecoder([i * 1 for i in get_inplanes()], **kwargs)
    return model


def resnet34_decoder(**kwargs):
    model = UNetDecoder([i * 1 for i in get_inplanes()], **kwargs)
    return model


def resnet50_decoder(**kwargs):
    model = UNetDecoder([i * 4 for i in get_inplanes()], **kwargs)
    return model


def resnet101_decoder(**kwargs):
    model = UNetDecoder([i * 4 for i in get_inplanes()], **kwargs)
    return model


def resnet152_decoder(**kwargs):
    model = UNetDecoder([i * 4 for i in get_inplanes()], **kwargs)
    return model


def resnet200_decoder(**kwargs):
    model = UNetDecoder([i * 4 for i in get_inplanes()], **kwargs)
    return model

