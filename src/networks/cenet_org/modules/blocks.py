import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConvBN(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, rate=1, depth_activation=False, epsilon=1e-3):
        super(SepConvBN, self).__init__()

        # Calculate padding
        # if stride == 1:
        #     self.padding = kernel_size // 2
        # else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        self.padding = (kernel_size_effective - 1) // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=rate,
            groups=in_channels,
            bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels, eps=epsilon)

        self.pointwise = nn.Conv2d(
            in_channels,
            filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.pointwise_bn = nn.BatchNorm2d(filters, eps=epsilon)
        self.depth_activation = depth_activation

    def forward(self, x):
        if not self.depth_activation:
            x = F.relu(x, inplace=True)

        x = self.depthwise(x)
        x = self.depthwise_bn(x)

        if self.depth_activation:
            x = F.relu(x, inplace=True)

        x = self.pointwise(x)
        x = self.pointwise_bn(x)

        if self.depth_activation:
            x = F.relu(x, inplace=True)

        return x