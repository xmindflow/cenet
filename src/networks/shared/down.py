
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from timm.layers import trunc_normal_tf_
from timm.models import named_apply
from contextlib import redirect_stderr
from .utils import _init_weights, act_layer
from .unet import UnetResBlock 


__all__ = ['DownConv', 'DownRb']


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer(activation, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.down(x)


class DownRb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super().__init__()

        norm_name = 'batch'
        act_name = (activation, {"inplace": True, "negative_slope": 0.01})
        rb_block = partial(UnetResBlock, spatial_dims=2, norm_name=norm_name, act_name=act_name, dropout=0)

        self.down = nn.Sequential(
            rb_block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        return self.down(x)
