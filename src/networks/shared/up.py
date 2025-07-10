from functools import partial
import torch.nn as nn
from timm.models import named_apply
from monai.networks.layers.factories import Act, Norm
from .utils import get_conv_layer, _init_weights, act_layer, channel_shuffle
from .unet import UnetResBlock, UnetUpBlock


__all__ = ['UpRb', 'UpConv', 'UpTConv', 'EUCB', 'UnetUpBlock']


class UpRb(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        norm_name = 'batch'
        act_name = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        rb_block = partial(UnetResBlock, spatial_dims=2, norm_name=norm_name, act_name=act_name, dropout=0)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            rb_block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1),
        )
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        return self.up(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, scale_factor=2, activation='relu'):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer(activation, inplace=True)
        )
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        return self.up(x)

   
class UpTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, activation='prelu', scale_factor=2):
        super().__init__()
        assert scale_factor==2, f"Invalid scale_factor: {scale_factor}, only support 2"
        self.up = get_conv_layer(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=0,
            bias=False,
            conv_only=True,
            is_transposed=True,
            
            act=activation,
            norm="batch" # Norm.INSTANCE
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        return self.up(x)


class EUCB(nn.Module):
    '''Efficient up-convolution block (EUCB)'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, scale_factor=2, activation='relu'):
        super(EUCB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
	        nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
