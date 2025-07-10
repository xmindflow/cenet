import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from .decoders import Decoder
from .modules.unet import (
    UnetResBlock,
    UnetOutBlock,
)



class Net(nn.Module):
    def __init__(self, num_classes=1, input_channels=1,
                 scale_factors=[0.6, 0.3], num_heads=[2,2,2],
                 encoder='pvt_v2_b2', pretrain=False, skip_mode="cat", base_ptdir='.'):
        super().__init__()

        # conv block to convert single channel to 3 channels
        if input_channels == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=1),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Identity()

        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = f'{base_ptdir}/pretrained_pth/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = f'{base_ptdir}/pretrained_pth/pvt/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = f'{base_ptdir}/pretrained_pth/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = f'{base_ptdir}/pretrained_pth/pvt/pvt_v2_b3.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = f'{base_ptdir}/pretrained_pth/pvt/pvt_v2_b4.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5() 
            path = f'{base_ptdir}/pretrained_pth/pvt/pvt_v2_b5.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()  
            path = f'{base_ptdir}/pretrained_pth/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2' in encoder:
            try:
                save_model = torch.load(path)
                model_dict = self.backbone.state_dict()
                state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                self.backbone.load_state_dict(model_dict)
                print(f'Loaded pretrained weights from {path}')
            except Exception as e:
                print(f'Error loading pretrained weights from {path}: {e}')
        
        # decoder initialization
        self.decoder = Decoder(channels=channels,
                                scale_factors=scale_factors, 
                                skip_mode=skip_mode,
                                num_heads=num_heads)
        
        # self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        # self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        # self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        # self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        
        norm_name = 'batch'
        act_name = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        rb_block = partial(UnetResBlock, spatial_dims=2, stride=1, kernel_size=3, norm_name=norm_name, act_name=act_name)
        fine_channels = [channels[-1]//2, channels[-1]]

        self.enc = nn.Sequential(rb_block(in_channels=input_channels, out_channels=fine_channels[0], dropout=0), nn.MaxPool2d(kernel_size=2, stride=2))
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                                rb_block(in_channels=fine_channels[1], out_channels=fine_channels[0], dropout=0))
        self.rb  = rb_block(in_channels=fine_channels[1], out_channels=fine_channels[1], dropout=0)
        self.out = UnetOutBlock(spatial_dims=2, in_channels=fine_channels[1], out_channels=num_classes, kernel_size=1)

        
    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        y = self.conv(x) # convert to 3 channels
        
        # encoder
        x1, x2, x3, x4 = self.backbone(y)

        # decoder
        deco = self.decoder(x4, [x3, x2, x1]) # [B, 64, 56, 56]

        enc = self.enc(x)                   # encode to [B, 32, 112, 112]
        dec = self.up(deco)                 # up to [B, 32, 112, 112]
        
        z = self.out(self.rb(torch.cat([dec, enc], 1)))
        p1 = F.interpolate(z, scale_factor=2, mode='bilinear')

        return p1
