import torch
from .pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def get_encoder2d(input_channels=1, encoder='pvt_v2_b2', pretrain=False, freeze_bb=False, base_ptdir='.'):
        # backbone network initialization with pretrained weight
        path = ""
        if encoder == 'pvt_v2_b0':
            backbone = pvt_v2_b0()
            path = f'{base_ptdir}/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            backbone = pvt_v2_b1()
            path = f'{base_ptdir}/pvt/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            backbone = pvt_v2_b2()
            path = f'{base_ptdir}/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            backbone = pvt_v2_b3()
            path = f'{base_ptdir}/pvt/pvt_v2_b3.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            backbone = pvt_v2_b4()
            path = f'{base_ptdir}/pvt/pvt_v2_b4.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            backbone = pvt_v2_b5() 
            path = f'{base_ptdir}/pvt/pvt_v2_b5.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'resnet18':
            backbone = resnet18(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet34':
            backbone = resnet34(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet50':
            backbone = resnet50(pretrained=pretrain)
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            backbone = resnet101(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            backbone = resnet152(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            backbone = pvt_v2_b2()  
            path = f'{base_ptdir}/pretrained_pth/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]

        if 'resnet' in encoder and input_channels!=3:
            print(f'Changing the first layer of {encoder} to accept {input_channels} channels')
            original_first_conv = backbone.conv1
            new_first_conv = torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=original_first_conv.out_channels, # type: ignore
                kernel_size=original_first_conv.kernel_size, # type: ignore
                stride=original_first_conv.stride, # type: ignore
                padding=original_first_conv.padding, # type: ignore
                bias=original_first_conv.bias is not None # type: ignore
            )
            torch.nn.init.kaiming_normal_(new_first_conv.weight, mode='fan_out', nonlinearity='relu')
            backbone.conv1 = new_first_conv
            if freeze_bb:
                for name, param in backbone.named_parameters():
                    if "conv1" not in name:  # Only allow conv1 to be trainable
                        param.requires_grad = False
        elif 'resnet' in encoder and freeze_bb:
                for name, param in backbone.named_parameters():
                    param.requires_grad = False

        if pretrain and base_ptdir and 'pvt_v2' in encoder:
            print(f'Loading pretrained weights from {path}')
            save_model = torch.load(path)
            model_dict = backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            backbone.load_state_dict(model_dict)
            if freeze_bb:
                for name, param in backbone.named_parameters():
                    param.requires_grad = False
        elif 'pvt_v2' in encoder:
            print('No pretrained weights loaded! ...')

        return backbone, channels
