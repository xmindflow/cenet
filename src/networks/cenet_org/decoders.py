import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_tf_
from timm.models import named_apply
 
from .modules.cfam import CFAMBlock
from .modules.multihead_diffattn import MultiheadDiffAttn



def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x



#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
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


class DoGEdge(nn.Module):
    def __init__(self,dim: int, scale_factors: list) -> None:
        super().__init__()
        self.scale_factors = scale_factors
        self.w = nn.Parameter(torch.ones(1, dim, 1, 1)*0.5)
    def forward(self, x):
        _, C, H, W = x.shape
        x_1 = F.interpolate(x, scale_factor = self.scale_factors[0], mode = "bilinear")
        x_2 = F.interpolate(x, scale_factor = self.scale_factors[1], mode = "bilinear")
        x_1 = F.interpolate(x_1, size=(H, W), mode = 'bilinear')
        x_2 = F.interpolate(x_2, size=(H, W), mode = 'bilinear')  
        x_hf = torch.abs(x_1-x_2)
        x = x + self.w*x_hf
        return x


class SkipEnhancer(nn.Module):
    def __init__(self, dim, scale_factors, num_heads, input_size, mode='add'):
        super().__init__()
        self.input_size = input_size
        self.mode = mode.lower()
        _dim = dim*2 if self.mode=='cat' else dim
        self.boundary = DoGEdge(dim=_dim, scale_factors=scale_factors)
        self.diffattn = MultiheadDiffAttn(embed_dim=_dim, depth=1, num_heads=num_heads)
        self.proj = nn.Conv2d(in_channels=_dim, out_channels=dim, kernel_size=1, stride=1) if self.mode=='cat' else nn.Identity()

    def forward(self, skip, dec):
        y = dec+skip if self.mode=='add' else torch.cat([dec, skip], dim=1)
        y = self.boundary(y)
        y_token = y.view(y.shape[0], -1, y.shape[1])
        diff = self.diffattn(y_token) * y_token
        diff = diff.view(diff.shape[0], diff.shape[2], diff.shape[1]//self.input_size, diff.shape[1]//self.input_size)
        z = y + diff
        return self.proj(z) + skip


class Decoder(nn.Module):
    def __init__(self, 
                 channels=[512,320,128,64], 
                 input_size=[14,28,56,112] , 
                 scale_factors=[0.8,0.4], 
                 skip_mode='add', # 'add' or 'cat',
                 num_heads=[2,2,2]):

        super().__init__()
        eucb_ks = 3 # kernel size for eucb
        self.input_size = input_size

        decblock = partial(CFAMBlock, ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value=1e-6,
                               attn_channel_split=[1,3,4],
                               attn_act_type="SiLU",
                               attn_force_fp32=False)
        skipblock = partial(SkipEnhancer, scale_factors=scale_factors, mode=skip_mode)
        
        self.dec4 = decblock(embed_dims= channels[0])
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.skip_enhancer3 = skipblock(dim=channels[1], num_heads=num_heads[0], input_size=input_size[0])
        self.dec3 = decblock(embed_dims= channels[1])
        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.skip_enhancer2 = skipblock(dim=channels[2], num_heads=num_heads[1], input_size=input_size[1])
        self.dec2 = decblock(embed_dims= channels[2])
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.skip_enhancer1 = skipblock(dim=channels[3], num_heads=num_heads[2], input_size=input_size[2])
        self.dec1 = decblock(embed_dims= channels[3])
        

    def forward(self, x, skips):
        d4 = self.dec4(x)
        d3 = self.eucb3(d4)
        
        skips_3 = self.skip_enhancer3(skips[0], d3)
        d3 = self.dec3(d3 + skips_3) 
        d2 = self.eucb2(d3)
        
        skips_2 = self.skip_enhancer2(skips[1], d2)
        d2 = self.dec2(d2 + skips_2)
        d1 = self.eucb1(d2)
        
        skips_1 = self.skip_enhancer1(skips[2], d1)
        d1 = self.dec1(d1 + skips_1)
        
        return d1
