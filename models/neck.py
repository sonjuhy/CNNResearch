import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardswish(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWSepConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = ConvModule(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = ConvModule(dim, dim, kernel_size=1, padding=0)

    def forward(self, x):
        return self.pw(self.dw(x))

from torch.ao.nn.quantized import FloatFunctional

class BiFPNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.td_conv_p4 = DWSepConv(channels)
        self.td_conv_p3 = DWSepConv(channels)
        self.bu_conv_p4 = DWSepConv(channels)
        self.bu_conv_p5 = DWSepConv(channels)
        
        self.add = nn.ModuleList([FloatFunctional() for _ in range(5)])
        
    def forward(self, p3, p4, p5):
        # Top-down
        p4_td = self.td_conv_p4(self.add[0].add(p4, F.interpolate(p5, scale_factor=2.0, mode='nearest')))
        p3_out = self.td_conv_p3(self.add[1].add(p3, F.interpolate(p4_td, scale_factor=2.0, mode='nearest')))
        
        # Bottom-up
        p4_out = self.bu_conv_p4(self.add[2].add(self.add[3].add(p4, p4_td), F.max_pool2d(p3_out, kernel_size=2, stride=2)))
        p5_out = self.bu_conv_p5(self.add[4].add(p5, F.max_pool2d(p4_out, kernel_size=2, stride=2)))
        
        return p3_out, p4_out, p5_out

class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_blocks):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            ConvModule(in_channels, out_channels, kernel_size=1) 
            for in_channels in in_channels_list
        ])
        self.blocks = nn.ModuleList([BiFPNBlock(out_channels) for _ in range(num_blocks)])
        
    def forward(self, features):
        p3, p4, p5 = [lat_conv(feat) for lat_conv, feat in zip(self.lateral_convs, features)]
        for block in self.blocks:
            p3, p4, p5 = block(p3, p4, p5)
        return [p3, p4, p5]
