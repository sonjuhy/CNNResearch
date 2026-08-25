import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
from .backbone import ConvNeXtBackbone
from .neck import BiFPN
from .head import SOTAHead

class SOTADetector(nn.Module):
    def __init__(self, num_classes=80, scale='nano', quantizable=False):
        super(SOTADetector, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.quantizable = quantizable
        
        self.quant = QuantStub() if quantizable else nn.Identity()
        self.dequant = DeQuantStub() if quantizable else nn.Identity()
        
        self.backbone = ConvNeXtBackbone(scale=scale)
        
        neck_configs = {
            'nano':   {'out_channels': 128, 'num_blocks': 2},
            'medium': {'out_channels': 256, 'num_blocks': 4},
            'xlarge': {'out_channels': 512, 'num_blocks': 6}
        }
        cfg = neck_configs[scale]
        self.neck = BiFPN(
            in_channels_list=self.backbone.dims[1:4], 
            out_channels=cfg['out_channels'], 
            num_blocks=cfg['num_blocks']
        )
        
        self.head = SOTAHead(num_classes=num_classes, in_channels=cfg['out_channels'])
        
    def forward(self, x):
        x = self.quant(x)
        features = self.backbone(x)
        features = self.neck(features)
        out = self.head(features)
        
        if self.quantizable:
            out = {k: (self.dequant(v) if torch.is_tensor(v) else v) for k, v in out.items()}
        return out
