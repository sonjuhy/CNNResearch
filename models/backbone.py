import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

from torch.ao.nn.quantized import FloatFunctional

class ConvNeXtBlockNPU(nn.Module):
    """ NPU-friendly ConvNeXt block with BatchNorm instead of LayerNorm and SiLU instead of GELU """
    def __init__(self, dim, layer_scale_init=1e-6, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.Hardswish(inplace=True)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim, 1, 1))
        self.drop_path = StochasticDepth(drop_path, mode='row') if drop_path > 0 else nn.Identity()
        self.gamma_mul = FloatFunctional()
        self.skip_add = FloatFunctional()

    def fold_gamma(self):
        """ 양자화 전에 LayerScale을 pwconv2로 흡수하여 곱셈 연산을 제거 """
        with torch.no_grad():
            g = self.gamma.reshape(-1)
            self.pwconv2.weight.mul_(g.view(-1, 1, 1, 1))
            if self.pwconv2.bias is not None:
                self.pwconv2.bias.mul_(g)
            self.gamma.fill_(1.0)
        self.gamma_mul = None

    def forward(self, x):
        residual = x
        x = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
        if self.gamma_mul is not None:
            x = self.gamma_mul.mul(x, self.gamma)
        return self.skip_add.add(residual, self.drop_path(x))

class ConvNeXtBackbone(nn.Module):
    def __init__(self, scale='nano'):
        super().__init__()
        configs = {
            'nano':   {'depths': [2, 2, 6, 2],  'dims': [48, 96, 192, 384]},
            'medium': {'depths': [3, 3, 9, 3],  'dims': [96, 192, 384, 768]},
            'xlarge': {'depths': [3, 3, 27, 3], 'dims': [256, 512, 1024, 2048]}
        }
        cfg = configs[scale]
        self.dims = cfg['dims']
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg['dims'][0], kernel_size=4, stride=4),
            nn.BatchNorm2d(cfg['dims'][0])
        )
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(cfg['depths']))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlockNPU(dim=cfg['dims'][i], drop_path=dp_rates[cur+j]) for j in range(cfg['depths'][i])]
            )
            self.stages.append(stage)
            cur += cfg['depths'][i]
            
        self.downsamples = nn.ModuleList()
        for i in range(3):
            self.downsamples.append(nn.Sequential(
                nn.Conv2d(cfg['dims'][i], cfg['dims'][i+1], kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(cfg['dims'][i+1])
            ))

    def forward(self, x):
        features = []
        x = self.stem(x)
        x = self.stages[0](x) # Stage 1 (Stride 4)
        
        for i in range(3):
            x = self.downsamples[i](x)
            x = self.stages[i+1](x)
            features.append(x) # Stride 8, 16, 32
            
        return features
