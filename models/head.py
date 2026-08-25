import torch
import torch.nn as nn
from .neck import ConvModule
from torch.ao.quantization import DeQuantStub

class DecoupledHead(nn.Module):
    def __init__(self, num_classes=80, in_channels=128, hidden_channels=256):
        super().__init__()
        self.cls_convs = nn.Sequential(
            ConvModule(in_channels, hidden_channels, kernel_size=3, padding=1),
            ConvModule(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )
        self.cls_preds = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        
        self.reg_convs = nn.Sequential(
            ConvModule(in_channels, hidden_channels, kernel_size=3, padding=1),
            ConvModule(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )
        self.reg_preds = nn.Conv2d(hidden_channels, 4, kernel_size=1) 
        self.obj_preds = nn.Conv2d(hidden_channels, 1, kernel_size=1) 
        
    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        return self.cls_preds(cls_feat), self.reg_preds(reg_feat), self.obj_preds(reg_feat)

class SOTAHead(nn.Module):
    def __init__(self, num_classes=80, in_channels=128, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        
        self.heads = nn.ModuleList([
            DecoupledHead(num_classes, in_channels, in_channels) for _ in strides
        ])
        
        for h in self.heads[1:]:
            for (n0, m0), (n1, m1) in zip(self.heads[0].named_modules(), h.named_modules()):
                if isinstance(m0, nn.Conv2d):
                    m1.weight = m0.weight
                    if m0.bias is not None:
                        m1.bias = m0.bias
                        
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in strides])
        self.dequant = DeQuantStub()
        
        # 기본 640x640 해상도용 Grid Point 사전 계산 (ONNX 동적 연산 노드 방지용)
        pts, sts = [], []
        for s in strides:
            hw = 640 // s
            pts.append(self.make_grid_points(hw, hw, s, torch.device('cpu'), torch.float32))
            sts.append(torch.full((hw * hw,), float(s)))
        self.register_buffer('default_points', torch.cat(pts), persistent=False)
        self.register_buffer('default_strides', torch.cat(sts), persistent=False)
        
    @staticmethod
    def make_grid_points(h: int, w: int, stride: int, device, dtype):
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing='ij',
        )
        points = torch.stack([(xs + 0.5) * stride, (ys + 0.5) * stride], dim=-1)
        return points.reshape(-1, 2)
        
    def forward(self, features):
        cls_list, reg_list, obj_list = [], [], []
        
        for feat, head, stride, scale in zip(features, self.heads, self.strides, self.scales):
            cls_out, reg_out, obj_out = head(feat)
            
            cls_out = self.dequant(cls_out)
            reg_out = self.dequant(reg_out)
            obj_out = self.dequant(obj_out)
            
            cls_list.append(cls_out.flatten(2).permute(0, 2, 1))
            obj_list.append(obj_out.flatten(2).permute(0, 2, 1))
            
            reg = ((reg_out.flatten(2).permute(0, 2, 1) * scale).clamp(max=8.0)).exp() * stride
            reg_list.append(reg)
            
        cls = torch.cat(cls_list, dim=1)
        reg = torch.cat(reg_list, dim=1)
        obj = torch.cat(obj_list, dim=1)
        
        # ONNX 등 정적 그래프 최적화를 위해, 640x640(8400 grid)일 경우 캐시된 버퍼 사용
        if cls.shape[1] == self.default_points.shape[0]:
            points = self.default_points.to(cls.device)
            strides_out = self.default_strides.to(cls.device)
        else:
            # 해상도가 다를 경우 동적 생성
            pts, sts = [], []
            for feat, stride in zip(features, self.strides):
                H, W = feat.shape[2], feat.shape[3]
                pts.append(self.make_grid_points(H, W, stride, cls.device, cls.dtype))
                sts.append(torch.full((H * W,), stride, device=cls.device, dtype=cls.dtype))
            points = torch.cat(pts, dim=0)
            strides_out = torch.cat(sts, dim=0)
            
        return {
            'cls':     cls,
            'reg':     reg,
            'obj':     obj,
            'points':  points,
            'strides': strides_out,
        }
