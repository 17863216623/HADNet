# ffr_module.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class FrequencyFeatureRectifier(nn.Module):
    def __init__(self, channels: int, bands=(0.15,0.45,0.75), width=0.20):
        super().__init__()
        self.register_buffer("bands", torch.tensor(bands, dtype=torch.float32))
        self.width = float(width)
        nb = len(bands)
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels//4), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(4, channels//4), nb, 1, bias=True),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _radial(self, H, W, device):
        yy = torch.linspace(-1,1,steps=H,device=device).view(H,1).expand(H,W//2+1)
        xx = torch.linspace(0,1,steps=W//2+1,device=device).view(1,W//2+1).expand(H,W//2+1)
        r = torch.sqrt(xx*xx + yy*yy)
        return (r - r.min())/(r.max()-r.min()+1e-12)

    def forward(self, x):
        B,C,H,W = x.shape
        Xf = torch.fft.rfft2(x, dim=(-2,-1))
        r = self._radial(H,W,x.device) # H x W//2+1
        masks=[]
        for c in self.bands:
            band = torch.exp(-((r-c)**2)/(2*(self.width**2)))
            masks.append(band)
        M = torch.stack(masks, dim=0).unsqueeze(0).unsqueeze(2)  # 1xKx1xHxW2
        w = self.adapter(x).view(B,-1,1,1,1)                      # BxKx1x1x1
        Mm = (w*M).sum(dim=1, keepdim=True)                       # Bx1x1xHxW2
        Mm = Mm.squeeze(1)                                         # Bx1xHxW2
        Mm = Mm.expand(B,C,H,W//2+1)                              # BxCxHxW2
        Yf = Xf * Mm
        y  = torch.fft.irfft2(Yf, s=(H,W), dim=(-2,-1))
        return x + self.scale * y
