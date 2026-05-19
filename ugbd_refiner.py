# ugbd_refiner.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def wavelet_like_edges(x: torch.Tensor) -> torch.Tensor:
    # x: Bx3xHxW in [0,1] or logits features (we'll use input image)
    # 3个固定高通核，近似小波高频（避免尺寸变化）
    kx = torch.tensor([[-1,-1,-1],[ 2, 2, 2],[-1,-1,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = kx.transpose(2,3).contiguous()
    kd = torch.tensor([[ 2,-1,-1],[-1, 2,-1],[-1,-1, 2]], dtype=x.dtype, device=x.device).view(1,1,3,3)

    def hp(conv_k):
        k = conv_k.repeat(x.size(1), 1, 1, 1)  # group conv
        y = F.conv2d(x, k, bias=None, stride=1, padding=1, groups=x.size(1))
        return y

    ex = hp(kx); ey = hp(ky); ed = hp(kd)
    e = torch.sqrt(ex.pow(2) + ey.pow(2) + ed.pow(2) + 1e-6)
    e = e.mean(1, keepdim=True)  # Bx1xHxW
    e = e / (e.amax(dim=(2,3), keepdim=True) + 1e-6)
    return e

class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super().__init__()
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2*hid_ch, 3, padding=1)
        self.conv_h  = nn.Conv2d(in_ch + hid_ch, hid_ch, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.size(0), self.conv_h.out_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        cat = torch.cat([x, h], dim=1)
        zr = torch.sigmoid(self.conv_zr(cat))
        z, r = torch.chunk(zr, 2, dim=1)
        cat2 = torch.cat([x, h * r], dim=1)
        h_tilde = torch.tanh(self.conv_h(cat2))
        h = (1 - z) * h + z * h_tilde
        return h

class UGBDRefiner(nn.Module):
    """
    Uncertainty-Guided Boundary Diffusion refiner.
    输入：image(Bx3xH xW), logits(Bx1xH xW)
    输出：refined logits（在logits上做残差）
    """
    def __init__(self, mid_ch=64, steps=3):
        super().__init__()
        self.steps = steps
        self.in_conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fuse    = nn.Conv2d(8+3, mid_ch, 3, padding=1)  # [img_feat(8), prob(1), edge(1), uncert(1)]
        self.gru     = ConvGRUCell(mid_ch, mid_ch)
        self.out     = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1)
        )

    def forward(self, image, logits):
        # prob & uncertainty
        prob = torch.sigmoid(logits)
        uncert = 4.0 * prob * (1.0 - prob)  # [0,1]
        # wavelet-like edge from image
        img_feat0 = F.relu(self.in_conv(image))
        edge = wavelet_like_edges(image)

        x = torch.cat([img_feat0, prob, edge, uncert], dim=1)
        x = self.fuse(x)
        h = None
        for _ in range(self.steps):
            h = self.gru(x, h)
        res = self.out(h)
        return logits + res
