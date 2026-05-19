# sdf_head.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np  # noqa: F401
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# 复用我们之前的小波风格边缘特征（稳定、轻量）
from ugbd_refiner import wavelet_like_edges


def sobel_grad(x: torch.Tensor):
    # x: Bx1xHxW
    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def laplace(x: torch.Tensor):
    k = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1)


def eikonal_loss(sdf: torch.Tensor, target_norm: float = 1.0):
    gx, gy = sobel_grad(sdf)
    g = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return ((g - target_norm) ** 2).mean()


def curvature_loss(sdf: torch.Tensor):
    # 轻量曲率平滑，抑制高曲率毛刺
    curv = laplace(sdf)
    return (curv * curv).mean()


def gt_sdf_from_mask(mask_np):
    """
    mask_np: HxW (0/1)
    inside  -> negative; outside -> positive (标准有符号距离)
    """
    m = (mask_np > 0).astype("uint8")
    if m.sum() == 0:
        # 全背景：SDF=+dist_to_any_fg；这里退化成全正距离
        dist_out = distance_transform_edt(1 - m)
        sdf = dist_out
    elif m.sum() == m.size:
        # 全前景：SDF=-dist_to_any_bg；退化成全负距离
        dist_in = distance_transform_edt(m)
        sdf = -dist_in
    else:
        dist_out = distance_transform_edt(1 - m)
        dist_in = distance_transform_edt(m)
        sdf = dist_out - dist_in
    return sdf


def batch_build_sdf(gt_masks: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    gt_masks: Bx1xHxW torch.float (0/1)
    return:  Bx1xHxW torch.float (SDF, 已按 scale 缩放)
    """
    if not _HAS_SCIPY:
        raise RuntimeError("计算 SDF 需要 scipy，请先: pip install scipy")
    gts = gt_masks.detach().cpu().numpy()
    out = []
    for b in range(gts.shape[0]):
        m = gts[b, 0]
        sdf = gt_sdf_from_mask((m > 0.5).astype("uint8"))
        # 归一化到 roughly [-1,1]，再乘 scale
        s = max(1.0, float(abs(sdf).max()))
        sdf = (sdf / s) * scale
        out.append(torch.from_numpy(sdf).float().unsqueeze(0))
    return torch.stack(out, dim=0).to(gt_masks.device, gt_masks.dtype)


class SDFHead(nn.Module):
    """
    形状先验头：预测 SDF，并以残差形式校正 logits
    输入: image(Bx3xHxW), logits(Bx1xHxW)
    输出: refined_logits, sdf_pred
    """
    def __init__(self, mid_ch: int = 64, fuse_lambda: float = 0.5):
        super().__init__()
        self.fuse_lambda = fuse_lambda
        in_ch = 3 + 1 + 1  # image + prob + edge
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
        )
        self.to_sdf = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, image: torch.Tensor, logits: torch.Tensor):
        prob = torch.sigmoid(logits)
        edge = wavelet_like_edges(image)
        x = torch.cat([image, prob, edge], dim=1)
        feat = self.enc(x)
        sdf = self.to_sdf(feat)  # Bx1xHxW (有符号，未约束)
        refined = logits - self.fuse_lambda * sdf
        return refined, sdf

# sdf_head.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# 复用我们之前的小波风格边缘特征（稳定、轻量）
from ugbd_refiner import wavelet_like_edges

def sobel_grad(x):
    # x: Bx1xHxW
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = kx.transpose(2,3).contiguous()
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy

def laplace(x):
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    return F.conv2d(x, k, padding=1)

def eikonal_loss(sdf, target_norm=1.0):
    gx, gy = sobel_grad(sdf)
    g = torch.sqrt(gx*gx + gy*gy + 1e-6)
    return ((g - target_norm)**2).mean()

def curvature_loss(sdf):
    # 轻量曲率平滑，抑制高曲率毛刺
    curv = laplace(sdf)
    return (curv*curv).mean()

def gt_sdf_from_mask(mask_np):
    """
    mask_np: HxW (0/1)
    inside  -> negative; outside -> positive (标准有符号距离)
    """
    m = (mask_np > 0).astype("uint8")
    if m.sum()==0:
        # 全背景：SDF=+dist_to_any_fg；这里退化成全正距离
        dist_out = distance_transform_edt(1 - m)
        sdf = dist_out
    elif m.sum()==m.size:
        # 全前景：SDF=-dist_to_any_bg；退化成全负距离
        dist_in = distance_transform_edt(m)
        sdf = -dist_in
    else:
        dist_out = distance_transform_edt(1 - m)
        dist_in  = distance_transform_edt(m)
        sdf = dist_out - dist_in
    return sdf

def batch_build_sdf(gt_masks, scale=1.0):
    """
    gt_masks: Bx1xHxW torch.float (0/1)
    return:  Bx1xHxW torch.float (SDF, 已按scale缩放)
    """
    if not _HAS_SCIPY:
        raise RuntimeError("计算 SDF 需要 scipy，请先: pip install scipy")
    gts = gt_masks.detach().cpu().numpy()
    out = []
    for b in range(gts.shape[0]):
        m = gts[b,0]
        sdf = gt_sdf_from_mask((m>0.5).astype("uint8"))
        # 归一化到 roughly [-1,1]，再乘 scale
        s = max(1.0, float(abs(sdf).max()))
        sdf = (sdf / s) * scale
        out.append(torch.from_numpy(sdf).float().unsqueeze(0))
    return torch.stack(out, dim=0).to(gt_masks.device, gt_masks.dtype)

class SDFHead(nn.Module):
    """
    形状先验头：预测 SDF，并以残差形式校正 logits
    输入: image(Bx3xHxW), logits(Bx1xHxW)
    输出: refined_logits, sdf_pred
    """
    def __init__(self, mid_ch=64, fuse_lambda=0.5):
        super().__init__()
        self.fuse_lambda = fuse_lambda
        in_ch = 3 + 1 + 1  # image + prob + edge
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False), nn.GroupNorm(8, mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False), nn.GroupNorm(8, mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False), nn.GroupNorm(8, mid_ch), nn.GELU(),
        )
        self.to_sdf = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, image, logits):
        prob = torch.sigmoid(logits)
        edge = wavelet_like_edges(image)
        x = torch.cat([image, prob, edge], dim=1)
        feat = self.enc(x)
        sdf  = self.to_sdf(feat)              # Bx1xHxW (有符号，未约束)
        refined = logits - self.fuse_lambda * sdf
        return refined, sdf
