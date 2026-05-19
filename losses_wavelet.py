# losses_wavelet.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F

# ---------- 2×2 Haar 小波 ----------
def _haar_filters(device, dtype):
    h0 = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / math.sqrt(2.0)
    h1 = torch.tensor([1.0,-1.0], device=device, dtype=dtype) / math.sqrt(2.0)
    LL = torch.outer(h0, h0)
    LH = torch.outer(h0, h1)
    HL = torch.outer(h1, h0)
    HH = torch.outer(h1, h1)
    k = torch.stack([LL, LH, HL, HH], dim=0)  # 4×2×2
    return k

def _build_group_kernels(C, device, dtype):
    base = _haar_filters(device, dtype)  # 4×2×2
    weight = torch.zeros((4*C, 1, 2, 2), device=device, dtype=dtype)
    for c in range(C):
        weight[c*4:(c+1)*4, 0, :, :] = base
    return weight

@torch.no_grad()
def dwt2_haar(x):
    B,C,H,W = x.shape
    weight = _build_group_kernels(C, x.device, x.dtype)
    y = F.conv2d(x, weight, stride=2, padding=0, groups=C)  # [B,4C,H/2,W/2]
    y = y.view(B, C, 4, H//2, W//2)
    LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
    return LL, (LH, HL, HH)

@torch.no_grad()
def dwt2_haar_levels(x, levels=3):
    out, cur = [], x
    for _ in range(levels):
        LL, details = dwt2_haar(cur)
        out.append((LL, details))
        cur = LL
    return out

# ---------- 小波细节损失 WDL ----------
def wavelet_detail_loss(prob, gt, levels=3, eps=1e-6):
    plist = dwt2_haar_levels(prob, levels=levels)
    glist = dwt2_haar_levels(gt,   levels=levels)
    losses = []
    for (PLL, (PLH, PHL, PHH)), (GLL, (GLH, GHL, GHH)) in zip(plist, glist):
        Pmag = torch.abs(PLH) + torch.abs(PHL) + torch.abs(PHH)
        Gmag = torch.abs(GLH) + torch.abs(GHL) + torch.abs(GHH)
        losses.append(F.l1_loss(Pmag, Gmag, reduction="mean"))
    return sum(losses) / max(1, len(losses))

# ---------- 小波边缘 Dice WED ----------
def wavelet_edge_dice(prob, gt, levels=3, eps=1e-6):
    plist = dwt2_haar_levels(prob, levels=levels)
    glist = dwt2_haar_levels(gt,   levels=levels)
    losses = []
    for (PLL, (PLH, PHL, PHH)), (GLL, (GLH, GHL, GHH)) in zip(plist, glist):
        Pmag = torch.abs(PLH) + torch.abs(PHL) + torch.abs(PHH)
        Gmag = torch.abs(GLH) + torch.abs(GHL) + torch.abs(GHH)
        Pn = (Pmag - Pmag.amin((2,3),True)) / (Pmag.amax((2,3),True)-Pmag.amin((2,3),True)+eps)
        Gn = (Gmag - Gmag.amin((2,3),True)) / (Gmag.amax((2,3),True)-Gmag.amin((2,3),True)+eps)
        inter = (Pn*Gn).sum(dim=(1,2,3))
        den   = Pn.sum(dim=(1,2,3)) + Gn.sum(dim=(1,2,3)) + eps
        dice  = (2*inter + eps)/den
        losses.append(1 - dice.mean())
    return sum(losses) / max(1, len(losses))

# ---------- TV-L1 去噪 ----------
def tv_l1_loss(prob, weight=1.0):
    dx = torch.abs(prob[:,:,:,1:] - prob[:,:,:,:-1]).mean()
    dy = torch.abs(prob[:,:,1:,:] - prob[:,:,:-1,:]).mean()
    return weight * (dx + dy)

# ---------- 边界焦点 BCE ----------
def _morph_band(gt):
    kernel = torch.ones((1,1,3,3), device=gt.device)
    s = F.conv2d(gt, kernel, padding=1)
    erode  = (s == 9).float()
    dilate = (s >= 1).float()
    band = (dilate - erode).clamp(0,1)
    return band

def boundary_focal_bce_loss(logits, gt, lam=1.0, gamma=2.0):
    p  = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    band = _morph_band(gt)
    u = 4.0 * p * (1.0 - p)           # 不确定性
    w = 1.0 + lam * band * (u ** gamma)
    return (w * bce).mean()
