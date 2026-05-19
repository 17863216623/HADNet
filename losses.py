# losses.py
# -*- coding: utf-8 -*-
import torch, math
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogits(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def forward(self, logits, target): return self.loss(logits, target)

# ---- IoU / Dice on probabilities ----
def iou_loss_from_probs(prob, target, eps=1e-6):
    inter = (prob*target).sum(dim=(1,2,3))
    union = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter + eps
    iou = (inter + eps)/union
    return (1 - iou).mean()

def dice_loss_from_probs(prob, target, eps=1e-6):
    inter = (prob*target).sum(dim=(1,2,3))
    den   = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    dice = (2*inter + eps)/den
    return (1 - dice).mean()

# ---- SSIM (fast approx) ----
def _ssim_local(x, y, c1=0.01**2, c2=0.03**2, k=7):
    pad = k//2
    mu_x = F.avg_pool2d(x, k, 1, pad)
    mu_y = F.avg_pool2d(y, k, 1, pad)
    sigma_x = F.avg_pool2d(x*x, k, 1, pad) - mu_x*mu_x
    sigma_y = F.avg_pool2d(y*y, k, 1, pad) - mu_y*mu_y
    sigma_xy= F.avg_pool2d(x*y, k, 1, pad) - mu_x*mu_y
    num = (2*mu_x*mu_y + c1) * (2*sigma_xy + c2)
    den = (mu_x*mu_x + mu_y*mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = num / (den + 1e-12)
    return ssim

def ssim_loss_from_probs(prob, target):
    return (1 - _ssim_local(prob, target).mean())

# ---- Fourier magnitude loss ----
def fourier_mag_loss(prob, target, log_mag=True, highfreq_boost=0.2):
    # prob/target: Bx1xHxW in [0,1]
    Pf = torch.fft.rfft2(prob, dim=(-2,-1)); Tf = torch.fft.rfft2(target, dim=(-2,-1))
    Pm, Tm = torch.abs(Pf), torch.abs(Tf)
    if log_mag:
        Pm = torch.log1p(Pm); Tm = torch.log1p(Tm)
    H = prob.shape[-2]; W2 = Pm.shape[-1]
    yy = torch.linspace(-1,1,steps=H,device=prob.device).view(H,1).expand(H,W2)
    xx = torch.linspace(0,1,steps=W2,device=prob.device).view(1,W2).expand(H,W2)
    r  = torch.sqrt(xx*xx + yy*yy); r = (r - r.min())/(r.max()-r.min()+1e-12)
    w  = 1 + highfreq_boost * r
    return ((w * (Pm - Tm).abs()).mean())

# ---- Edge loss (Sobel) ----
def _sobel_edge(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    g  = torch.sqrt(gx*gx + gy*gy + 1e-12)
    g  = g / (g.amax(dim=(2,3),keepdim=True)+1e-12)
    return g

def edge_dice_bce_loss(prob, target):
    pe, ge = _sobel_edge(prob), _sobel_edge(target)
    bce = F.binary_cross_entropy(pe, ge)
    inter = (pe*ge).sum(dim=(1,2,3))
    den   = pe.sum(dim=(1,2,3)) + ge.sum(dim=(1,2,3)) + 1e-6
    dice = (2*inter + 1e-6)/den
    return bce + (1 - dice).mean()

# ---- Boundary IoU ----
def boundary_iou_loss_from_probs(prob, target, eps=1e-6):
    pe, ge = _sobel_edge(prob), _sobel_edge(target)
    inter = (pe*ge).sum(dim=(1,2,3))
    union = pe.sum(dim=(1,2,3)) + ge.sum(dim=(1,2,3)) - inter + eps
    biou  = (inter + eps)/union
    return (1 - biou).mean()

# ---- Uncertainty-Weighted Boundary (UWB) ----
def _morph_band(gt):
    # gt: Bx1xHxW in {0,1}
    kernel = torch.ones((1,1,3,3), device=gt.device)
    s = F.conv2d(gt, kernel, padding=1)
    erode  = (s == 9).float()
    dilate = (s >= 1).float()
    band = (dilate - erode).clamp(0,1)
    return band

def uncertainty_weighted_boundary_loss(prob, gt, lam=2.0):
    # prob,gt in [0,1]
    band = _morph_band(gt)
    p = prob.clamp(1e-6,1-1e-6)
    unc = -(p*torch.log(p) + (1-p)*torch.log(1-p))
    unc = (unc - unc.amin((2,3),True)) / (unc.amax((2,3),True)-unc.amin((2,3),True)+1e-6)
    w = 1.0 + lam * band * unc
    l1 = (w * (prob - gt).abs()).mean()
    biou = boundary_iou_loss_from_probs(prob, gt)
    return l1 + biou

# ---- Edge GT for ERM ----
def make_edge_target(gt, thresh=0.1):
    # gt: Bx1xHxW in {0,1}
    e = _sobel_edge(gt)   # 0..1
    return (e > thresh).float()
