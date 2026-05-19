# refine_heads.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from ffr_module import FrequencyFeatureRectifier

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1, gn=False):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)]
    layers += [nn.GroupNorm(8, out_ch)] if gn else [nn.BatchNorm2d(out_ch)]
    layers += [nn.GELU()]
    return nn.Sequential(*layers)

class PPM(nn.Module):
    def __init__(self, in_ch, out_ch, bins=(1,2,3,6)):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(
            nn.AdaptiveAvgPool2d(b), nn.Conv2d(in_ch, out_ch//len(bins), 1, bias=False),
            nn.BatchNorm2d(out_ch//len(bins)), nn.GELU()
        ) for b in bins])
        self.fuse = conv_bn_relu(in_ch + out_ch, out_ch, k=3, s=1, p=1)
    def forward(self, x):
        H,W = x.shape[-2:]
        feats = [x] + [F.interpolate(s(x), size=(H,W), mode="bilinear", align_corners=False) for s in self.stages]
        return self.fuse(torch.cat(feats, dim=1))

class EdgeBranch(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            conv_bn_relu(ch, ch, 3,1,1),
            nn.Conv2d(ch, 1, 3,1,1)
        )
    def forward(self, feat):
        return self.conv(feat)  # logits of edge

class EdgeReconstructHead(nn.Module):
    """ERM：边缘重建辅助头（输入为 image+prob 的浅层特征，输出 edge_pred，并作为 gate 细化 logits）"""
    def __init__(self, in_ch=4, mid_ch=64):
        super().__init__()
        self.enc = nn.Sequential(
            conv_bn_relu(in_ch, mid_ch, 3,2,1),  # H/2
            conv_bn_relu(mid_ch, mid_ch, 3,2,1), # H/4
            conv_bn_relu(mid_ch, mid_ch, 3,1,1),
        )
        self.edge_head = nn.Conv2d(mid_ch, 1, 3,1,1)
        self.refine = nn.Sequential(
            conv_bn_relu(2, 16, 3,1,1),  # concat(logits, edge_sigmoid)
            nn.Conv2d(16, 1, 3,1,1)
        )

    def forward(self, image, logits):
        prob = torch.sigmoid(logits)
        x = torch.cat([image, prob], dim=1)             # Bx4xHxW
        feat = self.enc(x)                               # BxCxH/4xW/4
        edge_low = self.edge_head(feat)                  # Bx1xH/4xW/4
        edge = F.interpolate(edge_low, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(edge)
        res  = self.refine(torch.cat([logits, gate], dim=1))
        return logits + res, edge

class PPMEdgeFreqRefineHead(nn.Module):
    def __init__(self, use_freq=True, use_edge=True, use_erm=False, mid_ch=96):
        super().__init__()
        self.use_freq = use_freq
        self.use_edge = use_edge
        self.use_erm  = use_erm

        # 主 PPM 在 (prob 引导) 空间
        self.pre = conv_bn_relu(4, mid_ch, 3,1,1)
        self.ppm = PPM(mid_ch, mid_ch)

        # 频域校正
        if use_freq:
            self.ffr = FrequencyFeatureRectifier(mid_ch)

        # 显式 edge 分支（辅助融合）
        if use_edge:
            self.edge_branch = EdgeBranch(mid_ch)

        # 输出头
        self.out_head = nn.Conv2d(mid_ch, 1, 3,1,1)

        # 边缘重建辅助头（ERM）
        self.erm = EdgeReconstructHead(in_ch=4, mid_ch=mid_ch//2) if use_erm else None

    def forward(self, image, logits):
        prob = torch.sigmoid(logits)
        x = torch.cat([image, prob], dim=1)  # Bx4xHxW
        feat = self.pre(x)
        feat = self.ppm(feat)
        if self.use_freq: feat = self.ffr(feat)
        if self.use_edge:
            e = self.edge_branch(feat)
            feat = feat + torch.sigmoid(e).expand_as(feat) * 0.2  # 轻量边缘引导

        out = self.out_head(feat)
        out = out + logits  # 残差到主掩码

        ret = {"logits": out}
        if self.erm is not None:
            out_ref, edge_pred = self.erm(image, out)
            ret["logits"] = out_ref
            ret["edge_pred"] = edge_pred
        return ret
