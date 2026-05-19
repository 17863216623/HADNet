# mamba_decoder.py
# -*- coding: utf-8 -*-
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); assert dim % 4 == 0
        self.dim = dim
    def forward(self, H, W, device):
        pe = torch.zeros(1, self.dim, H, W, device=device)
        dim_half = self.dim // 2
        div_y = torch.exp(torch.arange(0, dim_half, 2, device=device, dtype=torch.float32) * (-math.log(10000.0)/dim_half))
        div_x = torch.exp(torch.arange(0, dim_half, 2, device=device, dtype=torch.float32) * (-math.log(10000.0)/dim_half))
        pos_y = torch.arange(0, H, device=device, dtype=torch.float32).unsqueeze(1)
        pos_x = torch.arange(0, W, device=device, dtype=torch.float32).unsqueeze(1)
        sin_y, cos_y = torch.sin(pos_y * div_y), torch.cos(pos_y * div_y)
        sin_x, cos_x = torch.sin(pos_x * div_x), torch.cos(pos_x * div_x)
        sin_y = sin_y.t().unsqueeze(0).unsqueeze(-1).expand(1, dim_half//2, H, W)
        cos_y = cos_y.t().unsqueeze(0).unsqueeze(-1).expand(1, dim_half//2, H, W)
        sin_x = sin_x.t().unsqueeze(0).unsqueeze(2).expand(1, dim_half//2, H, W)
        cos_x = cos_x.t().unsqueeze(0).unsqueeze(2).expand(1, dim_half//2, H, W)
        pe[:, 0:dim_half:2, :, :] = sin_y
        pe[:, 1:dim_half:2, :, :] = cos_y
        pe[:, dim_half::2, :, :]  = sin_x
        pe[:, dim_half+1::2, :, :] = cos_x
        return pe

class ImageTokenEncoder(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, out_ch=192):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch//2, 3,2,1, bias=False), nn.BatchNorm2d(base_ch//2), nn.GELU(),
            nn.Conv2d(base_ch//2, base_ch, 3,2,1, bias=False), nn.BatchNorm2d(base_ch), nn.GELU(),
            nn.Conv2d(base_ch, out_ch, 3,1,1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU()
        )
        self.pos = PositionalEncoding2D(out_ch)
    def forward(self, img, prob):
        x = torch.cat([img, prob], dim=1)      # Bx4xHxW
        feat = self.enc(x)                     # BxCxH/4xW/4
        B,C,H4,W4 = feat.shape
        feat = feat + self.pos(H4,W4,feat.device)
        tok = feat.flatten(2).permute(0,2,1)   # BxLxC
        return feat, tok

class GroupSSM1D(nn.Module):
    def __init__(self, dim, kernel_size=9, groups=4, expansion=2):
        super().__init__()
        assert kernel_size%2==1
        self.norm1 = nn.LayerNorm(dim)
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.proj = nn.Conv1d(dim, dim, 1, bias=False)
        self.glu_u = nn.Linear(dim, dim)
        self.glu_v = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim*expansion
        self.ff1 = nn.Linear(dim, hidden); self.ff2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
    def forward(self, x): # BxLxC
        idt = x; x = self.norm1(x)
        x = self.proj(self.dw(x.transpose(1,2))).transpose(1,2)
        g = torch.sigmoid(self.glu_u(x))
        x = idt + g * self.glu_v(x)
        idt2 = x; x = self.norm2(x)
        x = self.ff2(self.act(self.ff1(x)))
        return idt2 + x

class GMSAMDecoderLite(nn.Module):
    def __init__(self, num_queries=12, token_dim=192, num_layers=3, groups=4, kernel_size=9, mask_embed_dim=64):
        super().__init__()
        self.img_tok = ImageTokenEncoder(4, 64, token_dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, token_dim) * 0.02)
        self.layers  = nn.ModuleList([GroupSSM1D(token_dim, kernel_size, groups) for _ in range(num_layers)])
        self.tok_reduce = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim), nn.GELU())
        self.q_gate     = nn.Sequential(nn.Linear(token_dim, token_dim), nn.Sigmoid())
        self.mask_embed = nn.Sequential(
            nn.Conv2d(token_dim, mask_embed_dim, 3,1,1, bias=False), nn.GroupNorm(8, mask_embed_dim), nn.GELU()
        )
        self.to_mask_coeff = nn.Linear(token_dim, mask_embed_dim)
        self.iou_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim,1))

    def forward(self, image, logits):
        prob = torch.sigmoid(logits)
        base_feat, tokens = self.img_tok(image, prob)     # BxCxH/4xW/4, BxLxC
        B,L,C = tokens.shape
        q = self.queries.expand(B,-1,-1).contiguous()     # BxNxC
        tok_mean, tok_max = tokens.mean(dim=1), tokens.max(dim=1)[0]
        summary = self.tok_reduce(tok_mean + tok_max)     # BxC
        for blk in self.layers:
            gate = self.q_gate(summary).unsqueeze(1)      # Bx1xC
            q = q + gate * q
            q = blk(q)
        mask_feat = self.mask_embed(base_feat)            # BxE xH/4xW/4
        coeff = self.to_mask_coeff(q)                     # BxN xE
        masks_low = torch.einsum("bne,behw->bnhw", coeff, mask_feat)
        masks = F.interpolate(masks_low, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        w = torch.sigmoid(self.iou_head(q)).squeeze(-1)   # BxN
        w = (w/(w.sum(dim=1,keepdim=True)+1e-6)).unsqueeze(1)  # Bx1xN
        residual = torch.bmm(w, masks.view(B, masks.shape[1], -1)).view(B,1,*logits.shape[-2:])
        return logits + residual
