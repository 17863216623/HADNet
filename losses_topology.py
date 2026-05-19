# losses_topology.py
import torch
import torch.nn.functional as F

def soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3,1), stride=1, padding=(1,0))
    p2 = -F.max_pool2d(-img, kernel_size=(1,3), stride=1, padding=(0,1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def soft_open(img):
    return soft_dilate(soft_erode(img))

@torch.no_grad()
def _norm01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

def soft_skeletonize(img, iters=10):
    # img: Bx1xHxW in [0,1]
    img = img.clamp(0,1)
    skel = torch.zeros_like(img)
    for _ in range(iters):
        opened = soft_open(img)
        delta  = (img - opened).relu()
        skel = (skel + delta).clamp(0,1)
        img  = soft_erode(img).clamp(0,1)
    return skel

def cldice_loss(logits, target, skel_iters=10, eps=1e-6):
    # logits: Bx1xHxW, target: Bx1xHxW in {0,1}
    prob = torch.sigmoid(logits)
    SkelP = soft_skeletonize(prob, iters=skel_iters)
    SkelG = soft_skeletonize(target, iters=skel_iters)

    tprec = ((SkelP * target).sum(dim=(2,3)) + eps) / (SkelP.sum(dim=(2,3)) + eps)
    tsens = ((SkelG * prob ).sum(dim=(2,3)) + eps) / (SkelG.sum(dim=(2,3)) + eps)

    cl = (2 * tprec * tsens) / (tprec + tsens + eps)
    return 1 - cl.mean()
