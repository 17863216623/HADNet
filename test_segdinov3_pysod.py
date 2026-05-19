#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegDINO / WaveMamba-UGBR 单独测试脚本：
- 使用 usod10k_test 数据集下的 test/image 与 test/mask
- 按 PySODMetrics 标准在原始 GT 尺寸上评估
- 可选保存连续灰度预测图（0~255）

建议把本脚本放到你的工程目录（与 train_segdinov3.py 同级）后运行。
"""

import os
import sys
import csv
import json
import glob
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def _bootstrap_project_dir():
    # 在正式 argparse 之前，先从命令行里提取 --project_dir，用于导入项目模块
    project_dir = None
    argv = sys.argv[1:]
    for i, x in enumerate(argv):
        if x == "--project_dir" and i + 1 < len(argv):
            project_dir = argv[i + 1]
            break
        if x.startswith("--project_dir="):
            project_dir = x.split("=", 1)[1]
            break

    if project_dir is None:
        project_dir = os.path.dirname(os.path.abspath(__file__))

    project_dir = os.path.abspath(project_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    return project_dir


PROJECT_DIR = _bootstrap_project_dir()

from mamba_decoder import GMSAMDecoderLite
from ugbd_refiner import UGBDRefiner
from sdf_head import SDFHead

try:
    from py_sod_metrics import (
        MAE,
        Emeasure,
        Smeasure,
        Fmeasure,
        WeightedFmeasure,
        FmeasureV2,
        FmeasureHandler,
        IOUHandler,
    )
except ImportError as e:
    raise ImportError(
        "未检测到 py_sod_metrics，请先安装：pip install pysodmetrics"
    ) from e


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


class TestFolderDataset(Dataset):
    """
    与训练集配对逻辑保持一致，但测试时保留 GT 原始尺寸。
    模型输入会 resize 到指定大小；评估时再把预测 resize 回 GT 原图大小。
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        img_dir_name: str = "image",
        label_dir_name: str = "mask",
        img_ext: Optional[str] = None,
        mask_ext: Optional[str] = ".png",
        size: Tuple[int, int] = (384, 384),
    ):
        super().__init__()
        self.root = os.path.abspath(root)
        self.split = split
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.size = size  # (H, W)

        split_root = os.path.join(self.root, split)
        img_dir_split = os.path.join(split_root, img_dir_name)
        msk_dir_split = os.path.join(split_root, label_dir_name)
        img_dir_flat = os.path.join(self.root, img_dir_name)
        msk_dir_flat = os.path.join(self.root, label_dir_name)

        if os.path.isdir(img_dir_split) and os.path.isdir(msk_dir_split):
            self.img_dir, self.msk_dir = img_dir_split, msk_dir_split
        elif os.path.isdir(img_dir_flat) and os.path.isdir(msk_dir_flat):
            self.img_dir, self.msk_dir = img_dir_flat, msk_dir_flat
        else:
            raise FileNotFoundError(
                f"未找到数据目录：\n{img_dir_split}\n{msk_dir_split}\n{img_dir_flat}\n{msk_dir_flat}"
            )

        self.samples = self._pairs()
        if not self.samples:
            raise RuntimeError(f"在 {self.img_dir} 下没有找到有效图像-标注对")

        print(f"[TestFolderDataset] root={self.root}")
        print(f"[TestFolderDataset] split={self.split}, pairs={len(self.samples)}")
        print(f"[TestFolderDataset] img_dir={self.img_dir}")
        print(f"[TestFolderDataset] msk_dir={self.msk_dir}")

    def _pairs(self) -> List[Tuple[str, str]]:
        img_paths: List[str] = []
        if self.img_ext:
            img_paths += glob.glob(os.path.join(self.img_dir, f"*{self.img_ext}"))
            img_paths += glob.glob(os.path.join(self.img_dir, f"*{self.img_ext.upper()}"))
        else:
            for e in IMG_EXTS:
                img_paths += glob.glob(os.path.join(self.img_dir, f"*{e}"))
                img_paths += glob.glob(os.path.join(self.img_dir, f"*{e.upper()}"))

        img_paths = sorted(list(dict.fromkeys(img_paths)))
        pairs = []
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            if self.mask_ext:
                cand = os.path.join(self.msk_dir, stem + self.mask_ext)
                if os.path.isfile(cand):
                    pairs.append((ip, cand))
            else:
                for e in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".bmp", ".BMP"]:
                    cand = os.path.join(self.msk_dir, stem + e)
                    if os.path.isfile(cand):
                        pairs.append((ip, cand))
                        break
        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, gt_path = self.samples[idx]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None:
            raise RuntimeError(f"读取失败: {img_path}")
        if gt is None:
            raise RuntimeError(f"读取失败: {gt_path}")

        orig_h, orig_w = gt.shape[:2]
        in_h, in_w = self.size

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(img_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        img_in = img_in.astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.transpose(img_in, (2, 0, 1))).float()  # [3,H,W]

        gt_bin = (gt > 127).astype(np.uint8) * 255
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return {
            "image": img_t,
            "gt_uint8": gt_bin,
            "gt_path": gt_path,
            "img_path": img_path,
            "stem": stem,
            "orig_hw": (orig_h, orig_w),
        }


def collate_fn(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    gts = [x["gt_uint8"] for x in batch]
    gt_paths = [x["gt_path"] for x in batch]
    img_paths = [x["img_path"] for x in batch]
    stems = [x["stem"] for x in batch]
    orig_hws = [x["orig_hw"] for x in batch]
    return {
        "images": images,
        "gts": gts,
        "gt_paths": gt_paths,
        "img_paths": img_paths,
        "stems": stems,
        "orig_hws": orig_hws,
    }


@torch.no_grad()
def forward_with_tta(backbone, decoder, x, scales=(1.0, 0.75, 1.25), do_flip=True):
    outs = []
    _, _, H, W = x.shape
    for s in scales:
        xi = x if s == 1.0 else F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        flips = [False, True] if do_flip else [False]
        for flip in flips:
            xif = torch.flip(xi, dims=[3]) if flip else xi
            logits = decoder(xif, backbone(xif))
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            if flip:
                logits = torch.flip(logits, dims=[3])
            outs.append(torch.sigmoid(logits))
    prob = torch.mean(torch.stack(outs, dim=0), dim=0)
    return torch.logit(prob.clamp(1e-6, 1 - 1e-6))


class SODMetricRecorder:
    def __init__(self):
        self.mae = MAE()
        self.em = Emeasure()
        self.sm = Smeasure()
        self.fm = Fmeasure()
        self.wfm = WeightedFmeasure()
        self.fmv2 = FmeasureV2(
            metric_handlers={
                "fm": FmeasureHandler(beta=0.3, with_adaptive=True, with_dynamic=True),
                "iou": IOUHandler(with_adaptive=True, with_dynamic=True),
            }
        )

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        arr = img.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def step(self, pred: np.ndarray, gt: np.ndarray):
        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
        pred = self._to_uint8(pred)
        gt = self._to_uint8(gt)
        self.mae.step(pred, gt)
        self.em.step(pred, gt)
        self.sm.step(pred, gt)
        self.fm.step(pred, gt)
        self.wfm.step(pred, gt)
        self.fmv2.step(pred, gt)

    def get_results(self, num_bits: int = 4):
        res = {}
        res["MAE"] = round(float(self.mae.get_results()["mae"]), num_bits)
        res["S_alpha"] = round(float(self.sm.get_results()["sm"]), num_bits)
        res["Fw_beta"] = round(float(self.wfm.get_results()["wfm"]), num_bits)

        em = self.em.get_results()["em"]
        em_curve = em["curve"]
        res["mE_phi"] = round(float(em_curve.mean()), num_bits)
        res["maxE_phi"] = round(float(em_curve.max()), num_bits)
        res["adpE_phi"] = round(float(em["adp"]), num_bits)

        fm = self.fm.get_results()["fm"]
        fm_curve = fm["curve"]
        res["maxF"] = round(float(fm_curve.max()), num_bits)
        res["avgF"] = round(float(fm_curve.mean()), num_bits)
        res["adpF"] = round(float(fm["adp"]), num_bits)

        fmv2_all = self.fmv2.get_results()
        iou_data = fmv2_all["iou"]
        fm2_data = fmv2_all["fm"]

        iou_curve = iou_data["dynamic"]
        res["mIoU"] = round(float(iou_curve.mean()), num_bits)
        res["maxIoU"] = round(float(iou_curve.max()), num_bits)
        res["adpIoU"] = round(float(iou_data["adaptive"]), num_bits)

        fm2_curve = fm2_data["dynamic"]
        res["maxF_v2"] = round(float(fm2_curve.max()), num_bits)
        res["avgF_v2"] = round(float(fm2_curve.mean()), num_bits)
        res["adpF_v2"] = round(float(fm2_data["adaptive"]), num_bits)
        return res


def save_metrics(results: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics_py_sod.csv")
    json_path = os.path.join(save_dir, "metrics_py_sod.json")
    txt_path = os.path.join(save_dir, "metrics_py_sod.txt")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, f"{float(v):.4f}"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}: {float(v):.4f}\n")

    print(f"[Save] CSV : {csv_path}")
    print(f"[Save] JSON: {json_path}")
    print(f"[Save] TXT : {txt_path}")


def build_models(args, device):
    if args.dino_size == "b":
        vit = torch.hub.load(args.repo_dir, "dinov3_vitb16", source="local", weights=args.dino_ckpt)
    else:
        vit = torch.hub.load(args.repo_dir, "dinov3_vits16", source="local", weights=args.dino_ckpt)

    from dpt import DPT

    backbone = DPT(nclass=1, backbone=vit).to(device)
    decoder = GMSAMDecoderLite(
        num_queries=args.mamba_num_queries,
        token_dim=args.mamba_dim,
        num_layers=args.mamba_layers,
        groups=args.mamba_groups,
        kernel_size=args.mamba_kernel,
        mask_embed_dim=args.mamba_mask_embed_dim,
    ).to(device)

    refine = UGBDRefiner(mid_ch=args.ugbd_mid_ch, steps=args.ugbd_steps).to(device) if args.use_ugbd else None
    sdf_head = SDFHead(mid_ch=args.sdf_mid_ch, fuse_lambda=args.sdf_lambda).to(device) if args.use_sdf else None
    return backbone, decoder, refine, sdf_head

def clean_state_dict(state_dict):
    """
    去掉 profiling 产生的无关键，如 total_ops / total_params
    """
    cleaned = {}
    for k, v in state_dict.items():
        if k.endswith("total_ops") or k.endswith("total_params"):
            continue
        if ".total_ops" in k or ".total_params" in k:
            continue
        cleaned[k] = v
    return cleaned


def load_ckpt(ckpt_path, backbone, decoder, refine=None, sdf_head=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"checkpoint 格式异常: {type(ckpt)}")

    if "backbone" not in ckpt or "decoder" not in ckpt:
        raise KeyError(
            "当前 checkpoint 缺少 backbone / decoder 键，"
            "请确认它是 train_segdinov3.py 保存的标准权重。"
        )

    # 先清洗掉 total_ops / total_params
    backbone_sd = clean_state_dict(ckpt["backbone"])
    decoder_sd = clean_state_dict(ckpt["decoder"])

    msg_b = backbone.load_state_dict(backbone_sd, strict=False)
    msg_d = decoder.load_state_dict(decoder_sd, strict=False)

    print(f"[Load] backbone missing_keys   : {msg_b.missing_keys}")
    print(f"[Load] backbone unexpected_keys: {msg_b.unexpected_keys}")
    print(f"[Load] decoder  missing_keys   : {msg_d.missing_keys}")
    print(f"[Load] decoder  unexpected_keys: {msg_d.unexpected_keys}")

    if refine is not None:
        if "refine" not in ckpt:
            raise KeyError("你启用了 --use_ugbd，但 checkpoint 中没有 refine 权重")
        refine_sd = clean_state_dict(ckpt["refine"])
        msg_r = refine.load_state_dict(refine_sd, strict=False)
        print(f"[Load] refine   missing_keys   : {msg_r.missing_keys}")
        print(f"[Load] refine   unexpected_keys: {msg_r.unexpected_keys}")

    if sdf_head is not None:
        if "sdf_head" not in ckpt:
            raise KeyError("你启用了 --use_sdf，但 checkpoint 中没有 sdf_head 权重")
        sdf_sd = clean_state_dict(ckpt["sdf_head"])
        msg_s = sdf_head.load_state_dict(sdf_sd, strict=False)
        print(f"[Load] sdf_head missing_keys   : {msg_s.missing_keys}")
        print(f"[Load] sdf_head unexpected_keys: {msg_s.unexpected_keys}")

def pretty_print(results: dict):
    print("=" * 68)
    print("[PySODMetrics] Final Results on USOD10K-Test")
    print("=" * 68)
    order = [
        "MAE",
        "S_alpha",
        "Fw_beta",
        "mE_phi",
        "maxE_phi",
        "adpE_phi",
        "maxF",
        "avgF",
        "adpF",
        "mIoU",
        "maxIoU",
        "adpIoU",
        "maxF_v2",
        "avgF_v2",
        "adpF_v2",
    ]
    for k in order:
        if k in results:
            print(f"{k:<10}: {results[k]:.4f}")
    print("=" * 68)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="SegDINO/WaveMamba-UGBR test with PySODMetrics")

    # 数据
    parser.add_argument("--project_dir", type=str, default=PROJECT_DIR, help="项目根目录（需包含 mamba_decoder.py、dpt.py 等）")
    parser.add_argument("--data_dir", type=str, default="./segdata", help="数据根目录，例如 ./segdata")
    parser.add_argument("--dataset_test", type=str, default="usod10k_test", help="测试集目录名")
    parser.add_argument("--test_split", type=str, default="test", help="测试 split 名")
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--img_ext", type=str, default=None)
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--input_h", type=int, default=384)
    parser.add_argument("--input_w", type=int, default=384)
    parser.add_argument("--num_workers", type=int, default=4)

    # DINOv3 + 模型结构
    parser.add_argument("--repo_dir", type=str, required=True)
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--dino_size", type=str, default="s", choices=["s", "b"])
    parser.add_argument("--ckpt", type=str, required=True, help="训练得到的 best_*.pth 或 latest.pth")

    parser.add_argument("--mamba_num_queries", type=int, default=12)
    parser.add_argument("--mamba_dim", type=int, default=192)
    parser.add_argument("--mamba_layers", type=int, default=3)
    parser.add_argument("--mamba_groups", type=int, default=4)
    parser.add_argument("--mamba_kernel", type=int, default=9)
    parser.add_argument("--mamba_mask_embed_dim", type=int, default=64)

    parser.add_argument("--use_ugbd", action="store_true")
    parser.add_argument("--ugbd_steps", type=int, default=3)
    parser.add_argument("--ugbd_mid_ch", type=int, default=64)

    parser.add_argument("--use_sdf", action="store_true")
    parser.add_argument("--sdf_mid_ch", type=int, default=64)
    parser.add_argument("--sdf_lambda", type=float, default=0.5)

    # 推理与评测
    parser.add_argument("--eval_tta", action="store_true")
    parser.add_argument("--tta_scales", type=str, default="1.0,0.75,1.25")
    parser.add_argument("--tta_no_flip", action="store_true")
    parser.add_argument("--dual_sam_eval", action="store_true", help="若开启，则先二值化再评测；默认关闭以符合连续显著图评测")

    # 输出
    parser.add_argument("--save_root", type=str, default="./test_outputs")
    parser.add_argument("--save_pred", action="store_true", help="保存 0~255 连续灰度预测图")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_root, exist_ok=True)

    dataset_root = os.path.join(args.data_dir, args.dataset_test)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"未找到测试集根目录: {dataset_root}")

    run_name = Path(args.ckpt).stem
    save_dir = os.path.join(args.save_root, f"{args.dataset_test}_{run_name}")
    pred_dir = os.path.join(save_dir, "pred_gray")
    if args.save_pred:
        os.makedirs(pred_dir, exist_ok=True)

    test_set = TestFolderDataset(
        root=dataset_root,
        split=args.test_split,
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        size=(args.input_h, args.input_w),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    backbone, decoder, refine, sdf_head = build_models(args, device)
    load_ckpt(args.ckpt, backbone, decoder, refine=refine, sdf_head=sdf_head)

    backbone.eval()
    decoder.eval()
    if refine is not None:
        refine.eval()
    if sdf_head is not None:
        sdf_head.eval()

    recorder = SODMetricRecorder()
    scales = tuple(float(x) for x in args.tta_scales.split(",") if x.strip())
    do_flip = not args.tta_no_flip

    pbar = tqdm(test_loader, desc="[Test]")
    for batch in pbar:
        imgs = batch["images"].to(device, non_blocking=True)
        gt_uint8 = batch["gts"][0]
        stem = batch["stems"][0]
        orig_h, orig_w = batch["orig_hws"][0]

        if args.eval_tta:
            logits = forward_with_tta(backbone, decoder, imgs, scales=scales, do_flip=do_flip)
        else:
            logits = decoder(imgs, backbone(imgs))

        if refine is not None:
            logits = refine(imgs, logits)
        if sdf_head is not None:
            logits, _ = sdf_head(imgs, logits)

        prob = torch.sigmoid(logits)
        prob = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred = prob[0, 0].detach().cpu().numpy()

        if args.dual_sam_eval:
            pred = (pred > 0.5).astype(np.float32)

        pred_uint8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        recorder.step(pred_uint8, gt_uint8)

        if args.save_pred:
            cv2.imwrite(os.path.join(pred_dir, f"{stem}.png"), pred_uint8)

    results = recorder.get_results()
    pretty_print(results)
    save_metrics(results, save_dir)

    if args.save_pred:
        print(f"[Save] Pred dir: {pred_dir}")


if __name__ == "__main__":
    main()