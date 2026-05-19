#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 PySODMetrics 对 SOD / MAS / USOD10K 预测结果进行标准评估。

指标：
  - MAE
  - Sα (Smeasure)
  - Fwβ (WeightedFmeasure)
  - mEϕ / maxEϕ / adpEϕ (Emeasure)
  - maxF / avgF / adpF (Fmeasure, beta=0.3)
  - mIoU / maxIoU / adpIoU (通过 FmeasureV2+IOUHandler 得到)
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from py_sod_metrics import (  # pip install pysodmetrics
    MAE,
    Emeasure,
    Smeasure,
    Fmeasure,
    WeightedFmeasure,
    FmeasureV2,
    FmeasureHandler,
    IOUHandler,
)


class SODMetricRecorder:
    """
    基于 PySODMetrics 的指标收集器：
    - 逐样本调用 step(pred, gt)
    - 结束后调用 get_results() 得到整体指标
    """

    def __init__(self):
        # 基本五大指标
        self.mae = MAE()
        self.em = Emeasure()
        self.sm = Smeasure()
        self.fm = Fmeasure()           # 老版本 Fmeasure（方便拿 Em/Fm 的 max / mean / adp）
        self.wfm = WeightedFmeasure()

        # FmeasureV2：同时统计 Fβ 和 IoU 的 max/mean/adp（Dual-SAM 风格里的 mIoU 可以从这里取）
        self.fmv2 = FmeasureV2(
            metric_handlers={
                "fm": FmeasureHandler(beta=0.3, with_adaptive=True, with_dynamic=True),
                "iou": IOUHandler(with_adaptive=True, with_dynamic=True),
            }
        )

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """
        保证传入 PySODMetrics 的是 uint8 且 [0,255]，这是官方要求。
        gt 通常是 {0,255} 的二值图。
        """
        if img.dtype == np.uint8:
            return img
        # 先转 float
        arr = img.astype(np.float32)
        # 如果是 0~1 范围，放大到 0~255
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def step(self, pred: np.ndarray, gt: np.ndarray):
        """
        pred, gt: HxW 灰度图 (预测可以是灰度显著图, gt 通常是二值 mask)
        """
        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"

        pred = self._to_uint8(pred)
        gt = self._to_uint8(gt)

        # 更新所有指标
        self.mae.step(pred, gt)
        self.em.step(pred, gt)
        self.sm.step(pred, gt)
        self.fm.step(pred, gt)
        self.wfm.step(pred, gt)
        self.fmv2.step(pred, gt)

    def get_results(self, num_bits: int = 4):
        """
        返回一个 dict，字段命名尽量贴近你之前的命名习惯和 Dual-SAM 论文：
          - MAE
          - S_alpha
          - Fw_beta
          - mE_phi / maxE_phi / adpE_phi
          - maxF / avgF / adpF
          - mIoU / maxIoU / adpIoU
        """
        res = {}

        # 1) 基本指标
        res["MAE"] = round(float(self.mae.get_results()["mae"]), num_bits)
        res["S_alpha"] = round(float(self.sm.get_results()["sm"]), num_bits)
        res["Fw_beta"] = round(float(self.wfm.get_results()["wfm"]), num_bits)

        # 2) E-measure（mEϕ、maxEϕ、adpEϕ）
        em = self.em.get_results()["em"]
        em_curve = em["curve"]  # 256 阈值上的曲线
        res["mE_phi"] = round(float(em_curve.mean()), num_bits)   # mean Em (常用的 mEϕ)
        res["maxE_phi"] = round(float(em_curve.max()), num_bits)  # max Em
        res["adpE_phi"] = round(float(em["adp"]), num_bits)       # adaptive Em

        # 3) F-measure（maxF / avgF / adpF），这里用老版 Fmeasure
        fm = self.fm.get_results()["fm"]
        fm_curve = fm["curve"]
        res["maxF"] = round(float(fm_curve.max()), num_bits)
        res["avgF"] = round(float(fm_curve.mean()), num_bits)
        res["adpF"] = round(float(fm["adp"]), num_bits)

        # 4) FmeasureV2 里的 IoU (以及 Fβ 再算一遍，方便对比)
        fmv2_all = self.fmv2.get_results()
        iou_data = fmv2_all["iou"]
        fm2_data = fmv2_all["fm"]

        # IoU：dynamic 是 0~255 阈值上的 IoU 曲线
        iou_curve = iou_data["dynamic"]
        res["mIoU"] = round(float(iou_curve.mean()), num_bits)   # 这里定义为 mean IoU
        res["maxIoU"] = round(float(iou_curve.max()), num_bits)
        res["adpIoU"] = round(float(iou_data["adaptive"]), num_bits)

        # Fβ (beta=0.3) 的另一套数值（可以和上面 fm 对比）
        fm2_curve = fm2_data["dynamic"]
        res["maxF_v2"] = round(float(fm2_curve.max()), num_bits)
        res["avgF_v2"] = round(float(fm2_curve.mean()), num_bits)
        res["adpF_v2"] = round(float(fm2_data["adaptive"]), num_bits)

        return res


def collect_pairs(pred_dir: str, gt_dir: str,
                  pred_ext: str = ".png", gt_ext: str = ".png"):
    """
    根据文件名（不含后缀）自动配对 pred / gt.
    要求：pred 和 gt 的文件名一致（比如 xxx.png 对 xxx.png）。
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    pred_files = sorted(
        [p for p in pred_dir.iterdir()
         if p.is_file() and p.suffix.lower() == pred_ext.lower()]
    )
    gt_files = sorted(
        [p for p in gt_dir.iterdir()
         if p.is_file() and p.suffix.lower() == gt_ext.lower()]
    )

    gt_map = {p.stem: p for p in gt_files}
    pairs = []

    for p in pred_files:
        stem = p.stem
        if stem not in gt_map:
            print(f"[WARN] 找不到对应 GT: {stem}{gt_ext}，跳过该预测: {p.name}")
            continue
        pairs.append((str(p), str(gt_map[stem])))

    if len(pairs) == 0:
        raise RuntimeError("没有成功配对到任何 pred / gt，请检查目录和后缀。")

    print(f"[Info] 共找到 {len(pairs)} 对有效的 (pred, gt) 图像进行评估。")
    return pairs


def evaluate_with_py_sod(pred_dir: str,
                         gt_dir: str,
                         pred_ext: str = ".png",
                         gt_ext: str = ".png",
                         save_csv: Optional[str] = None):
    """
    pred_dir: 预测显著图 / 分割图所在目录（灰度图）
    gt_dir:   GT mask 目录（0/255）
    """
    pairs = collect_pairs(pred_dir, gt_dir, pred_ext, gt_ext)
    recorder = SODMetricRecorder()

    for idx, (pred_path, gt_path) in enumerate(pairs, start=1):
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred is None:
            print(f"[WARN] 读取预测失败: {pred_path}，跳过。")
            continue
        if gt is None:
            print(f"[WARN] 读取 GT 失败: {gt_path}，跳过。")
            continue

        # 尺寸对齐
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, dsize=(gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        recorder.step(pred, gt)

        if idx % 200 == 0 or idx == len(pairs):
            print(f"[Eval] 已处理 {idx}/{len(pairs)}")

    results = recorder.get_results()

    print("=" * 60)
    print("[PySODMetrics] Evaluation Results (整体指标)")
    print("=" * 60)
    # 核心五个：和 Dual-SAM 论文一一对应
    print(f"mIoU    : {results['mIoU']:.4f}")
    print(f"Sα      : {results['S_alpha']:.4f}")
    print(f"Fwβ     : {results['Fw_beta']:.4f}")
    print(f"mEϕ     : {results['mE_phi']:.4f}")
    print(f"MAE     : {results['MAE']:.4f}")
    print("-" * 60)
    # 其他可选：方便你做消融分析 / 附表
    print(f"maxEϕ   : {results['maxE_phi']:.4f}")
    print(f"adpEϕ   : {results['adpE_phi']:.4f}")
    print(f"maxF    : {results['maxF']:.4f}")
    print(f"avgF    : {results['avgF']:.4f}")
    print(f"adpF    : {results['adpF']:.4f}")
    print(f"maxIoU  : {results['maxIoU']:.4f}")
    print(f"adpIoU  : {results['adpIoU']:.4f}")
    print("=" * 60)

    if save_csv is not None:
        import csv
        os.makedirs(os.path.dirname(save_csv) if os.path.dirname(save_csv) else ".", exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in results.items():
                writer.writerow([k, f"{v:.4f}"])
        print(f"[CSV] 指标已保存到: {save_csv}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use PySODMetrics to evaluate SOD/MAS predictions."
    )
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="预测结果(灰度显著图)目录")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="GT mask 目录")
    parser.add_argument("--pred_ext", type=str, default=".png",
                        help="预测图后缀 (默认 .png)")
    parser.add_argument("--gt_ext", type=str, default=".png",
                        help="GT 后缀 (默认 .png)")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="可选，将指标保存到 csv，路径例如 ./metrics_pysod_usod10k.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_with_py_sod(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        pred_ext=args.pred_ext,
        gt_ext=args.gt_ext,
        save_csv=args.save_csv,
    )

