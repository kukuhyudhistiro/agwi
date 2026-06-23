#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_final_figures.py — Final qualitative figures for the A-GWi paper
Author: Kukuh Yudhistiro, 2026

Produces the per-dataset qualitative comparison figures used in Section 5.2.
Rows = methods (Original, GT, Canny, Sobel, PC, GWC, GWi, A-GWi).
Columns = representative images.

White-background convention: edge maps and GT are shown with cmap='gray_r'
(black edges on white) to meet scientific writing standards, matching the
06c figure convention.

Image IDs:
  BSDS500: 3063, 29030, 128035, 35049
  UDED:    04-0896x4, 05-WIREFRAME-2, 28-img_043_SRF_2_HR

Usage:
  python generate_final_figures.py --data-root ./data --output-root ./output \
      --figures-dir ./figures
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
import cv2
cv2.setNumThreads(1)
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SAMPLES = {
    "BSDS500": ["3063", "29030", "128035", "35049"],
    "UDED": ["04-0896x4", "05-WIREFRAME-2", "28-img_043_SRF_2_HR"],
}

DATASET_LAYOUT = {
    "BSDS500": {"img_dir": "BSDS500/images/test",
                "gt_dir": "BSDS500/groundTruth/test", "gt_fmt": "mat"},
    "UDED": {"img_dir": "UDED/imgs",
             "gt_dir": "UDED/gt", "gt_fmt": "png"},
}

# Methods shown as columns of edge maps (folder names under output/<dataset>/)
METHOD_ORDER = ["Canny", "Sobel", "PC", "GWC", "GWi", "AGWi"]
METHOD_LABELS = {
    "Canny": "Canny", "Sobel": "Sobel", "PC": "PC",
    "GWC": "GWC", "GWi": "GWi", "AGWi": "A-GWi",
}

IMG_EXTS = [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]


def setup_matplotlib():
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "axes.titlesize": 10,
        "savefig.dpi": 300,
    })


def find_image_file(image_dir, image_id):
    for ext in IMG_EXTS:
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def load_gt_for_display(dataset, gt_dir, image_id):
    """Return GT boundary map as 0..255 (edges=255), or None."""
    fmt = DATASET_LAYOUT[dataset]["gt_fmt"]
    if fmt == "mat":
        gt_path = gt_dir / f"{image_id}.mat"
        if not gt_path.exists():
            return None
        mat = loadmat(str(gt_path))
        gt = mat["groundTruth"]
        acc = None
        for i in range(gt.shape[1]):
            b = (gt[0, i]["Boundaries"][0, 0] > 0).astype(np.float32)
            acc = b if acc is None else np.maximum(acc, b)
        return (acc * 255).astype(np.uint8)
    else:
        gt_path = gt_dir / f"{image_id}.png"
        if not gt_path.exists():
            return None
        arr = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        return (arr > 127).astype(np.uint8) * 255


def load_edge_map(output_root, dataset, method, image_id):
    p = output_root / dataset / method / f"{image_id}.png"
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def apply_clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.5)


def figure_qualitative(dataset, data_root, output_root, figures_dir, sample_ids):
    layout = DATASET_LAYOUT[dataset]
    img_dir = data_root / layout["img_dir"]
    gt_dir = data_root / layout["gt_dir"]

    row_labels = ["Original", "GT"] + [METHOD_LABELS[m] for m in METHOD_ORDER]
    n_rows = len(row_labels)
    n_cols = len(sample_ids)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.1 * n_cols, 2.1 * n_rows))
    if n_cols == 1:
        axes = axes[:, None]

    for c, image_id in enumerate(sample_ids):
        img_path = find_image_file(img_dir, image_id)
        if img_path is None:
            print(f"  [WARN] {dataset}/{image_id}: image not found")
            for r in range(n_rows):
                axes[r, c].axis("off")
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # Row 0: original (true grayscale)
        axes[0, c].imshow(gray, cmap="gray", vmin=0, vmax=255)
        apply_clean_axis(axes[0, c])
        axes[0, c].set_title(image_id, fontsize=8)

        # Row 1: GT inverted (black edges on white)
        gt = load_gt_for_display(dataset, gt_dir, image_id)
        if gt is not None:
            axes[1, c].imshow(gt, cmap="gray_r", vmin=0, vmax=255)
        apply_clean_axis(axes[1, c])

        # Rows 2..: edge maps, inverted (black edges on white)
        for r, method in enumerate(METHOD_ORDER, start=2):
            edge = load_edge_map(output_root, dataset, method, image_id)
            if edge is not None:
                axes[r, c].imshow(edge, cmap="gray_r", vmin=0, vmax=255)
            apply_clean_axis(axes[r, c])

    # Row labels on the left
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=90,
                              labelpad=8, va="center")

    plt.tight_layout(pad=0.3)
    out = figures_dir / f"qualitative_{dataset}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close()
    print(f"[OK] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--figures-dir", type=Path, default=Path("./figures"))
    ap.add_argument("--datasets", nargs="+", default=["BSDS500", "UDED"])
    args = ap.parse_args()

    setup_matplotlib()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        if dataset not in DATASET_LAYOUT:
            print(f"[WARN] unknown dataset {dataset}")
            continue
        sample_ids = DEFAULT_SAMPLES.get(dataset, [])
        if not sample_ids:
            continue
        print(f"\n[{dataset}] {sample_ids}")
        figure_qualitative(dataset, args.data_root, args.output_root,
                            args.figures_dir, sample_ids)

    print("\nDone. Figures use white background (black edges on white).")


if __name__ == "__main__":
    main()
