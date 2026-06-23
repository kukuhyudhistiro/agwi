"""
make_figures.py — Generate paper figures (self-contained)
Author: Kukuh Yudhistiro, 2026

Produces:
  1. Qualitative comparison grid: Original | rho_L | Canny | Sobel | PC | GWC | GWi | A-GWi
  2. rho_L density distribution plot (justifies adaptation)
  3. Adaptive parameter visualization (lambda map, kernel-size map)

Usage:
  python make_figures.py --data-root ./data --output-root ./output \
      --figures-dir ./figures --dataset BSDS500 \
      --image-ids 100007 101085 3096 --n-thresholds 99
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from methods import (
    preprocess, run_method, run_agwi, estimate_density,
    warmup_agwi, AGWiParams, HAS_PHASEPACK,
)


DATASET_LAYOUT = {
    "BSDS500": {"img_dir": "BSDS500/images/test", "img_ext": ".jpg"},
    "UDED": {"img_dir": "UDED/imgs", "img_ext": ".jpg"},
}


def normalize01(m):
    mn, mx = m.min(), m.max()
    return (m - mn) / (mx - mn) if mx > mn else np.zeros_like(m)


def qualitative_grid(data_root, dataset, image_ids, figures_dir, params):
    """One row per image, columns = methods."""
    methods = ["Canny", "Sobel", "GWC", "GWi", "AGWi"]
    if HAS_PHASEPACK:
        methods.insert(2, "PC")

    n_cols = 2 + len(methods)  # original + rho_L + methods
    n_rows = len(image_ids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    layout = DATASET_LAYOUT[dataset]
    for ri, img_id in enumerate(image_ids):
        img_path = data_root / layout["img_dir"] / f"{img_id}{layout['img_ext']}"
        if not img_path.exists():
            print(f"[WARN] missing {img_path}")
            continue
        gray_f, gray_u8 = preprocess(img_path)
        rho_L = estimate_density(gray_f)

        col = 0
        axes[ri, col].imshow(gray_u8, cmap="gray")
        if ri == 0:
            axes[ri, col].set_title("Original", fontsize=9)
        axes[ri, col].axis("off")
        col += 1

        im = axes[ri, col].imshow(rho_L, cmap="hot", vmin=0, vmax=1)
        if ri == 0:
            axes[ri, col].set_title(r"$\rho_L$", fontsize=9)
        axes[ri, col].axis("off")
        col += 1

        for method in methods:
            mag, _ = run_method(method, gray_f, gray_u8, agwi_params=params)
            axes[ri, col].imshow(normalize01(mag), cmap="gray")
            if ri == 0:
                axes[ri, col].set_title(method, fontsize=9)
            axes[ri, col].axis("off")
            col += 1

    plt.tight_layout()
    out = figures_dir / f"qualitative_{dataset}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def density_distribution(data_root, dataset, figures_dir, max_images=50):
    """Histogram of rho_L values across the dataset (justifies adaptation)."""
    layout = DATASET_LAYOUT[dataset]
    img_dir = data_root / layout["img_dir"]
    paths = sorted(img_dir.glob(f"*{layout['img_ext']}"))[:max_images]

    all_rho = []
    per_image_std = []
    for p in paths:
        gray_f, _ = preprocess(p)
        rho = estimate_density(gray_f)
        all_rho.append(rho.ravel())
        per_image_std.append(rho.std())
    all_rho = np.concatenate(all_rho)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(all_rho, bins=50, color="steelblue", edgecolor="black")
    axes[0].set_xlabel(r"$\rho_L$ value")
    axes[0].set_ylabel("Pixel count")
    axes[0].set_title(f"{dataset}: pixel-level $\\rho_L$ distribution")

    axes[1].hist(per_image_std, bins=20, color="indianred", edgecolor="black")
    axes[1].set_xlabel(r"Per-image $\rho_L$ std. dev.")
    axes[1].set_ylabel("Image count")
    axes[1].set_title(f"{dataset}: within-image density variation")

    plt.tight_layout()
    out = figures_dir / f"density_distribution_{dataset}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")
    print(f"  Mean within-image rho_L std: {np.mean(per_image_std):.4f}")


def adaptive_param_maps(data_root, dataset, image_id, figures_dir, params):
    """Visualize lambda map and kernel-size map for one image."""
    layout = DATASET_LAYOUT[dataset]
    img_path = data_root / layout["img_dir"] / f"{image_id}{layout['img_ext']}"
    if not img_path.exists():
        print(f"[WARN] missing {img_path}")
        return
    gray_f, gray_u8 = preprocess(img_path)
    rho_L = estimate_density(gray_f)

    # Compute f0 and sigma maps (f0 formulation)
    sig = 1.0 / (1.0 + np.exp(-params.k_steepness * (rho_L - 0.5)))
    f0 = params.f_min + (params.f_max - params.f_min) * sig
    sigma = (1.0 / (np.pi * f0)) * np.sqrt(np.log(2.0) / 2.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gray_u8, cmap="gray"); axes[0].set_title("Original")
    im1 = axes[1].imshow(rho_L, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(r"$\rho_L$"); plt.colorbar(im1, ax=axes[1], shrink=0.7)
    im2 = axes[2].imshow(f0, cmap="viridis")
    axes[2].set_title(r"$f_{0,adapt}$"); plt.colorbar(im2, ax=axes[2], shrink=0.7)
    im3 = axes[3].imshow(sigma, cmap="plasma")
    axes[3].set_title(r"$\sigma_{adapt}$"); plt.colorbar(im3, ax=axes[3], shrink=0.7)
    for a in axes:
        a.axis("off")

    plt.tight_layout()
    out = figures_dir / f"adaptive_maps_{dataset}_{image_id}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("./output"))
    parser.add_argument("--figures-dir", type=Path, default=Path("./figures"))
    parser.add_argument("--dataset", default="BSDS500")
    parser.add_argument("--image-ids", nargs="+", required=True)
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    params = AGWiParams()
    warmup_agwi(params)

    qualitative_grid(args.data_root, args.dataset, args.image_ids,
                     args.figures_dir, params)
    density_distribution(args.data_root, args.dataset, args.figures_dir)
    adaptive_param_maps(args.data_root, args.dataset, args.image_ids[0],
                        args.figures_dir, params)


if __name__ == "__main__":
    main()
