#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_agwi.py — A-GWi vs static GWi vs Canny qualitative comparison
Author: Kukuh Yudhistiro, 2026

Restores the f0-based adaptive formulation from the original experiment
(sigma = (1/(pi*f0)) * sqrt(ln2/2)), which gives sharper A-GWi responses
than the wavelength-based version. Clean, compact subplot layout at 300 dpi.

Adaptive logic:
    dense  (rho_L -> 1) -> f0 -> f_max (high freq, fine detail)
    sparse (rho_L -> 0) -> f0 -> f_min (low freq, broad structure)

Usage:
    python compare_agwi.py --image path/to/image.png --output ./figures
    python compare_agwi.py --image-dir ./imagetest --output ./figures
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import jit


# ============================================================================
# Parameters (from original experiment, proven sharper)
# ============================================================================
KERNEL_SIZE = 7
NUM_ORIENTATIONS = 8
K_STEEPNESS = 25.0
F_MIN = 0.05
F_MAX = 0.45


# ============================================================================
# Static GWi kernel (imaginary only, f0 parameterization)
# ============================================================================
def gabor_imag_f0(f0, sigma_x, sigma_y, theta, size):
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    y = np.arange(-size // 2 + 1., size // 2 + 1.)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    xp = xv * np.cos(theta) + yv * np.sin(theta)
    yp = -xv * np.sin(theta) + yv * np.cos(theta)
    gauss = np.exp(-0.5 * (xp ** 2 / sigma_x ** 2 + yp ** 2 / sigma_y ** 2))
    return gauss * np.sin(2.0 * np.pi * f0 * xp)


def adaptive_scale(f0, aspect=0.5):
    sigma = (1.0 / (np.pi * f0)) * np.sqrt(np.log(2.0) / 2.0)
    return sigma, sigma / aspect


# ============================================================================
# A-GWi adaptive SVC core (f0-based, Numba accelerated)
# ============================================================================
@jit(nopython=True, cache=True)
def agwi_loop(H, W, ksize, n_orient, padded, rho_L, k_s, f_min, f_max):
    half_k = ksize // 2
    responses = np.zeros((n_orient, H, W))
    step = np.pi / n_orient
    thetas = np.arange(0.0, np.pi, step)

    for ko in range(n_orient):
        theta = thetas[ko]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for r in range(H):
            for c in range(W):
                rho = rho_L[r, c]
                # Adaptive f0: dense -> high freq
                f_sig = 1.0 / (1.0 + np.exp(-k_s * (rho - 0.5)))
                f0 = f_min + (f_max - f_min) * f_sig
                sigma_a = (1.0 / (np.pi * f0)) * np.sqrt(np.log(2.0) / 2.0)
                sigma_x = sigma_a
                sigma_y = sigma_a / 0.5

                resp = 0.0
                for xi in range(ksize):
                    for yi in range(ksize):
                        x = xi - half_k + 0.0
                        y = yi - half_k + 0.0
                        xp = x * cos_t + y * sin_t
                        yp = -x * sin_t + y * cos_t
                        g = np.exp(-0.5 * (xp ** 2 / sigma_x ** 2
                                           + yp ** 2 / sigma_y ** 2))
                        kval = g * np.sin(2.0 * np.pi * f0 * xp)
                        resp += padded[r + xi, c + yi] * kval
                responses[ko, r, c] = resp
    return responses


def run_agwi(image_f, rho_L):
    H, W = image_f.shape
    half_k = KERNEL_SIZE // 2
    padded = cv2.copyMakeBorder(image_f, half_k, half_k, half_k, half_k,
                                cv2.BORDER_REFLECT)
    t0 = time.time()
    responses = agwi_loop(H, W, KERNEL_SIZE, NUM_ORIENTATIONS,
                          padded, rho_L, K_STEEPNESS, F_MIN, F_MAX)
    rt = time.time() - t0
    return np.max(np.abs(responses), axis=0), rt


def run_gwi_static(image_f):
    """Static GWi at f0 = mid-range (fair baseline)."""
    t0 = time.time()
    f0 = (F_MIN + F_MAX) / 2.0
    sx, sy = adaptive_scale(f0)
    H, W = image_f.shape
    responses = np.zeros((NUM_ORIENTATIONS, H, W))
    for ko, theta in enumerate(np.linspace(0, np.pi, NUM_ORIENTATIONS,
                                           endpoint=False)):
        k = gabor_imag_f0(f0, sx, sy, theta, KERNEL_SIZE)
        responses[ko] = cv2.filter2D(image_f, -1, k)
    rt = time.time() - t0
    return np.max(np.abs(responses), axis=0), rt


def run_canny(gray_u8):
    t0 = time.time()
    v = np.median(gray_u8)
    lo = int(max(0, 0.67 * v))
    hi = int(min(255, 1.33 * v))
    blurred = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    edges = cv2.Canny(blurred, lo, hi)
    return edges, time.time() - t0


def shannon_entropy(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    if hist.sum() == 0:
        return 0.0
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log2(hist + 1e-7)))


def estimate_density(image_f):
    sx = cv2.Sobel(image_f, cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(image_f, cv2.CV_64F, 0, 1, ksize=5)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    rho = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    return cv2.GaussianBlur(rho, (5, 5), 0)


# ============================================================================
# Visualization (compact, 300 dpi)
# ============================================================================
def make_comparison(image_path, out_dir):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[ERR] cannot read {image_path}")
        return
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_f = gray.astype(np.float32) / 255.0

    rho_L = estimate_density(image_f)
    canny_map, t_canny = run_canny(gray)
    gwi_map, t_gwi = run_gwi_static(image_f)
    agwi_map, t_agwi = run_agwi(image_f, rho_L)

    ent_gwi = shannon_entropy(gwi_map)
    ent_agwi = shannon_entropy(agwi_map)

    # Compact 1x5 layout, tight spacing, no wasted margins  
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.0))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02,
                        wspace=0.05)

    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('(a) Original', fontsize=11, fontweight='bold')

    im = axes[1].imshow(rho_L, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(r'(b) Density $\rho_L$', fontsize=11, fontweight='bold')
    # Horizontal colorbar directly beneath the rho_L panel. It sits below
    # the axes box (outside [0,1] in axes-fraction y), so it doesn't
    # touch wspace/panel spacing; bbox_inches='tight' on savefig expands
    # the saved canvas to include it.
    cax = axes[1].inset_axes([0.1, -0.2, 0.8, 0.09])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=6)

    axes[2].imshow(canny_map, cmap='gray')
    axes[2].set_title(f'(c) Canny\n{t_canny*1000:.0f} ms',
                      fontsize=11, fontweight='bold')

    axes[3].imshow(cv2.normalize(gwi_map, None, 0, 255, cv2.NORM_MINMAX),
                   cmap='gray')
    axes[3].set_title(f'(d) GWi static\nH={ent_gwi:.2f}',
                      fontsize=11, fontweight='bold')

    axes[4].imshow(cv2.normalize(agwi_map, None, 0, 255, cv2.NORM_MINMAX),
                   cmap='gray')
    axes[4].set_title(f'(e) A-GWi\nH={ent_agwi:.2f}',
                      fontsize=11, fontweight='bold')

    for ax in axes:
        ax.axis('off')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{image_path.stem}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"[OK] {out_path}  "
          f"(GWi H={ent_gwi:.2f}, A-GWi H={ent_agwi:.2f}, "
          f"A-GWi {t_agwi*1000:.0f}ms)")
    return {'image': image_path.stem, 'ent_gwi': ent_gwi,
            'ent_agwi': ent_agwi, 't_agwi': t_agwi}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("./figures"))
    args = parser.parse_args()

    # Warmup Numba
    dummy = np.random.rand(16, 16).astype(np.float32)
    _ = run_agwi(dummy, estimate_density(dummy))

    if args.image:
        make_comparison(Path(args.image), args.output)
    elif args.image_dir:
        d = Path(args.image_dir)
        imgs = sorted([p for p in d.iterdir()
                       if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        for p in imgs:
            make_comparison(p, args.output)
    else:
        print("Provide --image or --image-dir")


if __name__ == "__main__":
    main()
