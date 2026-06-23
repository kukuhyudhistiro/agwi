"""
evaluate_stratified.py — Density-stratified ODS evaluation
Author: Kukuh Yudhistiro, 2026

Computes ODS separately for LOW / MID / HIGH local-density regions.
This is the KEY quantitative evidence that A-GWi's adaptive mechanism
improves edge detection where it matters (high-density regions), even
when global ODS gains are small.

REUSES existing edge map PNGs in output/<dataset>/<method>/. Does NOT
regenerate edge maps. Only the matching is split by density stratum.

Method:
  1. For each image, compute rho_L (Sobel gradient density, same as A-GWi).
  2. Assign each pixel to a stratum by global tertile thresholds of rho_L
     (LOW < t33 <= MID < t66 <= HIGH).
  3. During Berkeley matching, count TP/FP/FN separately per stratum,
     based on the stratum of each GT pixel (for recall) and each pred
     pixel (for precision).
  4. Aggregate per-stratum accumulators across the dataset -> ODS per stratum.

Usage:
  python evaluate_stratified.py \
      --data-root ./data --output-root ./output \
      --results-dir ./eval_results \
      --datasets BSDS500 UDED \
      --methods AGWi GWi GWC \
      --n-thresholds 99
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import KDTree
from skimage.morphology import thin as sk_thin


STRATA = ["LOW", "MID", "HIGH"]


# ============================================================================
# Loaders (same as evaluate.py)
# ============================================================================
def load_gt_bsds_mat(mat_path):
    mat = loadmat(str(mat_path))
    gt = mat["groundTruth"]
    return [(gt[0, i]["Boundaries"][0, 0] > 0).astype(bool)
            for i in range(gt.shape[1])]


def load_gt_png(png_path):
    arr = np.array(Image.open(str(png_path)).convert("L"))
    return [(arr > 127).astype(bool)]


def load_pred_png(png_path):
    arr = np.array(Image.open(str(png_path)).convert("L"))
    return arr.astype(np.float32) / 255.0


# ============================================================================
# Density map (must match methods.py estimate_density exactly)
# ============================================================================
def estimate_density(image_norm, sobel_ksize=5, blur_ksize=5):
    gx = cv2.Sobel(image_norm, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(image_norm, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    energy = np.sqrt(gx ** 2 + gy ** 2)
    rho = cv2.normalize(energy, None, 0, 1, cv2.NORM_MINMAX)
    rho = cv2.GaussianBlur(rho, (blur_ksize, blur_ksize), 0)
    return rho.astype(np.float64)


def preprocess_for_density(image_path):
    """BT.601 grayscale + HE + normalize (same as methods.py)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    b, g, r = cv2.split(img)
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    eq = cv2.equalizeHist(gray_u8)
    return eq.astype(np.float64) / 255.0


# ============================================================================
# Stratified matching
# ============================================================================
def stratum_of(rho_value, t33, t66):
    if rho_value < t33:
        return 0  # LOW
    elif rho_value < t66:
        return 1  # MID
    return 2  # HIGH


def evaluate_image_stratified(pred, gts, rho_L, t33, t66,
                              thresholds, max_dist_frac=0.0075):
    """
    Returns per-stratum accumulators:
      counts[stratum] = list over thresholds of (cntR, sumR, cntP, sumP)
    """
    h, w = pred.shape
    diag = np.sqrt(h * h + w * w)
    max_dist_px = max_dist_frac * diag

    # Pre-thin GT + build KDTree + record stratum of each GT pixel
    gt_cache = []
    sumR_stratum = [0, 0, 0]
    for gt in gts:
        gt_thin = sk_thin(gt)
        gt_pts = np.argwhere(gt_thin)
        tree = KDTree(gt_pts) if len(gt_pts) > 0 else None
        gt_strata = np.array([
            stratum_of(rho_L[r, c], t33, t66) for r, c in gt_pts
        ], dtype=np.int32) if len(gt_pts) > 0 else np.array([], dtype=np.int32)
        for s in range(3):
            sumR_stratum[s] += int(np.sum(gt_strata == s))
        gt_cache.append((gt_pts, tree, gt_strata))

    n_thresh = len(thresholds)
    counts = {s: [] for s in range(3)}

    for t in thresholds:
        pred_bin = pred >= t
        pred_thin = sk_thin(pred_bin)
        pred_pts = np.argwhere(pred_thin)
        n_pred = len(pred_pts)

        if n_pred == 0:
            for s in range(3):
                counts[s].append((0, sumR_stratum[s], 0, 0))
            continue

        # Stratum of each pred pixel
        pred_strata = np.array([
            stratum_of(rho_L[r, c], t33, t66) for r, c in pred_pts
        ], dtype=np.int32)

        pred_matched_any = np.zeros(n_pred, dtype=bool)
        gt_matched_stratum = [0, 0, 0]

        for gt_pts, tree, gt_strata in gt_cache:
            if tree is None or len(gt_pts) == 0:
                continue
            cand = tree.query_ball_point(pred_pts, r=max_dist_px)
            pi, gi, cost = [], [], []
            for p_i, cl in enumerate(cand):
                for g_i in cl:
                    d = np.linalg.norm(pred_pts[p_i] - gt_pts[g_i])
                    pi.append(p_i); gi.append(g_i); cost.append(d)
            if not cost:
                continue
            order = np.argsort(cost)
            gt_done = np.zeros(len(gt_pts), dtype=bool)
            for idx in order:
                p, g = pi[idx], gi[idx]
                if not pred_matched_any[p] and not gt_done[g]:
                    pred_matched_any[p] = True
                    gt_done[g] = True
                    gt_matched_stratum[gt_strata[g]] += 1

        # Per-stratum precision counts
        for s in range(3):
            pred_s_mask = pred_strata == s
            sumP_s = int(np.sum(pred_s_mask))
            cntP_s = int(np.sum(pred_matched_any & pred_s_mask))
            cntR_s = gt_matched_stratum[s]
            counts[s].append((cntR_s, sumR_stratum[s], cntP_s, sumP_s))

    return counts


# ============================================================================
# Aggregation per stratum
# ============================================================================
@dataclass
class StratifiedScore:
    dataset: str
    method: str
    stratum: str
    n_images: int
    ods_f: float
    ois_f: float


def aggregate_stratum(per_image_counts, thresholds, dataset, method, stratum):
    n_thresh = len(thresholds)
    eps = 1e-10
    ods = np.zeros((n_thresh, 4))
    for counts in per_image_counts:
        for ti, (cr, sr, cp, sp) in enumerate(counts):
            ods[ti] += [cr, sr, cp, sp]
    p_t = ods[:, 2] / np.maximum(ods[:, 3], eps)
    r_t = ods[:, 0] / np.maximum(ods[:, 1], eps)
    f_t = 2 * p_t * r_t / np.maximum(p_t + r_t, eps)
    ods_f = float(np.max(f_t))

    ois = []
    for counts in per_image_counts:
        a = np.array(counts, dtype=np.float64)
        p_i = a[:, 2] / np.maximum(a[:, 3], eps)
        r_i = a[:, 0] / np.maximum(a[:, 1], eps)
        f_i = 2 * p_i * r_i / np.maximum(p_i + r_i, eps)
        ois.append(np.max(f_i))
    ois_f = float(np.mean(ois)) if ois else 0.0

    return StratifiedScore(dataset, method, stratum,
                           len(per_image_counts), ods_f, ois_f)


# ============================================================================
# Driver
# ============================================================================
DATASET_LAYOUTS = {
    "BSDS500": {"img_dir": "BSDS500/images/test",
                "gt_subdir": "groundTruth/test", "gt_format": "mat",
                "img_ext": ".jpg"},
    "UDED": {"img_dir": "UDED/imgs",
             "gt_subdir": "gt", "gt_format": "png", "img_ext": ".jpg"},
}


def compute_global_tertiles(data_root, dataset, image_stems, layout):
    """Compute t33, t66 of rho_L pooled across the dataset."""
    all_vals = []
    for stem in image_stems:
        img_path = data_root / layout["img_dir"] / f"{stem}{layout['img_ext']}"
        if not img_path.exists():
            for ext in [".png", ".jpeg", ".JPG"]:
                alt = data_root / layout["img_dir"] / f"{stem}{ext}"
                if alt.exists():
                    img_path = alt
                    break
        gray = preprocess_for_density(img_path)
        rho = estimate_density(gray)
        # Subsample for speed
        all_vals.append(rho.ravel()[::10])
    pooled = np.concatenate(all_vals)
    t33 = float(np.percentile(pooled, 33.33))
    t66 = float(np.percentile(pooled, 66.67))
    return t33, t66


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["BSDS500", "UDED"])
    parser.add_argument("--methods", nargs="+",
                        default=["AGWi", "GWi", "GWC"])
    parser.add_argument("--n-thresholds", type=int, default=99)
    parser.add_argument("--max-dist", type=float, default=0.0075)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(1.0 / (args.n_thresholds + 1),
                             1.0 - 1.0 / (args.n_thresholds + 1),
                             args.n_thresholds)
    all_scores = []

    for dataset in args.datasets:
        if dataset not in DATASET_LAYOUTS:
            print(f"[WARN] Unknown dataset {dataset}")
            continue
        layout = DATASET_LAYOUTS[dataset]
        gt_dir = args.data_root / dataset / layout["gt_subdir"]
        pred_root = args.output_root / dataset

        # Image stems from first method's predictions
        first_method_dir = pred_root / args.methods[0]
        if not first_method_dir.exists():
            print(f"[WARN] {first_method_dir} missing, skip {dataset}")
            continue
        image_stems = [p.stem for p in sorted(first_method_dir.glob("*.png"))]
        print(f"\n{'='*70}\n  {dataset}: {len(image_stems)} images\n{'='*70}")

        # Global tertile thresholds + cache rho_L per image
        print("  Computing global density tertiles...")
        t33, t66 = compute_global_tertiles(
            args.data_root, dataset, image_stems, layout
        )
        print(f"  rho_L tertiles: t33={t33:.4f}, t66={t66:.4f}")

        rho_cache = {}
        gt_cache_imgs = {}
        for stem in image_stems:
            img_path = (args.data_root / layout["img_dir"]
                        / f"{stem}{layout['img_ext']}")
            if not img_path.exists():
                for ext in [".png", ".jpeg", ".JPG"]:
                    alt = (args.data_root / layout["img_dir"]
                           / f"{stem}{ext}")
                    if alt.exists():
                        img_path = alt
                        break
            gray = preprocess_for_density(img_path)
            rho_cache[stem] = estimate_density(gray)
            gt_path = (gt_dir / f"{stem}.mat" if layout["gt_format"] == "mat"
                       else gt_dir / f"{stem}.png")
            gt_cache_imgs[stem] = (
                load_gt_bsds_mat(gt_path) if layout["gt_format"] == "mat"
                else load_gt_png(gt_path)
            )

        for method in args.methods:
            method_dir = pred_root / method
            if not method_dir.exists():
                print(f"  [WARN] {method} missing")
                continue
            print(f"\n  [{method}] stratified evaluation...")

            per_image_strata = {0: [], 1: [], 2: []}
            for i, stem in enumerate(image_stems, 1):
                pred_path = method_dir / f"{stem}.png"
                if not pred_path.exists():
                    continue
                pred = load_pred_png(pred_path)
                rho_L = rho_cache[stem]
                gts = gt_cache_imgs[stem]
                counts = evaluate_image_stratified(
                    pred, gts, rho_L, t33, t66, thresholds, args.max_dist
                )
                for s in range(3):
                    per_image_strata[s].append(counts[s])
                if i % 25 == 0 or i == len(image_stems):
                    print(f"    {i}/{len(image_stems)}")

            for s in range(3):
                score = aggregate_stratum(
                    per_image_strata[s], thresholds,
                    dataset, method, STRATA[s]
                )
                all_scores.append(score)
                print(f"    {STRATA[s]:5s}: ODS={score.ods_f:.4f} "
                      f"OIS={score.ois_f:.4f}")

    if all_scores:
        df = pd.DataFrame([asdict(s) for s in all_scores])
        out = args.results_dir / "ods_stratified.csv"
        df.to_csv(out, index=False)

        # Pretty pivot table
        print(f"\n{'='*70}\nSTRATIFIED ODS SUMMARY\n{'='*70}")
        for dataset in args.datasets:
            sub = df[df["dataset"] == dataset]
            if sub.empty:
                continue
            print(f"\n### {dataset}")
            pivot = sub.pivot(index="method", columns="stratum",
                              values="ods_f")
            pivot = pivot[["LOW", "MID", "HIGH"]]
            print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
