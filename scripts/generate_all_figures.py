"""
generate_all_figures.py — ALL final paper figures for A-GWi (IIETA JESA)

Generates Figures 3-7 of the A-GWi paper in one run:
  Figure 3: PR curves (BSDS500 left, UDED right) with iso-F and human baseline
  Figure 4: BSDS500 qualitative grid (rows=methods, cols=image IDs)
  Figure 5: UDED qualitative grid
  Figure 6: Density distribution (rho_L histogram + per-image heterogeneity)
  Figure 7: Dense-object adaptivity (Original, rho_L, Canny, GWi, A-GWi + entropy)

Conventions (matching 06c):
  - White background, black edges on white (cmap='gray_r')
  - Black border (1.0 pt) around each image cell
  - Serif font, 400 dpi
  - A-GWi (ours) in red, GWi in lighter red

Sample IDs:
  BSDS500: 3063, 29030, 128035, 35049
  UDED:    04-0896x4, 05-WIREFRAME-2, 28-img_043_SRF_2_HR

Method order (top to bottom in qualitative):
  Image -> Ground Truth -> Canny -> Sobel -> PC -> GWC -> GWi -> A-GWi (ours)

Author: Kukuh Yudhistiro, 2026

Usage:
  python scripts/generate_all_figures.py \
      --data-root ./data --output-root ./output \
      --eval-results ./eval_results --figures-dir ./figures

  Required inputs:
    - Edge maps in output/<dataset>/<method>/*.png  (from run_experiment.py)
    - eval_results/ods_summary.csv                  (from evaluate.py)
    - eval_results/pr_curves/<dataset>_<method>.csv  (from evaluate.py --dump-pr)
      OR the script will compute PR from edge maps if CSVs are missing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat

try:
    import cv2
    cv2.setNumThreads(1)
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from skimage.morphology import thin
    HAS_THIN = True
except ImportError:
    HAS_THIN = False


# ============================================================================
# Configuration
# ============================================================================

METHOD_ORDER = ["Canny", "Sobel", "PC", "GWC", "GWi", "AGWi"]

METHOD_LABELS = {
    "Canny": "Canny", "Sobel": "Sobel", "PC": "PC",
    "GWC": "GWC", "GWi": "GWi", "AGWi": "A-GWi\n(ours)",
}

METHOD_COLORS = {
    "AGWi":   "#D62728",   # red - proposed
    "GWi":    "#FF6B6B",   # lighter red - imaginary baseline
    "GWC":    "#FF7F0E",   # orange
    "Canny":  "#2CA02C",   # green
    "Sobel":  "#9467BD",   # purple
    "PC":     "#17BECF",   # cyan
    "Human":  "#006400",   # dark green
}

METHOD_MARKERS = {
    "AGWi": "D", "GWi": "^", "GWC": "s",
    "Canny": "v", "Sobel": "o", "PC": "p",
}

DEFAULT_SAMPLES = {
    "BSDS500": ["3063", "29030", "128035", "35049"],
    "UDED":    ["04-0896x4", "05-WIREFRAME-2", "28-img_043_SRF_2_HR"],
}

# Figure 7: dense-object comparison IDs
COMPARE_IDS = {
    "BSDS500": ["3063", "29030"],
    "UDED":    ["04-0896x4"],
}

IMAGE_SUBDIR = {"BSDS500": "images/test", "UDED": "imgs"}
GT_SUBDIR    = {"BSDS500": "groundTruth/test", "UDED": "gt"}

HUMAN_BASELINE = {
    "BSDS500": {"ods": 0.803, "source": "Arbelaez et al., TPAMI 2011"},
}

BORDER_WIDTH = 1.0
BORDER_COLOR = "black"


# ============================================================================
# Style setup
# ============================================================================

def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================================================================
# Loaders
# ============================================================================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]

def find_image_file(image_dir: Path, image_id: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def load_image_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_gt_for_display(dataset: str, gt_dir: Path,
                        image_id: str) -> Optional[np.ndarray]:
    if dataset == "BSDS500":
        gt_path = gt_dir / f"{image_id}.mat"
        if not gt_path.exists():
            return None
        mat = loadmat(str(gt_path))
        gt_struct = mat["groundTruth"]
        union = None
        for a in range(gt_struct.shape[1]):
            b = (gt_struct[0, a]["Boundaries"][0, 0] > 0).astype(np.uint8)
            union = b if union is None else (union | b)
        return (union * 255).astype(np.uint8)
    else:
        gt_path = gt_dir / f"{image_id}.png"
        if not gt_path.exists():
            return None
        arr = np.array(Image.open(gt_path).convert("L"))
        return ((arr > 127).astype(np.uint8) * 255)


def load_gt_binary_list(dataset: str, gt_dir: Path,
                        image_id: str) -> List[np.ndarray]:
    """Load GT as list of binary arrays (multiple annotators for BSDS500)."""
    if dataset == "BSDS500":
        gt_path = gt_dir / f"{image_id}.mat"
        if not gt_path.exists():
            return []
        mat = loadmat(str(gt_path))
        gt_struct = mat["groundTruth"]
        out = []
        for a in range(gt_struct.shape[1]):
            b = (gt_struct[0, a]["Boundaries"][0, 0] > 0).astype(np.uint8)
            out.append(b)
        return out
    else:
        gt_path = gt_dir / f"{image_id}.png"
        if not gt_path.exists():
            return []
        arr = np.array(Image.open(gt_path).convert("L"))
        return [((arr > 127).astype(np.uint8))]


def apply_black_border(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BORDER_COLOR)
        spine.set_linewidth(BORDER_WIDTH)


# ============================================================================
# Density map computation (matches methods.py)
# ============================================================================

def compute_density_map(image_path: Path) -> np.ndarray:
    """Compute rho_L exactly as in methods.py (Sobel + GaussianBlur)."""
    if not HAS_CV2:
        raise RuntimeError("cv2 required for density map computation")
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    norm = gray_eq.astype(np.float32) / 255.0
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=5)
    e_grad = np.sqrt(gx**2 + gy**2)
    # min-max normalize
    mn, mx = e_grad.min(), e_grad.max()
    if mx > mn:
        e_grad = (e_grad - mn) / (mx - mn)
    rho_L = cv2.GaussianBlur(e_grad, (5, 5), 0)
    return rho_L


def shannon_entropy(edge_map: np.ndarray) -> float:
    """Shannon entropy of a grayscale edge magnitude map."""
    arr = edge_map.astype(np.float64).ravel()
    arr = arr / (arr.sum() + 1e-15)
    arr = arr[arr > 0]
    return -np.sum(arr * np.log2(arr))


# ============================================================================
# PR curve computation (simplified Berkeley, from edge maps)
# ============================================================================

def compute_pr_single_image(pred_path: Path, gt_list: List[np.ndarray],
                            n_thresh: int = 99,
                            max_dist: float = 0.0075) -> Tuple[np.ndarray,
                                                                 np.ndarray]:
    """Compute precision/recall at multiple thresholds for one image."""
    pred = np.array(Image.open(pred_path).convert("L")).astype(np.float64) / 255.0
    h, w = pred.shape
    diag = np.sqrt(h**2 + w**2)
    tol = max_dist * diag

    thresholds = np.linspace(1.0 / (n_thresh + 1), n_thresh / (n_thresh + 1),
                             n_thresh)
    precisions = np.zeros(n_thresh)
    recalls = np.zeros(n_thresh)

    for ti, t in enumerate(thresholds):
        binary = (pred >= t).astype(np.uint8)
        if HAS_THIN:
            binary = thin(binary).astype(np.uint8)

        pred_pts = np.argwhere(binary > 0).astype(np.float64)

        best_p, best_r = 0.0, 0.0
        for gt in gt_list:
            gt_pts = np.argwhere(gt > 0).astype(np.float64)
            if len(pred_pts) == 0 and len(gt_pts) == 0:
                best_p, best_r = 1.0, 1.0
                break
            if len(pred_pts) == 0:
                best_r = max(best_r, 0.0)
                continue
            if len(gt_pts) == 0:
                continue

            # fast distance matching using scipy KDTree
            from scipy.spatial import cKDTree
            gt_tree = cKDTree(gt_pts)
            d_pred, _ = gt_tree.query(pred_pts)
            tp_pred = np.sum(d_pred <= tol)

            pred_tree = cKDTree(pred_pts)
            d_gt, _ = pred_tree.query(gt_pts)
            tp_gt = np.sum(d_gt <= tol)

            p = tp_pred / max(len(pred_pts), 1)
            r = tp_gt / max(len(gt_pts), 1)
            if p + r > best_p + best_r:
                best_p, best_r = p, r

        precisions[ti] = best_p
        recalls[ti] = best_r

    return recalls, precisions


def compute_or_load_pr(dataset: str, method: str, data_root: Path,
                       output_root: Path, eval_results: Path,
                       n_thresh: int = 99) -> Optional[pd.DataFrame]:
    """Load precomputed PR CSV, or compute from edge maps."""
    pr_dir = eval_results / "pr_curves"
    pr_path = pr_dir / f"{dataset}_{method}.csv"

    if pr_path.exists():
        return pd.read_csv(pr_path)

    # Compute from edge maps
    print(f"    Computing PR for {dataset}/{method} ({n_thresh} thresholds)...")
    gt_dir = data_root / dataset / GT_SUBDIR[dataset]
    edge_dir = output_root / dataset / method
    if not edge_dir.exists():
        print(f"    [WARN] edge dir missing: {edge_dir}")
        return None

    edge_files = sorted(edge_dir.glob("*.png"))
    if not edge_files:
        return None

    all_r = np.zeros(n_thresh)
    all_p = np.zeros(n_thresh)
    count = 0

    for ef in edge_files:
        image_id = ef.stem
        gt_list = load_gt_binary_list(dataset, gt_dir, image_id)
        if not gt_list:
            continue
        r, p = compute_pr_single_image(ef, gt_list, n_thresh)
        all_r += r
        all_p += p
        count += 1
        if count % 50 == 0:
            print(f"      {count}/{len(edge_files)}")

    if count == 0:
        return None

    all_r /= count
    all_p /= count

    df = pd.DataFrame({"recall": all_r, "precision": all_p})
    pr_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pr_path, index=False)
    print(f"    Saved: {pr_path} ({count} images)")
    return df


# ============================================================================
# Figure 3: PR curves (BSDS500 + UDED side by side)
# ============================================================================

def draw_iso_f(ax):
    """Draw iso-F contours as light gray dotted lines."""
    for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r = np.linspace(f / (2 - f) + 1e-3, 1.0, 200)
        p = (f * r) / np.maximum(2 * r - f, 1e-10)
        valid = (p >= 0) & (p <= 1.0)
        ax.plot(r[valid], p[valid], color="lightgray",
                linestyle=":", linewidth=0.7, zorder=0)
        if valid.any():
            idx = np.where(valid)[0][-1]
            ax.annotate(f"F={f:.1f}",
                        xy=(r[idx] * 0.97, p[idx] * 0.96),
                        color="gray", fontsize=5.5, alpha=0.85)


def figure_pr_curves(data_root: Path, output_root: Path,
                     eval_results: Path, figures_dir: Path,
                     ods_summary: pd.DataFrame) -> bool:
    """Figure 3: side-by-side PR curves for BSDS500 and UDED."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax_idx, dataset in enumerate(["BSDS500", "UDED"]):
        ax = axes[ax_idx]
        draw_iso_f(ax)

        for method in METHOD_ORDER:
            pr_df = compute_or_load_pr(dataset, method, data_root,
                                       output_root, eval_results)
            if pr_df is None:
                continue

            color = METHOD_COLORS.get(method, "#333333")
            ods_row = ods_summary[(ods_summary["dataset"] == dataset) &
                                   (ods_summary["method"] == method)]
            ods_val = ods_row["ods_f"].values[0] if not ods_row.empty else 0

            is_ours = method == "AGWi"
            is_gwi = method == "GWi"

            if is_ours:
                lw, ls, zorder = 2.2, "-", 4
                label = f"[F={ods_val:.3f}] A-GWi (ours)"
            elif is_gwi:
                lw, ls, zorder = 1.4, "-.", 3
                label = f"[F={ods_val:.3f}] GWi"
            else:
                lw, ls, zorder = 1.2, "--", 2
                label = f"[F={ods_val:.3f}] {METHOD_LABELS.get(method, method).split(chr(10))[0]}"

            ax.plot(pr_df["recall"], pr_df["precision"],
                    color=color, linewidth=lw, linestyle=ls,
                    label=label, zorder=zorder)

        # Human baseline (BSDS500 only)
        if dataset in HUMAN_BASELINE:
            h = HUMAN_BASELINE[dataset]["ods"]
            ax.scatter([h], [h], s=80, color="#006400", marker="o",
                       edgecolor="black", linewidth=1.2,
                       label=f"[F={h:.3f}] Human", zorder=5)
            ax.annotate("Human", xy=(h, h),
                        xytext=(h - 0.12, h + 0.02),
                        fontsize=8, color="#006400", weight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(dataset)
        ax.legend(loc="lower left", fontsize=6.5, framealpha=0.95,
                  handlelength=2.0, handletextpad=0.5)

    plt.tight_layout()
    out = figures_dir / "fig3_pr_curves.png"
    plt.savefig(out, dpi=400, bbox_inches="tight",
                facecolor="white", pad_inches=0.02)
    plt.close()
    print(f"[OK] Figure 3: {out}")
    return True


# ============================================================================
# Figure 4 & 5: Qualitative grids per dataset
# ============================================================================

def figure_qualitative(dataset: str, data_root: Path,
                       output_root: Path, sample_ids: List[str],
                       figures_dir: Path, fig_num: int) -> bool:
    image_dir = data_root / dataset / IMAGE_SUBDIR[dataset]
    gt_dir = data_root / dataset / GT_SUBDIR[dataset]

    if not image_dir.exists() or not gt_dir.exists():
        print(f"[WARN] {dataset}: paths missing ({image_dir}, {gt_dir})")
        return False

    methods = METHOD_ORDER
    row_labels = ["Image", "Ground Truth"] + [
        METHOD_LABELS.get(m, m) for m in methods
    ]
    n_rows = len(row_labels)
    n_cols = len(sample_ids)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.0, n_rows * 1.3),
                             squeeze=False)

    for c, image_id in enumerate(sample_ids):
        # Row 0: source image (grayscale)
        img_path = find_image_file(image_dir, image_id)
        if img_path is not None:
            try:
                img = load_image_gray(img_path)
                axes[0, c].imshow(img, cmap="gray", vmin=0, vmax=255,
                                  aspect="auto")
            except Exception as e:
                print(f"  [WARN] {dataset}/{image_id}: {e}")

        # Row 1: GT (white bg, black edges)
        gt = load_gt_for_display(dataset, gt_dir, image_id)
        if gt is not None:
            axes[1, c].imshow(gt, cmap="gray_r", vmin=0, vmax=255,
                              aspect="auto")

        # Rows 2+: edge maps (white bg, black edges)
        for r, method in enumerate(methods, start=2):
            edge_path = output_root / dataset / method / f"{image_id}.png"
            if edge_path.exists():
                edge = np.array(Image.open(edge_path).convert("L"))
                axes[r, c].imshow(edge, cmap="gray_r", vmin=0, vmax=255,
                                  aspect="auto")
            else:
                axes[r, c].imshow(np.ones((100, 100)) * 255,
                                  cmap="gray", vmin=0, vmax=255,
                                  aspect="auto")
                print(f"  [WARN] missing: {edge_path}")

        # Black borders + remove ticks
        for r in range(n_rows):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            apply_black_border(axes[r, c])

    # Row labels
    for r, label in enumerate(row_labels):
        is_ours = "A-GWi" in label or "ours" in label.lower()
        is_gwi = label == "GWi"
        if is_ours:
            color, weight = METHOD_COLORS["AGWi"], "bold"
        elif is_gwi:
            color, weight = "#666666", "normal"
        else:
            color, weight = "black", "normal"
        axes[r, 0].set_ylabel(label, rotation=0, fontsize=13,
                              labelpad=6, ha="right", va="center",
                              color=color, weight=weight)

    plt.subplots_adjust(left=0.15, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.015, hspace=0.015)
    out = figures_dir / f"fig{fig_num}_qualitative_{dataset}.png"
    plt.savefig(out, dpi=400, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close()
    print(f"[OK] Figure {fig_num}: {out}")
    return True


# ============================================================================
# Figure 6: Density distribution analysis
# ============================================================================

def figure_density_distribution(data_root: Path, figures_dir: Path) -> bool:
    """Density distribution: rho_L histogram + per-image std scatter."""
    if not HAS_CV2:
        print("[WARN] Figure 6 requires cv2")
        return False

    image_dir = data_root / "BSDS500" / IMAGE_SUBDIR["BSDS500"]
    if not image_dir.exists():
        print(f"[WARN] BSDS500 image dir missing: {image_dir}")
        return False

    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_files:
        print("[WARN] no images found in BSDS500")
        return False

    all_rho_values = []
    per_image_mean = []
    per_image_std = []

    print(f"  Computing rho_L for {len(image_files)} images...")
    for i, img_path in enumerate(image_files):
        rho = compute_density_map(img_path)
        all_rho_values.append(rho.ravel())
        per_image_mean.append(rho.mean())
        per_image_std.append(rho.std())
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(image_files)}")

    all_rho = np.concatenate(all_rho_values)
    per_image_mean = np.array(per_image_mean)
    per_image_std = np.array(per_image_std)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: rho_L value histogram
    ax = axes[0]
    ax.hist(all_rho, bins=100, density=True, color="#4C72B0", alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel(r"$\rho_L$ value")
    ax.set_ylabel("Density (normalized)")
    ax.set_title("Distribution of local gradient density across BSDS500")
    ax.axvline(np.median(all_rho), color="red", linestyle="--", linewidth=1.2,
               label=f"Median = {np.median(all_rho):.3f}")
    ax.axvline(np.mean(all_rho), color="orange", linestyle="-.", linewidth=1.2,
               label=f"Mean = {np.mean(all_rho):.3f}")
    ax.legend(fontsize=7)

    # Right: per-image mean vs std scatter (heterogeneity)
    ax = axes[1]
    ax.scatter(per_image_mean, per_image_std, s=12, alpha=0.6,
               color="#D62728", edgecolors="none")
    ax.set_xlabel(r"Per-image mean $\rho_L$")
    ax.set_ylabel(r"Per-image std $\rho_L$")
    ax.set_title("Within-image density heterogeneity (BSDS500)")
    # Add annotation for heterogeneity
    high_het = per_image_std > np.percentile(per_image_std, 90)
    ax.scatter(per_image_mean[high_het], per_image_std[high_het],
               s=18, alpha=0.8, color="#D62728", edgecolors="black",
               linewidths=0.5, label=f"Top 10% heterogeneity (n={high_het.sum()})")
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    out = figures_dir / "fig6_density_distribution.png"
    plt.savefig(out, dpi=400, bbox_inches="tight",
                facecolor="white", pad_inches=0.02)
    plt.close()
    print(f"[OK] Figure 6: {out}")
    return True


# ============================================================================
# Figure 7: Dense-object adaptivity (compare_agwi style)
# ============================================================================

def figure_dense_object_comparison(data_root: Path, output_root: Path,
                                   figures_dir: Path) -> bool:
    """Single-image comparison: Original, rho_L, Canny, GWi, A-GWi + entropy."""
    if not HAS_CV2:
        print("[WARN] Figure 7 requires cv2")
        return False

    # Collect all comparison images
    all_ids = []
    for dataset, ids in COMPARE_IDS.items():
        image_dir = data_root / dataset / IMAGE_SUBDIR[dataset]
        for image_id in ids:
            img_path = find_image_file(image_dir, image_id)
            if img_path:
                all_ids.append((dataset, image_id, img_path))
            else:
                print(f"  [WARN] Figure 7: {dataset}/{image_id} not found")

    if not all_ids:
        print("[WARN] Figure 7: no images found")
        return False

    n_rows = len(all_ids)
    col_labels = ["Original", r"$\rho_L$", "Canny", "GWi (static)", "A-GWi (ours)"]
    n_cols = len(col_labels)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.2, n_rows * 1.8),
                             squeeze=False)

    for r, (dataset, image_id, img_path) in enumerate(all_ids):
        # Col 0: Original
        gray = load_image_gray(img_path)
        axes[r, 0].imshow(gray, cmap="gray", vmin=0, vmax=255, aspect="auto")

        # Col 1: rho_L density map (jet colormap)
        rho = compute_density_map(img_path)
        axes[r, 1].imshow(rho, cmap="jet", vmin=0, vmax=1, aspect="auto")

        # Col 2: Canny
        canny_path = output_root / dataset / "Canny" / f"{image_id}.png"
        if canny_path.exists():
            canny_edge = np.array(Image.open(canny_path).convert("L"))
            axes[r, 2].imshow(canny_edge, cmap="gray_r", vmin=0, vmax=255,
                              aspect="auto")

        # Col 3: GWi (static) + entropy
        gwi_path = output_root / dataset / "GWi" / f"{image_id}.png"
        if gwi_path.exists():
            gwi_edge = np.array(Image.open(gwi_path).convert("L"))
            axes[r, 3].imshow(gwi_edge, cmap="gray_r", vmin=0, vmax=255,
                              aspect="auto")
            h_gwi = shannon_entropy(gwi_edge)
            axes[r, 3].text(0.02, 0.02, f"H={h_gwi:.2f}",
                            transform=axes[r, 3].transAxes,
                            fontsize=7, color="red", weight="bold",
                            va="bottom", ha="left",
                            bbox=dict(facecolor="white", alpha=0.8,
                                      edgecolor="none", pad=1))

        # Col 4: A-GWi + entropy
        agwi_path = output_root / dataset / "AGWi" / f"{image_id}.png"
        if agwi_path.exists():
            agwi_edge = np.array(Image.open(agwi_path).convert("L"))
            axes[r, 4].imshow(agwi_edge, cmap="gray_r", vmin=0, vmax=255,
                              aspect="auto")
            h_agwi = shannon_entropy(agwi_edge)
            axes[r, 4].text(0.02, 0.02, f"H={h_agwi:.2f}",
                            transform=axes[r, 4].transAxes,
                            fontsize=7, color="red", weight="bold",
                            va="bottom", ha="left",
                            bbox=dict(facecolor="white", alpha=0.8,
                                      edgecolor="none", pad=1))

        # Row label (dataset/id)
        axes[r, 0].set_ylabel(f"{dataset}\n{image_id}",
                              rotation=0, fontsize=7, labelpad=8,
                              ha="right", va="center")

        # Black borders + remove ticks
        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            apply_black_border(axes[r, c])

    # Column titles (top row only)
    for c, label in enumerate(col_labels):
        is_ours = "ours" in label.lower()
        color = METHOD_COLORS["AGWi"] if is_ours else "black"
        weight = "bold" if is_ours else "normal"
        axes[0, c].set_title(label, fontsize=10, color=color, weight=weight)

    plt.subplots_adjust(left=0.12, right=0.995, top=0.94, bottom=0.005,
                        wspace=0.015, hspace=0.015)
    out = figures_dir / "fig7_dense_object_comparison.png"
    plt.savefig(out, dpi=400, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close()
    print(f"[OK] Figure 7: {out}")
    return True


# ============================================================================
# Main driver
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate ALL final figures for A-GWi paper (IIETA JESA)")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root containing BSDS500/ and UDED/")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root containing edge maps: output/<dataset>/<method>/")
    parser.add_argument("--eval-results", type=Path, required=True,
                        help="Dir with ods_summary.csv (and optional pr_curves/)")
    parser.add_argument("--figures-dir", type=Path, default=Path("./figures"),
                        help="Output directory for all figures")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip specific figures, e.g. --skip 3 6")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    skip = set(str(s) for s in args.skip)

    # Load ODS summary (needed for PR curve legend)
    ods_path = args.eval_results / "ods_summary.csv"
    if ods_path.exists():
        ods_summary = pd.read_csv(ods_path)
    else:
        print(f"[WARN] ODS summary not found: {ods_path}")
        ods_summary = pd.DataFrame(columns=["dataset", "method", "ods_f",
                                             "ois_f", "ap"])

    results = {}

    # --- Figure 3: PR curves ---
    if "3" not in skip:
        print(f"\n{'='*60}")
        print("  Figure 3: PR curves (BSDS500 + UDED)")
        print(f"{'='*60}")
        results["fig3"] = figure_pr_curves(
            args.data_root, args.output_root,
            args.eval_results, args.figures_dir, ods_summary)

    # --- Figure 4: BSDS500 qualitative ---
    if "4" not in skip:
        print(f"\n{'='*60}")
        print("  Figure 4: BSDS500 qualitative")
        print(f"{'='*60}")
        results["fig4"] = figure_qualitative(
            "BSDS500", args.data_root, args.output_root,
            DEFAULT_SAMPLES["BSDS500"], args.figures_dir, fig_num=4)

    # --- Figure 5: UDED qualitative ---
    if "5" not in skip:
        print(f"\n{'='*60}")
        print("  Figure 5: UDED qualitative")
        print(f"{'='*60}")
        results["fig5"] = figure_qualitative(
            "UDED", args.data_root, args.output_root,
            DEFAULT_SAMPLES["UDED"], args.figures_dir, fig_num=5)

    # --- Figure 6: Density distribution ---
    if "6" not in skip:
        print(f"\n{'='*60}")
        print("  Figure 6: Density distribution")
        print(f"{'='*60}")
        results["fig6"] = figure_density_distribution(
            args.data_root, args.figures_dir)

    # --- Figure 7: Dense-object comparison ---
    if "7" not in skip:
        print(f"\n{'='*60}")
        print("  Figure 7: Dense-object adaptivity")
        print(f"{'='*60}")
        results["fig7"] = figure_dense_object_comparison(
            args.data_root, args.output_root, args.figures_dir)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        status = "OK" if v else "MISSING/FAILED"
        print(f"  {k}: {status}")
    print(f"\n  All figures saved to: {args.figures_dir}")
    print(f"  Figures use white background (black edges on white), 400 dpi.")


if __name__ == "__main__":
    main()
