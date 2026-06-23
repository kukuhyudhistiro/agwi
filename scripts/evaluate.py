"""
evaluate.py — Berkeley ODS/OIS/AP evaluation (self-contained)
Author: Kukuh Yudhistiro, 2026

Self-contained Berkeley protocol evaluator. Implements per-annotator
greedy bipartite matching via KDTree (Arbelaez et al., 2011), identical
in method to standard BSDS500 evaluation:

  - prediction thinned (skimage.morphology.thin)
  - each predicted pixel matched independently against each annotator
  - a prediction is TP if matched by ANY annotator (OR)
  - total GT = sum across annotators
  - 99 thresholds, tolerance = max_dist_frac * diagonal
  - ODS: best F at single dataset-wide threshold
  - OIS: mean of per-image best F
  - AP:  mean per-image area under PR curve

This file has NO dependency on Paper 1 scripts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import KDTree
from skimage.morphology import thin as sk_thin


# ============================================================================
# GT and prediction loaders
# ============================================================================
def load_gt_bsds_mat(mat_path):
    """BSDS500 multi-annotator .mat -> list of bool arrays."""
    mat = loadmat(str(mat_path))
    gt = mat["groundTruth"]
    n_ann = gt.shape[1]
    out = []
    for i in range(n_ann):
        b = gt[0, i]["Boundaries"][0, 0]
        out.append((b > 0).astype(bool))
    return out


def load_gt_png(png_path):
    """Single-annotator PNG GT (UDED, BIPED) -> list with one bool array."""
    arr = np.array(Image.open(str(png_path)).convert("L"))
    return [(arr > 127).astype(bool)]


def load_pred_png(png_path):
    """8-bit prediction PNG -> float32 [0,1]."""
    arr = np.array(Image.open(str(png_path)).convert("L"))
    return arr.astype(np.float32) / 255.0


# ============================================================================
# Matching (greedy bipartite, Arbelaez 2011)
# ============================================================================
def match_single_annotator(pred_pts, gt_pts, tree_gt, max_dist_px):
    """Greedy bipartite match between pred pts and one annotator's GT pts."""
    n_pred = len(pred_pts)
    n_gt = len(gt_pts)
    pred_matched = np.zeros(n_pred, dtype=bool)
    if n_pred == 0 or n_gt == 0 or tree_gt is None:
        return pred_matched, 0

    candidates = tree_gt.query_ball_point(pred_pts, r=max_dist_px)
    pred_idx, gt_idx, cost = [], [], []
    for p_i, cand in enumerate(candidates):
        for g_i in cand:
            d = np.linalg.norm(pred_pts[p_i] - gt_pts[g_i])
            pred_idx.append(p_i)
            gt_idx.append(g_i)
            cost.append(d)
    if not cost:
        return pred_matched, 0

    order = np.argsort(cost)
    gt_matched = np.zeros(n_gt, dtype=bool)
    for idx in order:
        p, g = pred_idx[idx], gt_idx[idx]
        if not pred_matched[p] and not gt_matched[g]:
            pred_matched[p] = True
            gt_matched[g] = True
    return pred_matched, int(gt_matched.sum())


def evaluate_threshold(pred_bool, gt_cache, total_gt, max_dist_px):
    """One threshold: returns (count_r, sum_r, count_p, sum_p)."""
    pred_thin = sk_thin(pred_bool)
    pred_pts = np.argwhere(pred_thin)
    n_pred = len(pred_pts)
    if n_pred == 0:
        return 0, total_gt, 0, 0

    pred_matched_any = np.zeros(n_pred, dtype=bool)
    total_gt_matched = 0
    for gt_pts, tree_gt in gt_cache:
        pm, n_gt_matched = match_single_annotator(
            pred_pts, gt_pts, tree_gt, max_dist_px
        )
        pred_matched_any |= pm
        total_gt_matched += n_gt_matched
    return int(total_gt_matched), total_gt, int(pred_matched_any.sum()), n_pred


def evaluate_image(pred, gts, thresholds, max_dist_frac=0.0075):
    """Per-image multi-threshold evaluation."""
    h, w = pred.shape
    diag = np.sqrt(h * h + w * w)
    max_dist_px = max_dist_frac * diag

    gt_cache = []
    total_gt = 0
    for gt in gts:
        gt_pts = np.argwhere(sk_thin(gt))
        tree = KDTree(gt_pts) if len(gt_pts) > 0 else None
        gt_cache.append((gt_pts, tree))
        total_gt += len(gt_pts)

    results = []
    for t in thresholds:
        bmap = pred >= t
        results.append(evaluate_threshold(bmap, gt_cache, total_gt, max_dist_px))
    return results


# ============================================================================
# Aggregation
# ============================================================================
@dataclass
class Score:
    dataset: str
    method: str
    n_images: int
    ods_threshold: float
    ods_f: float
    ois_f: float
    ap: float


def aggregate(per_image_counts, thresholds, dataset, method):
    n_thresh = len(thresholds)
    eps = 1e-10

    ods = np.zeros((n_thresh, 4))
    for counts in per_image_counts:
        for ti, (cr, sr, cp, sp) in enumerate(counts):
            ods[ti] += [cr, sr, cp, sp]
    p_t = ods[:, 2] / np.maximum(ods[:, 3], eps)
    r_t = ods[:, 0] / np.maximum(ods[:, 1], eps)
    f_t = 2 * p_t * r_t / np.maximum(p_t + r_t, eps)
    ods_i = int(np.argmax(f_t))

    ois_f = []
    ap_list = []
    for counts in per_image_counts:
        a = np.array(counts, dtype=np.float64)
        p_i = a[:, 2] / np.maximum(a[:, 3], eps)
        r_i = a[:, 0] / np.maximum(a[:, 1], eps)
        f_i = 2 * p_i * r_i / np.maximum(p_i + r_i, eps)
        ois_f.append(np.max(f_i))
        order = np.argsort(r_i)
        try:
            ap_list.append(float(np.trapezoid(p_i[order], r_i[order])))
        except AttributeError:
            ap_list.append(float(np.trapz(p_i[order], r_i[order])))

    return Score(
        dataset=dataset, method=method, n_images=len(per_image_counts),
        ods_threshold=float(thresholds[ods_i]),
        ods_f=float(f_t[ods_i]),
        ois_f=float(np.mean(ois_f)),
        ap=float(np.mean(ap_list)),
    )


# ============================================================================
# Driver
# ============================================================================
DATASET_LAYOUTS = {
    "BSDS500": {"gt_subdir": "groundTruth/test", "gt_format": "mat"},
    "UDED": {"gt_subdir": "gt", "gt_format": "png"},
    "BIPED": {"gt_subdir": "edge_maps/test", "gt_format": "png"},
}


def evaluate_method_dataset(pred_dir, gt_dir, dataset, method,
                            max_dist_frac, n_thresholds, gt_format):
    thresholds = np.linspace(1.0 / (n_thresholds + 1),
                             1.0 - 1.0 / (n_thresholds + 1), n_thresholds)
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions in {pred_dir}")

    per_image = []
    n_proc, n_skip = 0, 0
    for pred_path in pred_files:
        stem = pred_path.stem
        gt_path = (gt_dir / f"{stem}.mat" if gt_format == "mat"
                   else gt_dir / f"{stem}.png")
        if not gt_path.exists():
            n_skip += 1
            continue
        gts = (load_gt_bsds_mat(gt_path) if gt_format == "mat"
               else load_gt_png(gt_path))
        pred = load_pred_png(pred_path)
        per_image.append(evaluate_image(pred, gts, thresholds, max_dist_frac))
        n_proc += 1
        if n_proc % 10 == 0:
            print(f"  [{dataset}/{method}] {n_proc}/{len(pred_files)}")

    print(f"  Processed {n_proc}, skipped {n_skip}")
    if n_proc == 0:
        raise RuntimeError(f"No images evaluated for {dataset}/{method}")
    return aggregate(per_image, thresholds, dataset, method)


def main():
    parser = argparse.ArgumentParser(
        description="Self-contained Berkeley ODS/OIS/AP evaluation."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["BSDS500", "UDED"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--n-thresholds", type=int, default=99)
    parser.add_argument("--max-dist", type=float, default=0.0075)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    all_scores = []

    for dataset in args.datasets:
        if dataset not in DATASET_LAYOUTS:
            print(f"[WARN] Unknown dataset {dataset}")
            continue
        layout = DATASET_LAYOUTS[dataset]
        gt_dir = args.data_root / dataset / layout["gt_subdir"]
        pred_root = args.output_root / dataset
        if not gt_dir.exists() or not pred_root.exists():
            print(f"[WARN] Skipping {dataset}: paths missing "
                  f"({gt_dir} or {pred_root})")
            continue

        if args.methods is None:
            method_dirs = sorted([d for d in pred_root.iterdir()
                                  if d.is_dir()
                                  and not d.name.endswith("_binary")])
        else:
            method_dirs = [pred_root / m for m in args.methods
                           if (pred_root / m).exists()]

        print(f"\n{'=' * 70}\n  Dataset: {dataset}\n{'=' * 70}")
        for method_dir in method_dirs:
            method = method_dir.name
            print(f"\n[INFO] Evaluating {dataset}/{method}")
            try:
                score = evaluate_method_dataset(
                    method_dir, gt_dir, dataset, method,
                    args.max_dist, args.n_thresholds, layout["gt_format"]
                )
                all_scores.append(score)
                print(f"  ODS={score.ods_f:.4f} OIS={score.ois_f:.4f} "
                      f"AP={score.ap:.4f}")
            except Exception as e:
                print(f"  [FAIL] {e}")

    if all_scores:
        df = pd.DataFrame([asdict(s) for s in all_scores])
        out = args.results_dir / "ods_summary.csv"
        df.to_csv(out, index=False)
        print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
        print(df[["dataset", "method", "n_images",
                  "ods_f", "ois_f", "ap"]].to_string(index=False))
        print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
