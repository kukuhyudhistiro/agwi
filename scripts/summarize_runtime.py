#!/usr/bin/env python3
"""
summarize_runtime.py — Aggregate runtime.csv into Table 4 (mean and std per method)
Author: Kukuh Yudhistiro, 2026

Reads runtime_logs/runtime.csv (produced by run_experiment.py) and prints
the per-method mean and standard deviation of per-image runtime on a dataset,
plus the speed ratio relative to A-GWi.

Usage:
  python summarize_runtime.py --runtime-csv runtime_logs/runtime.csv --dataset BSDS500
"""

import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["Canny", "Sobel", "PC", "GWC", "GWi", "AGWi"]
LABELS = {"AGWi": "A-GWi"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-csv", type=Path, required=True)
    ap.add_argument("--dataset", default="BSDS500")
    args = ap.parse_args()

    df = pd.read_csv(args.runtime_csv)
    df = df[(df["dataset"] == args.dataset) & (df["status"] == "OK")]
    if df.empty:
        print(f"No rows for dataset {args.dataset}")
        return

    df["runtime_ms"] = df["runtime_s"].astype(float) * 1000.0

    rows = []
    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        rows.append({
            "method": LABELS.get(m, m),
            "n": len(sub),
            "mean_ms": sub["runtime_ms"].mean(),
            "std_ms": sub["runtime_ms"].std(),
        })

    agwi_mean = next((r["mean_ms"] for r in rows if r["method"] == "A-GWi"), None)

    print(f"\nTable 4. Mean per-image runtime on {args.dataset} "
          f"(single-threaded). Speedup is relative to A-GWi.\n")
    print(f"{'Method':<8} {'n':>4} {'Mean (ms)':>12} {'Std (ms)':>10} "
          f"{'A-GWi / method':>16}")
    print("-" * 54)
    for r in rows:
        ratio = (agwi_mean / r["mean_ms"]) if (agwi_mean and r["mean_ms"] > 0) else float("nan")
        print(f"{r['method']:<8} {r['n']:>4} {r['mean_ms']:>12.1f} "
              f"{r['std_ms']:>10.1f} {ratio:>15.1f}x")

    # Markdown table for direct paste into the paper
    print("\n--- Markdown (paste into Table 4) ---\n")
    print("| Method | Mean (ms) | Std (ms) | Speedup vs A-GWi |")
    print("|--------|-----------|----------|------------------|")
    for r in rows:
        ratio = (agwi_mean / r["mean_ms"]) if (agwi_mean and r["mean_ms"] > 0) else float("nan")
        ratio_s = "1.0x (ref)" if r["method"] == "A-GWi" else f"{ratio:.1f}x"
        print(f"| {r['method']} | {r['mean_ms']:.1f} | {r['std_ms']:.1f} | {ratio_s} |")


if __name__ == "__main__":
    main()
