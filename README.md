# A-GWi: Adaptive Gabor Wavelet Imaginary

Self-contained code package for the paper:
**"Adaptive Imaginary Gabor Wavelet (A-GWi): Per-Pixel Kernel Modulation
via Local Gradient Density for Edge Detection"**

Kukuh Yudhistiro, Nova Rijati, Ruri Suko Basuki (2026)

This package is fully independent. It does not depend on any other paper's
code. All six methods, the Berkeley evaluator, and figure generation are
included here.

---

## Contents

```
agwi_paper/
  scripts/
    methods.py          # All 6 methods: A-GWi, GWi, GWC, Canny, Sobel, PC
    run_experiment.py   # Generate edge maps for all methods/datasets
    evaluate.py         # Berkeley ODS/OIS/AP (greedy bipartite + KDTree)
    make_figures.py     # Qualitative grids, density distribution, param maps
    capture_env.py      # Environment snapshot for reproducibility
  README.md             # This file
  requirements.txt      # Dependencies
```

---

## Installation

```bash
pip install -r requirements.txt
```

Critical: `numba` is required for A-GWi speed (≈50x faster). `phasepack`
is required for the Phase Congruency baseline.

---

## Data layout

Place datasets under `data/`:

```
data/
  BSDS500/
    images/test/*.jpg
    groundTruth/test/*.mat      # multi-annotator BSDS .mat
  UDED/
    imgs/*.jpg
    gt/*.png                    # single-annotator PNG
```

---

## Workflow

### 1. Capture environment (reproducibility)
```bash
python scripts/capture_env.py --output ./
```

### 2. Generate edge maps (all 6 methods)
```bash
python scripts/run_experiment.py \
    --data-root ./data --output-root ./output \
    --datasets BSDS500 UDED \
    --methods AGWi GWi GWC Canny Sobel PC
```
Single-threaded. Estimated time on i5-14400: BSDS500 (200 img) A-GWi ~18 min,
baselines ~2 min total. UDED ~3 min.

Quick dry run first:
```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output \
    --datasets BSDS500 --methods AGWi GWi Canny --max-images 5
```

### 3. Evaluate (ODS/OIS/AP)
```bash
python scripts/evaluate.py \
    --data-root ./data --output-root ./output \
    --results-dir ./eval_results \
    --datasets BSDS500 UDED \
    --methods AGWi GWi GWC Canny Sobel PC \
    --n-thresholds 99 --max-dist 0.0075
```
Output: `eval_results/ods_summary.csv`

### 4. Figures
```bash
python scripts/make_figures.py \
    --data-root ./data --output-root ./output \
    --figures-dir ./figures --dataset BSDS500 \
    --image-ids 100007 101085 3096
```

---

## Ablation studies

All A-GWi hyperparameters are CLI flags. Each ablation run writes to a
different method folder, then evaluate all together.

### Ablation 1: wavelength range (lambda_max)
```bash
for LMAX in 6 7 8 9 10; do
  python scripts/run_experiment.py --data-root ./data --output-root ./output \
    --datasets BSDS500 --methods AGWi --lambda-max $LMAX \
    --agwi-name "AGWi_lmax${LMAX}"
done
python scripts/evaluate.py --data-root ./data --output-root ./output \
    --results-dir ./eval_results --datasets BSDS500 \
    --methods AGWi_lmax6 AGWi_lmax7 AGWi_lmax8 AGWi_lmax9 AGWi_lmax10
```

### Ablation 2: sigmoid steepness (k_s)
```bash
for KS in 5 10 15 20 25; do
  python scripts/run_experiment.py --data-root ./data --output-root ./output \
    --datasets BSDS500 --methods AGWi --k-steepness $KS \
    --agwi-name "AGWi_ks${KS}"
done
```

### Ablation 3: density estimator
```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output \
    --datasets BSDS500 --methods AGWi --density sobel --agwi-name AGWi_sobel
python scripts/run_experiment.py --data-root ./data --output-root ./output \
    --datasets BSDS500 --methods AGWi --density variance --agwi-name AGWi_variance
```

---

## A-GWi parameters (defaults)

| Param | Value | Meaning |
|-------|-------|---------|
| f_min | 0.05 | sparse region frequency (broad structure) |
| f_max | 0.45 | dense region frequency (fine detail) |
| k_steepness | 25.0 | sigmoid transition sharpness |
| n_orientations | 8 | 22.5 deg intervals |
| aspect (gamma) | 0.5 | sigma_y = sigma_x / aspect |
| sigma | (1/(pi*f0))*sqrt(ln2/2) | Gabor bandwidth relation |
| kernel_size | 7 | fixed 7x7 |
| density | Sobel(5) + GaussianBlur(5) | rho_L estimation |

Adaptive logic (f0 formulation): dense (rho->1) -> high f0, small sigma (sharp); sparse (rho->0) -> low f0, large sigma (broad).

---

## Method summary

| Method | Type | Kernel | Magnitude |
|--------|------|--------|-----------|
| Canny | gradient + NMS | adaptive threshold | binary |
| Sobel | first-order gradient | 3x3 | sqrt(gx^2+gy^2) |
| PC | log-Gabor multi-scale | 4 scale, 6 orient | max moment |
| GWC | complex Gabor | k=7, lambda=4, 8 orient | sqrt(re^2+im^2) |
| GWi | imaginary Gabor | k=7, lambda=4, 8 orient | abs |
| A-GWi | adaptive imaginary Gabor (SVC) | per-pixel k,lambda | abs |

---

## Reproducibility

- All runs single-threaded (OMP/MKL/NUMBA threads = 1, cv2.setNumThreads(1))
- Numba JIT warmed up before timing
- Evaluation: 99 thresholds, morphological thinning, greedy bipartite
  matching via KDTree, tolerance 0.0075 x diagonal
- Environment captured via capture_env.py
