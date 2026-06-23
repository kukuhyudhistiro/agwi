# A-GWi Experiment Procedure (Updated, No Ablation)

> **Paper:** Adaptive Imaginary Gabor Wavelet (A-GWi), standalone paper
> **Target:** IIETA JESA
> **Algorithm:** f0-based adaptive imaginary Gabor (no quantization, no CNN, no ODPS)
> **Datasets:** BSDS500 (200 test) + UDED (30)
> **Hardware:** Intel Core i5-14400, 16GB RAM, Windows 11, single-threaded

---

## SCRIPT-TO-PAPER ELEMENT MAP

| Paper Element | Script | Command |
|---|---|---|
| Table 2 (BSDS500 ODS/OIS/AP) | `evaluate.py` | `python scripts/evaluate.py --data-root ./data --output-root ./output --results-dir ./eval_results --datasets BSDS500 --methods AGWi GWi GWC Canny Sobel PC --n-thresholds 99 --max-dist 0.0075` |
| Table 3 (UDED ODS/OIS/AP) | `evaluate.py` | `python scripts/evaluate.py --data-root ./data --output-root ./output --results-dir ./eval_results --datasets UDED --methods AGWi GWi GWC Canny Sobel PC --n-thresholds 99 --max-dist 0.0075` |
| Table 4 (runtime) | `summarize_runtime.py` | `python scripts/summarize_runtime.py --runtime-csv runtime_logs/runtime.csv --dataset BSDS500` |
| Table 5 (BSDS500 stratified ODS) | `evaluate_stratified.py` | `python scripts/evaluate_stratified.py --data-root ./data --output-root ./output --results-dir ./eval_results --datasets BSDS500 --methods AGWi GWi GWC --n-thresholds 99` |
| Table 6 (UDED stratified ODS) | `evaluate_stratified.py` | `python scripts/evaluate_stratified.py --data-root ./data --output-root ./output --results-dir ./eval_results --datasets UDED --methods AGWi GWi GWC --n-thresholds 99` |
| Figure 1 (flowchart GWC vs GWi) | (draw manually) | - |
| Figure 2 (flowchart A-GWi 6 stages) | (draw manually) | - |
| Figure 3 (PR curves) | `evaluate.py` (extend) | See Day 5h below |
| Figure 4 (BSDS500 qualitative) | `generate_final_figures.py` | `python scripts/generate_final_figures.py --data-root ./data --output-root ./output --figures-dir ./figures --datasets BSDS500` |
| Figure 5 (UDED qualitative) | `generate_final_figures.py` | `python scripts/generate_final_figures.py --data-root ./data --output-root ./output --figures-dir ./figures --datasets UDED` |
| Figure 6 (density distribution) | `make_figures.py` | `python scripts/make_figures.py --data-root ./data --output-root ./output --figures-dir ./figures --dataset BSDS500 --image-ids 3063 29030 128035` |
| Figure 7 (dense-object A-GWi) | `compare_agwi.py` | `python scripts/compare_agwi.py --image data/BSDS500/images/test/3063.jpg --output ./figures` |
| Edge maps (all methods) | `run_experiment.py` | `python scripts/run_experiment.py --data-root ./data --output-root ./output --datasets BSDS500 UDED --methods AGWi GWi GWC Canny Sobel PC` |
| runtime.csv | `run_experiment.py` | (produced automatically alongside edge maps) |
| env_info.json | `capture_env.py` | `python scripts/capture_env.py --output ./` |

---

## SCRIPTS INVENTORY

| Script | Purpose | Input | Output |
|---|---|---|---|
| `methods.py` | Library: 6 edge detectors incl. f0-based A-GWi | (imported) | (imported) |
| `run_experiment.py` | Generate edge map PNGs + runtime.csv | images | `output/<dataset>/<method>/*.png`, `runtime_logs/runtime.csv` |
| `evaluate.py` | Berkeley ODS/OIS/AP from edge maps vs GT | edge maps + GT | `eval_results/ods_summary.csv` |
| `evaluate_stratified.py` | Density-stratified ODS (LOW/MID/HIGH) | edge maps + GT | `eval_results/ods_stratified.csv` |
| `summarize_runtime.py` | Aggregate runtime.csv into Table 4 format | `runtime.csv` | terminal table (mean/std/min/max) |
| `generate_final_figures.py` | Qualitative grid (rows=methods, cols=IDs), white bg | edge maps + images | `figures/qualitative_BSDS500.png`, `figures/qualitative_UDED.png` |
| `make_figures.py` | Density distribution + adaptive f0/sigma maps | images | `figures/density_distribution_BSDS500.png` |
| `compare_agwi.py` | Single-image comparison (Original, rho_L, Canny, GWi, A-GWi) + Shannon entropy | 1 image | `figures/compare_<id>.png` |
| `capture_env.py` | Record Python/library versions, hardware | - | `env_info.json` |

---

## FIXED PARAMETERS (do not change)

```
A-GWi (f0 formulation):
  f_min          = 0.05
  f_max          = 0.45
  k_steepness    = 25.0
  n_orientations = 8        (22.5 deg intervals)
  aspect (gamma) = 0.5
  kernel_size    = 7        (fixed 7x7)
  sigma          = (1/(pi*f0)) * sqrt(ln2/2)
  density (rho_L)= Sobel(ksize=5) + GaussianBlur(5x5)
  preprocessing  = BT.601 grayscale + histogram equalization + /255
  magnitude      = max_k |response|

Baselines:
  GWi   : static imaginary Gabor, k=7, lambda=4, 8 orient, abs
  GWC   : static complex Gabor,  k=7, lambda=4, 8 orient, sqrt(re^2+im^2)
  Canny : median-adaptive thresholds
  Sobel : 3x3 gradient magnitude
  PC    : phasepack, 4 scales, 6 orientations

Protocol (Berkeley):
  n_thresholds = 99
  thinning     = morphological (skimage.thin)
  matching     = greedy bipartite via KDTree
  tolerance    = 0.0075 x image diagonal
  metrics      = ODS, OIS, AP

Single-thread: OMP/MKL/NUMBA threads = 1, cv2.setNumThreads(1)
```

---

## EXECUTION ORDER

### Day 1: Setup + Validation

```bash
pip install -r requirements.txt
python scripts/capture_env.py --output ./
python scripts/compare_agwi.py --image data/BSDS500/images/test/3063.jpg --output ./figures
```

Confirm numba JIT compiles, A-GWi produces sharp boundaries in dense regions.

Data layout:
```
data/
  BSDS500/images/test/*.jpg       (200 images)
  BSDS500/groundTruth/test/*.mat  (200 GT)
  UDED/imgs/*.jpg                 (30 images)
  UDED/gt/*.png                   (30 GT)
```

### Day 2: Generate Edge Maps + Runtime
### -> runtime.csv (Table 4 source), edge maps (input to all evaluations)

Dry run (5 images):
```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output \
    --datasets BSDS500 --methods AGWi GWi Canny --max-images 5
```

Full run:
```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output \
    --datasets BSDS500 UDED \
    --methods AGWi GWi GWC Canny Sobel PC
```

Output: `output/<dataset>/<method>/*.png` + `runtime_logs/runtime.csv`
Estimated wall time: ~25-30 min (A-GWi dominates).

**If you only need to re-time** (without overwriting edge maps that produced ODS):
```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output_timing \
    --datasets BSDS500 --methods AGWi GWi GWC Canny Sobel PC \
    --runtime-csv runtime_logs/runtime.csv
```

### Day 3: Berkeley Evaluation
### -> Table 2 (BSDS500) + Table 3 (UDED)

```bash
python scripts/evaluate.py --data-root ./data --output-root ./output \
    --results-dir ./eval_results \
    --datasets BSDS500 UDED \
    --methods AGWi GWi GWC Canny Sobel PC \
    --n-thresholds 99 --max-dist 0.0075
```

Output: `eval_results/ods_summary.csv`

Confirmed results:
```
BSDS500: AGWi 0.534 > PC 0.509 > Sobel 0.464 > Canny 0.444 > GWi 0.406 > GWC 0.392
UDED:    Canny 0.704 > AGWi 0.668 > Sobel 0.637 > PC 0.588 > GWi 0.565 > GWC 0.502
```

### Day 4: Density-Stratified ODS (KEY EVIDENCE)
### -> Table 5 (BSDS500) + Table 6 (UDED)

Reuses edge maps from Day 2 (no regeneration needed).

```bash
python scripts/evaluate_stratified.py --data-root ./data --output-root ./output \
    --results-dir ./eval_results \
    --datasets BSDS500 UDED \
    --methods AGWi GWi GWC \
    --n-thresholds 99
```

Output: `eval_results/ods_stratified.csv`

Confirmed results:
```
BSDS500 (t33=0.0506, t66=0.1387):
  LOW:  AGWi 0.214 > GWi 0.193 > GWC 0.156
  MID:  AGWi 0.301 > GWi 0.300 > GWC 0.247
  HIGH: GWi 0.426 > AGWi 0.419 > GWC 0.374

UDED (t33=0.0276, t66=0.0930):
  LOW:  AGWi 0.303 > GWC 0.190 > GWi 0.139
  MID:  AGWi 0.418 > GWi 0.399 > GWC 0.334
  HIGH: AGWi 0.725 > GWi 0.673 > GWC 0.633
```

### Day 5: Runtime Summary + All Figures

**5a. Runtime -> Table 4**
```bash
python scripts/summarize_runtime.py --runtime-csv runtime_logs/runtime.csv --dataset BSDS500
```

Confirmed results:
```
Sobel:  1.67 +/- 0.14 ms   (380x faster than A-GWi)
Canny:  1.45 +/- 0.69 ms   (438x faster)
GWi:   25.55 +/- 1.38 ms   (24.8x faster)
GWC:   55.89 +/- 3.08 ms   (11.3x faster)
PC:   410.07 +/- 6.49 ms   (1.5x faster)
A-GWi: 633.29 +/- 18.43 ms (reference)
```

**5b. Qualitative grids -> Figure 4 (BSDS500) + Figure 5 (UDED)**
```bash
python scripts/generate_final_figures.py --data-root ./data --output-root ./output \
    --figures-dir ./figures --datasets BSDS500 UDED
```
Image IDs: BSDS500: 3063, 29030, 128035, 35049. UDED: 04-0896x4, 05-WIREFRAME-2, 28-img_043_SRF_2_HR.
White background (black edges on white), 300 dpi.
Output: `figures/qualitative_BSDS500.png`, `figures/qualitative_UDED.png`

**5c. Density distribution -> Figure 6**
```bash
python scripts/make_figures.py --data-root ./data --output-root ./output \
    --figures-dir ./figures --dataset BSDS500 \
    --image-ids 3063 29030 128035
```
Output: `figures/density_distribution_BSDS500.png`

**5d. Dense-object adaptivity -> Figure 7 (end of Section 5.5)**
```bash
python scripts/compare_agwi.py --image data/BSDS500/images/test/3063.jpg --output ./figures
python scripts/compare_agwi.py --image data/BSDS500/images/test/29030.jpg --output ./figures
```
Reports Shannon entropy H for GWi vs A-GWi.
Output: `figures/compare_3063.png`, etc.

**5e. PR curves -> Figure 3**
Extend `evaluate.py` to dump per-threshold precision/recall arrays, then plot.
Or use a standalone plotting script reading the per-threshold results.
Output: `figures/pr_curves.png`

### Day 6: Paper Assembly

Open `A-GWi_paper_JESA.docx`. Insert figures into placeholders:
- Figure 3 <- Day 5e (PR curves)
- Figure 4 <- Day 5b (BSDS500 qualitative)
- Figure 5 <- Day 5b (UDED qualitative)
- Figure 6 <- Day 5c (density distribution)
- Figure 7 <- Day 5d (dense-object comparison)
- Figure 1, 2 <- draw manually (flowcharts)

Already filled in docx: Table 2, 3 (main ODS), Table 4 (runtime), Table 5, 6 (stratified ODS).

### Day 7: Polish + Submit

- No em-dashes (use commas, parentheses, colons).
- No AI buzzwords.
- Citations sequential [1]..[42].
- "Edge-based boundary detection" terminology.
- AI detection check (GPTZero).
- Format per IIETA JESA template.
- Cover letter, submit.

---

## RISK MITIGATION

| Risk | Mitigation |
|---|---|
| Re-timing overwrites edge maps | Use `--output-root ./output_timing` for timing-only runs |
| A-GWi does not beat Canny on UDED ODS | True (0.668 vs 0.704). A-GWi leads OIS/AP and is best multi-orientation. State honestly. |
| HIGH-stratum BSDS500 slightly below GWi | 0.419 vs 0.426. Gain concentrates in LOW/MID. On UDED, A-GWi leads every stratum. |
| Runtime criticism | Frame as cost for spatial adaptivity. CPU-only, training-free. Quantization/FPGA is future work. |
| Parameter sensitivity without ablation | Add one sentence in Limitations: "Parameter values selected empirically; systematic parameter study is future work." |
| Day 2 behind schedule | Reduce to 100 BSDS images |
| Day 5 behind schedule | Skip Figure 6 (density distribution), use remaining figures |

---

## CHECKPOINTS

| Day | Checkpoint | Paper Elements | If fails |
|---|---|---|---|
| 1 | numba + A-GWi validated | - | use NumPy fallback (slow) |
| 2 | edge maps + runtime.csv generated | input to everything | reduce to 100 BSDS images |
| 3 | ODS/OIS/AP computed | Table 2, 3 | re-check preprocessing/paths |
| 4 | stratified ODS computed | Table 5, 6 | do not skip (key evidence) |
| 5 | runtime + all figures | Table 4; Fig 3,4,5,6,7 | skip density distribution |
| 6 | draft fully assembled | all placeholders filled | reuse draft text |
| 7 | submitted | - | push 1-2 days |

---

## FILE MAP

```
agwi_paper/
  README.md
  requirements.txt
  A-GWi_paper_JESA.docx          # final DOCX (Tables 2-6 filled, no ablation)
  EXPERIMENT_PROCEDURE.md        # this file
  scripts/
    methods.py                   # 6 methods incl. f0-based A-GWi (library)
    run_experiment.py            # generate edge maps + runtime.csv
    evaluate.py                  # Berkeley ODS/OIS/AP              -> Table 2, 3
    evaluate_stratified.py       # density-stratified ODS           -> Table 5, 6
    summarize_runtime.py         # runtime.csv aggregation          -> Table 4
    generate_final_figures.py    # qualitative grids white bg       -> Figure 4, 5
    make_figures.py              # density distribution + param maps -> Figure 6
    compare_agwi.py              # single-image A-GWi comparison    -> Figure 7
    capture_env.py               # environment snapshot
```
