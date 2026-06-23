# A-GWi Experiment Procedure

> **Paper:** Adaptive Imaginary Gabor Wavelet (A-GWi), standalone paper
> **Target:** IIETA JESA
> **Algorithm:** f0-based adaptive imaginary Gabor (no quantization, no CNN, no ODPS)
> **Datasets:** BSDS500 (200 test) + UDED (30)
> **Hardware:** Intel Core i5-14400, 16GB RAM, RTX 3050 6GB (GPU unused), Windows 11

> **Note on paper elements:** Each step below is tagged with the exact table or
> figure it fills in `A-GWi_paper_JESA.docx`. Use these tags when transferring
> results into the paper.

---

## PAPER ELEMENT MAP (quick reference)

| Paper element (in A-GWi_paper_JESA.docx) | Produced by | Status |
|------------------------------------------|-------------|--------|
| Figure 1 (flowchart GWC vs GWi) | drawn manually | to draw |
| Figure 2 (flowchart A-GWi six stages) | drawn manually | to draw |
| Table 2 (BSDS500 ODS/OIS/AP) | run_experiment.py + evaluate.py | FILLED |
| Table 3 (UDED ODS/OIS/AP) | run_experiment.py + evaluate.py | FILLED |
| Figure 3 (PR curves BSDS500 + UDED) | PR-curve plot (06c-style) | to fill |
| Figure 4 (BSDS500 qualitative) | generate_final_figures.py | to fill |
| Figure 5 (UDED qualitative) | generate_final_figures.py | to fill |
| Table 4 (runtime) | run_experiment.py -> runtime.csv | to fill |
| Table 5 (ablation k_s and f_max) | run_experiment.py + evaluate.py | to fill |
| Table 6 (ablation density estimator) | run_experiment.py + evaluate.py | to fill |
| Figure 6 (density distribution) | make_figures.py | to fill |
| Table 7 (BSDS500 stratified ODS) | evaluate_stratified.py | FILLED |
| Table 8 (UDED stratified ODS) | evaluate_stratified.py | FILLED |
| Figure 7 (compare_agwi on dense objects) | compare_agwi.py | to fill |

---

## ENVIRONMENT SETUP (do first)

```bash
pip install -r requirements.txt
```

`numba` is required (A-GWi ~50x faster). `phasepack` is required for the PC baseline.

Verified A-GWi runtime on i5-14400, single-thread, kernel 7x7, 8 orientations:
- 481x321 (BSDS500): ~5-7 sec/image
- 200 BSDS500 images: ~18-23 min
- UDED (~30 images): ~3 min
- Baselines (Canny, Sobel, GWi, GWC, PC): a few seconds total per method

A-GWi is heavier than static GWi because it generates a kernel per pixel.
This is expected and framed honestly in the paper (cost for spatial adaptivity).

---

## FIXED PARAMETERS (do not change)

```
A-GWi (f0 formulation, matches proven experiment + Paper 2 draft):
  f_min          = 0.05     (sparse -> low freq, broad structure)
  f_max          = 0.45     (dense  -> high freq, fine detail)
  k_steepness    = 25.0     (sigmoid sharpness)
  n_orientations = 8        (22.5 deg intervals)
  aspect (gamma) = 0.5      (sigma_y = sigma_x / aspect)
  kernel_size    = 7        (fixed 7x7)
  sigma          = (1/(pi*f0)) * sqrt(ln2/2)   [Gabor bandwidth relation]
  density (rho_L)= Sobel(ksize=5) + GaussianBlur(5x5)
  preprocessing  = BT.601 grayscale + histogram equalization + /255
  magnitude      = max_k |response|   (imaginary-only, abs, max-pool)

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

## DAY 1: Setup + A-GWi Validation

1. `pip install -r requirements.txt` (confirm numba + phasepack present)
2. Confirm BSDS500 and UDED datasets in `data/` (layout below)
3. Validate A-GWi on a few images:
   ```bash
   python scripts/compare_agwi.py --image data/BSDS500/images/test/<id>.jpg --output ./figures
   ```
   Check A-GWi produces sharp boundaries in dense regions, clean in sparse.
4. Capture environment:
   ```bash
   python scripts/capture_env.py --output ./
   ```

Data layout:
```
data/
  BSDS500/images/test/*.jpg
  BSDS500/groundTruth/test/*.mat
  UDED/imgs/*.jpg
  UDED/gt/*.png
```

Deliverables: numba working, A-GWi validated visually, env_info.json saved.

---

## DAY 2: Generate Edge Maps (all 6 methods)
### -> Produces runtime.csv (for Table 4) and edge maps (for Tables 2, 3, 7, 8 and Figures 3, 4, 5)

Dry run first (5 images):
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

Output: `output/<dataset>/<method>/*.png` + `runtime_logs/runtime.csv`.
Estimated wall time: ~25-30 min (A-GWi dominates).

Deliverables: all edge maps generated, runtime logged, results backed up.

---

## DAY 3: Berkeley Evaluation (ODS / OIS / AP)
### -> Fills Table 2 (BSDS500) and Table 3 (UDED)

```bash
python scripts/evaluate.py --data-root ./data --output-root ./output \
    --results-dir ./eval_results \
    --datasets BSDS500 UDED \
    --methods AGWi GWi GWC Canny Sobel PC \
    --n-thresholds 99 --max-dist 0.0075
```

Output: `eval_results/ods_summary.csv`.

**Confirmed results (latest run):**
```
BSDS500: AGWi 0.534 (best overall) > PC 0.509 > Sobel 0.464 > Canny 0.444 > GWi 0.406 > GWC 0.392
UDED:    Canny 0.704 > AGWi 0.668 (best multi-orientation) > Sobel 0.637 > PC 0.588 > GWi 0.565 > GWC 0.502
```

Verify these match the paper Tables 2 and 3.

Deliverables: ODS/OIS/AP confirming A-GWi best overall on BSDS500.

---

## DAY 4: Density-Stratified ODS (KEY EVIDENCE)
### -> Fills Table 7 (BSDS500) and Table 8 (UDED)

This is the strongest quantitative support for the adaptivity claim. It REUSES
the edge maps from Day 2 (no regeneration). It splits Berkeley matching into
LOW / MID / HIGH density strata.

```bash
python scripts/evaluate_stratified.py --data-root ./data --output-root ./output \
    --results-dir ./eval_results \
    --datasets BSDS500 UDED \
    --methods AGWi GWi GWC \
    --n-thresholds 99
```

Output: `eval_results/ods_stratified.csv`.

**Confirmed results (latest run):**
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

These values are already in Tables 7 and 8 of the paper.

Deliverables: stratified ODS confirming gain in LOW/MID on BSDS500, all strata on UDED.

---

## DAY 5: Runtime, Figures, Ablation

### 5a. Runtime  ->  fills Table 4
The runtime.csv from Day 2 already contains all timing. Summarize mean and std per method:
```bash
python scripts/summarize_runtime.py --runtime-csv runtime_logs/runtime.csv --dataset BSDS500
```
(If you re-run only for timing, write to a separate output folder so the Day 2
edge maps are not overwritten:)
```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output_timing \
    --datasets BSDS500 --methods AGWi GWi GWC Canny Sobel PC \
    --runtime-csv runtime_logs/runtime.csv
```
Deliverable: mean +/- std per method -> Table 4.

### 5b. Qualitative figures  ->  fills Figure 4 (BSDS500) and Figure 5 (UDED)
```bash
python scripts/generate_final_figures.py --data-root ./data --output-root ./output \
    --figures-dir ./figures --datasets BSDS500 UDED
```
Uses the fixed IDs (BSDS500: 3063, 29030, 128035, 35049; UDED: 04-0896x4,
05-WIREFRAME-2, 28-img_043_SRF_2_HR). White background (black edges on white).
Deliverable: qualitative_BSDS500.png -> Figure 4, qualitative_UDED.png -> Figure 5.

### 5c. Density distribution  ->  fills Figure 6
```bash
python scripts/make_figures.py --data-root ./data --output-root ./output \
    --figures-dir ./figures --dataset BSDS500 \
    --image-ids 3063 29030 128035
```
Deliverable: density_distribution_BSDS500.png -> Figure 6.

### 5d. Dense-object adaptivity  ->  fills Figure 7 (end of Section 5.7)
```bash
python scripts/compare_agwi.py --image-dir ./dense_images --output ./figures
```
Use density-varying images (corn cluster, rose, geese, zebra). Reports Shannon entropy.
Deliverable: compare_<id>.png -> Figure 7.

### 5e. Ablation 1: frequency range f_max  ->  fills Table 5 (f_max rows)
```bash
for FMAX in 0.25 0.35 0.45 0.55; do
  python scripts/run_experiment.py --data-root ./data --output-root ./output_ablation \
    --datasets BSDS500 --methods AGWi --f-max $FMAX \
    --agwi-name "AGWi_fmax${FMAX}"
done
python scripts/evaluate.py --data-root ./data --output-root ./output_ablation \
    --results-dir ./eval_results --datasets BSDS500 \
    --methods AGWi_fmax0.25 AGWi_fmax0.35 AGWi_fmax0.45 AGWi_fmax0.55
```
Deliverable: ODS vs f_max -> Table 5 (f_max part).

### 5f. Ablation 2: sigmoid steepness k_s  ->  fills Table 5 (k_s rows)
```bash
for KS in 5 10 15 20 25 30; do
  python scripts/run_experiment.py --data-root ./data --output-root ./output_ablation \
    --datasets BSDS500 --methods AGWi --k-steepness $KS \
    --agwi-name "AGWi_ks${KS}"
done
python scripts/evaluate.py --data-root ./data --output-root ./output_ablation \
    --results-dir ./eval_results --datasets BSDS500 \
    --methods AGWi_ks5 AGWi_ks10 AGWi_ks15 AGWi_ks20 AGWi_ks25 AGWi_ks30
```
Deliverable: ODS vs k_s -> Table 5 (k_s part).

### 5g. Ablation 3: density estimator  ->  fills Table 6
```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output_ablation \
    --datasets BSDS500 --methods AGWi --density sobel --agwi-name AGWi_sobel
python scripts/run_experiment.py --data-root ./data --output-root ./output_ablation \
    --datasets BSDS500 --methods AGWi --density variance --agwi-name AGWi_variance
python scripts/evaluate.py --data-root ./data --output-root ./output_ablation \
    --results-dir ./eval_results --datasets BSDS500 \
    --methods AGWi_sobel AGWi_variance
```
Note: local-entropy estimator is listed in the paper as a third option. If not
implemented in methods.py, either add it or remove "local entropy" from Table 6
caption to keep the claim accurate.
Deliverable: ODS vs density estimator -> Table 6.

### 5h. PR curves  ->  fills Figure 3
Generate precision-recall curves for all six methods on BSDS500 and UDED, with
iso-F contours and the BSDS500 human baseline (ODS=0.803). Use the per-threshold
precision/recall already computed inside evaluate.py (extend it to dump the PR
arrays, or add a small plotting script).
Deliverable: pr_curves.png -> Figure 3.

---

## DAY 6: Paper Writing

Open `A-GWi_paper_JESA.docx`. Fill the remaining placeholders using the element map:
- Table 4 (runtime) <- Day 5a
- Table 5 (ablation k_s, f_max) <- Day 5e, 5f
- Table 6 (ablation density estimator) <- Day 5g
- Figure 3 (PR curves) <- Day 5h
- Figure 4 (BSDS500 qualitative) <- Day 5b
- Figure 5 (UDED qualitative) <- Day 5b
- Figure 6 (density distribution) <- Day 5c
- Figure 7 (dense-object adaptivity) <- Day 5d
- Figure 1, Figure 2 (flowcharts) <- draw manually

Already filled: Table 2, Table 3 (main ODS), Table 7, Table 8 (stratified ODS).

Deliverables: complete draft with all tables and figures filled.

---

## DAY 7: Finalize + Submit

### Polish
- Read full paper start to finish.
- No em-dashes (use commas/parentheses).
- No AI buzzwords: comprehensive, robust (unless inside a published title),
  leverage, state-of-the-art, delve, moreover, furthermore, notably.
- Check equation/figure/table numbering and citation order (sequential [1]..[26]).
- Use "edge-based boundary detection" terminology throughout.

### Pre-submission
- AI detection (GPTZero) acceptable.
- Turnitin similarity acceptable.
- Format per IIETA JESA template.
- Cover letter.
- Submit.

---

## RISK MITIGATION

### Reproducing runtime without breaking ODS
Run_experiment.py overwrites edge maps. To re-time only, set --output-root to a
separate folder (output_timing) so the validated Day 2 maps that produced the
ODS numbers are untouched. evaluate.py is NOT needed for runtime alone.

### A-GWi does not beat Canny on UDED ODS
True (Canny 0.704 vs A-GWi 0.668). A-GWi leads UDED on OIS and AP, and is best
among multi-orientation methods. On BSDS500 A-GWi is best overall. State this
honestly; do not overclaim.

### HIGH-stratum ODS on BSDS500 slightly below GWi
A-GWi 0.419 vs GWi 0.426 in the HIGH stratum on BSDS500. Framed honestly: the
adaptive gain concentrates in LOW/MID (over-detection reduction), HIGH stays
competitive; on UDED A-GWi leads every stratum.

### Runtime criticism
A-GWi is heavier than static methods (per-pixel kernels). Frame as cost for
spatial adaptivity; still CPU-only and training-free. Quantization/FPGA is
future work.

### Time overrun
- Day 2 behind: 100 BSDS images instead of 200.
- Day 5 behind: skip density-estimator ablation (Table 6), remove it from text.
- Day 6 behind: reuse more draft text verbatim.

---

## CHECKPOINTS

| Day | Checkpoint | Fills | If fails |
|-----|-----------|-------|----------|
| 1 | numba + A-GWi validated visually | - | use NumPy fallback (slow) |
| 2 | all edge maps + runtime.csv | inputs to all tables | reduce to 100 BSDS images |
| 3 | A-GWi best overall on BSDS500 | Table 2, 3 | re-check preprocessing/paths |
| 4 | stratified ODS computed | Table 7, 8 | this is key, do not skip |
| 5 | runtime + figures + ablation | Table 4,5,6; Fig 3,4,5,6,7 | skip density-estimator ablation |
| 6 | draft filled | all placeholders | reuse more draft text |
| 7 | submitted | - | push 1-2 days |

---

## FILE MAP

```
agwi_paper/
  README.md
  requirements.txt
  PAPER_DRAFT_AGWi.md            # markdown draft
  A-GWi_paper_JESA.docx          # final DOCX (Tables 2,3,7,8 filled)
  EXPERIMENT_PROCEDURE.md        # this file
  scripts/
    methods.py                   # 6 methods incl. f0-based A-GWi
    run_experiment.py            # generate edge maps + runtime.csv  (Table 4 source)
    evaluate.py                  # Berkeley ODS/OIS/AP               (Tables 2,3,5,6)
    evaluate_stratified.py       # density-stratified ODS            (Tables 7,8)
    make_figures.py              # density distribution              (Figure 6)
    generate_final_figures.py    # qualitative grids white bg        (Figures 4,5)
    compare_agwi.py              # dense-object comparison           (Figure 7)
    capture_env.py               # environment snapshot
```
