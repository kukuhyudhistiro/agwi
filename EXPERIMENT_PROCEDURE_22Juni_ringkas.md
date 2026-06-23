# A-GWi Experiment Procedure

> **Target:** IIETA JESA | **Datasets:** BSDS500 (200) + UDED (30) | **Hardware:** i5-14400, 16GB, single-thread

---

## SCRIPT-TO-PAPER MAP

| Paper Element | Script | Command |
|---|---|---|
| Edge maps + runtime.csv | `run_experiment.py` | `python scripts/run_experiment.py --data-root ./data --output-root ./output --datasets BSDS500 UDED --methods AGWi GWi GWC Canny Sobel PC` |
| Table 2, 3 (ODS/OIS/AP) | `evaluate.py` | `python scripts/evaluate.py --data-root ./data --output-root ./output --results-dir ./eval_results --datasets BSDS500 UDED --methods AGWi GWi GWC Canny Sobel PC --n-thresholds 99 --max-dist 0.0075` |
| Table 4 (runtime) | `summarize_runtime.py` | `python scripts/summarize_runtime.py --runtime-csv runtime_logs/runtime.csv --dataset BSDS500` |
| Table 5, 6 (stratified ODS) | `evaluate_stratified.py` | `python scripts/evaluate_stratified.py --data-root ./data --output-root ./output --results-dir ./eval_results --datasets BSDS500 UDED --methods AGWi GWi GWC --n-thresholds 99` |
| **Figure 3-7 (ALL)** | **`generate_all_figures.py`** | **`python scripts/generate_all_figures.py --data-root ./data --output-root ./output --eval-results ./eval_results --figures-dir ./figures`** |
| env_info.json | `capture_env.py` | `python scripts/capture_env.py --output ./` |

### generate_all_figures.py output detail

| Output file | Paper element |
|---|---|
| `fig3_pr_curves.png` | Figure 3: PR curves (BSDS500 left, UDED right, iso-F, human baseline) |
| `fig4_qualitative_BSDS500.png` | Figure 4: qualitative grid BSDS500 (IDs 3063, 29030, 128035, 35049) |
| `fig5_qualitative_UDED.png` | Figure 5: qualitative grid UDED (IDs 04-0896x4, 05-WIREFRAME-2, 28-img_043_SRF_2_HR) |
| `fig6_density_distribution.png` | Figure 6: rho_L histogram + per-image heterogeneity scatter |
| `fig7_dense_object_comparison.png` | Figure 7: Original, rho_L, Canny, GWi, A-GWi + Shannon entropy |

Skip specific figures with `--skip 3 6` (useful during debugging).
Figure 1 (flowchart GWC vs GWi) and Figure 2 (A-GWi six stages) are drawn manually.

---

## FIXED PARAMETERS

```
A-GWi: f_min=0.05  f_max=0.45  k_steepness=25.0  orientations=8  gamma=0.5
       kernel_size=7  sigma=(1/(pi*f0))*sqrt(ln2/2)  density=Sobel(5)+GaussBlur(5)
       preprocess=BT.601 gray + equalizeHist + /255  magnitude=max|response|

GWi:   k=7  lambda=4  8 orient  abs          GWC: same but sqrt(re^2+im^2)
Canny: median-adaptive thresholds             Sobel: 3x3 gradient magnitude
PC:    phasepack, 4 scales, 6 orientations

Protocol: 99 thresholds, morphological thin, KDTree matching, tol=0.0075*diag
Thread:   OMP=1  MKL=1  NUMBA=1  cv2.setNumThreads(1)
```

---

## EXECUTION ORDER

### Day 1: Setup
```bash
pip install -r requirements.txt
python scripts/capture_env.py --output ./
python scripts/compare_agwi.py --image data/BSDS500/images/test/3063.jpg --output ./figures
```
Confirm: numba compiles, A-GWi sharp in dense regions. Data in `data/BSDS500/` and `data/UDED/`.

### Day 2: Edge Maps + Runtime (~25 min)
```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output \
    --datasets BSDS500 UDED --methods AGWi GWi GWC Canny Sobel PC
```
Re-timing only (does not overwrite validated edge maps):
```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output_timing \
    --datasets BSDS500 --methods AGWi GWi GWC Canny Sobel PC
```

### Day 3: Evaluation -> Table 2, 3
```bash
python scripts/evaluate.py --data-root ./data --output-root ./output \
    --results-dir ./eval_results --datasets BSDS500 UDED \
    --methods AGWi GWi GWC Canny Sobel PC --n-thresholds 99 --max-dist 0.0075
```

### Day 4: Stratified ODS -> Table 5, 6
```bash
python scripts/evaluate_stratified.py --data-root ./data --output-root ./output \
    --results-dir ./eval_results --datasets BSDS500 UDED \
    --methods AGWi GWi GWC --n-thresholds 99
```

### Day 5: Runtime Summary + All Figures -> Table 4, Figure 3-7
```bash
python scripts/summarize_runtime.py --runtime-csv runtime_logs/runtime.csv --dataset BSDS500
python scripts/generate_all_figures.py --data-root ./data --output-root ./output \
    --eval-results ./eval_results --figures-dir ./figures
```

### Day 6-7: Assemble + Submit
Insert figures into `A-GWi_paper_JESA.docx` placeholders. Polish (no em-dash, no buzzwords, citations [1]..[42] sequential, "edge-based boundary detection" terminology). Format per JESA template. Submit.

---

## CONFIRMED RESULTS

```
BSDS500 ODS: AGWi 0.534 > PC 0.509 > Sobel 0.464 > Canny 0.444 > GWi 0.406 > GWC 0.392
UDED ODS:    Canny 0.704 > AGWi 0.668 > Sobel 0.637 > PC 0.588 > GWi 0.565 > GWC 0.502

Stratified BSDS (t33=0.0506, t66=0.1387):  LOW  AGWi 0.214 > GWi 0.193  |  MID  AGWi 0.301  |  HIGH  GWi 0.426 > AGWi 0.419
Stratified UDED (t33=0.0276, t66=0.0930):  LOW  AGWi 0.303  |  MID  AGWi 0.418  |  HIGH  AGWi 0.725

Runtime (ms, BSDS500): Sobel 1.67  Canny 1.45  GWi 25.55  GWC 55.89  PC 410.07  A-GWi 633.29
```

---

## RISK MITIGATION

| Risk | Action |
|---|---|
| Re-timing overwrites edge maps | `--output-root ./output_timing` |
| A-GWi < Canny on UDED | State honestly; A-GWi best multi-orientation, leads OIS/AP |
| HIGH stratum BSDS slightly below GWi | Gain in LOW/MID; UDED A-GWi leads all strata |
| Runtime criticism | Cost of per-pixel adaptivity; CPU-only, training-free; quantization is future work |
| No ablation | One sentence in Limitations: "Parameters selected empirically; systematic study is future work" |
| Behind schedule | Day 2: reduce to 100 images. Day 5: `--skip 6` to skip density distribution |

---

## FILE MAP

```
agwi_paper/
  A-GWi_paper_JESA.docx             # final paper (Tables 2-6 filled)
  EXPERIMENT_PROCEDURE.md            # this file
  requirements.txt
  scripts/
    methods.py                       # 6 edge detectors incl. f0 A-GWi (library)
    run_experiment.py                # edge maps + runtime.csv
    evaluate.py                      # ODS/OIS/AP                    -> Table 2, 3
    evaluate_stratified.py           # stratified ODS                -> Table 5, 6
    summarize_runtime.py             # runtime aggregation           -> Table 4
    generate_all_figures.py          # ALL figures in one run        -> Figure 3-7
    compare_agwi.py                  # single-image quick comparison (Day 1 validation)
    capture_env.py                   # environment snapshot
```
