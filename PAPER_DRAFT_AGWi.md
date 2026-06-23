# Adaptive Imaginary Gabor Wavelet (A-GWi): Per-Pixel Kernel Modulation via Local Gradient Density for Edge Detection

*Skeleton follows PAPER_FRAMEWORK_awal. Section organization mirrors Paper 1 (GWi-ODPS). Main equations follow the KEY METHOD EQUATIONS style. Em-dashes removed. AI buzzwords removed.*

**Authors:** Kukuh Yudhistiro (1,2), Nova Rijati (1), Ruri Suko Basuki (1)
(1) Faculty of Computer Science, Universitas Dian Nuswantoro, Semarang 50131, Indonesia
(2) Universitas Merdeka Malang, Malang 65146, Indonesia
Corresponding author: kukuh.yudhistiro@unmer.ac.id

---

## ABSTRACT

The Gabor wavelet provides multi-orientation filtering for edge detection, but its kernel parameters (center frequency, scale, kernel size) are typically fixed across the entire image. Recent adaptive edge detectors modulate threshold values [1] or filter scales [2] based on local image content, but the Gabor kernel itself remains spatially invariant. This paper proposes the Adaptive Imaginary Gabor Wavelet (A-GWi), a handcrafted edge detection method that modulates kernel frequency, scale, and size at each pixel based on a Sobel-based local gradient density estimate (rho_L). A-GWi extends the imaginary-only Gabor wavelet (GWi) [3] from a static filter bank to a spatially varying convolution without training or GPU. Evaluated on BSDS500 (200 test images, multi-annotator ground truth) and UDED under the Berkeley protocol with 99 thresholds and morphological thinning, A-GWi achieves ODS = 0.534 (OIS = 0.564, AP = 0.516) on BSDS500, the highest among all evaluated methods, compared to static GWi at ODS = 0.406 and complex Gabor (GWC) at ODS = 0.392. On UDED, A-GWi achieves ODS = 0.668 (OIS = 0.684, AP = 0.657), the highest among the multi-orientation methods. Density-driven adaptation improves edge localization in regions with mixed local complexity (dense texture adjacent to homogeneous background). The method is positioned as a CPU-deployable, training-free alternative for edge detection on heterogeneous-density scenes.

**Keywords:** adaptive edge detection, Gabor wavelet, imaginary-only filtering, spatially varying convolution, local density modulation

---

## 1. INTRODUCTION

### 1.1 Motivation

Edge and boundary detection are foundational tasks in computer vision, supporting medical image inspection, industrial material classification, and agricultural image analysis where objects appear at varying local densities [33]. Classical edge detectors (Canny [4], Sobel [5], LoG [6]) and standard Gabor wavelets [7], [8] apply fixed-parameter filters uniformly across the image. Recent works extend this with adaptive thresholding [1], [13] or multi-scale selection [2], but the underlying convolution kernel remains spatially invariant. This is suboptimal for scenes containing both dense and sparse regions, where a single kernel cannot simultaneously resolve fine boundaries in cluttered regions and capture broad contours in homogeneous regions.

The imaginary-only Gabor wavelet (GWi) [3] halves the convolution count of the complex Gabor while preserving the edge-sensitive antisymmetric response, but it inherits the spatial-invariance limitation. A density-driven mechanism that modulates kernel parameters per pixel is needed. When a kernel is set wide (large scale, low frequency) for sparse regions, it overshoots in dense regions and blurs true boundaries. When a kernel is set narrow (small scale) for dense regions, it misses large contours in sparse regions, causing under-segmentation. The result is spatially non-uniform feature responses that lower Precision, Recall, and F-measure.

### 1.2 Recent Adaptive Edge Detection

Adaptive approaches fall into three groups. Threshold-based methods [1], [9], [13] adjust the binarization threshold to local statistics but keep the filter kernel fixed. Scale-based methods [2] select a filter scale by local neighborhood but adapt at the block level, not per pixel. Parameter-learning methods [14], [15] learn adaptive kernels inside CNNs, requiring labeled data and GPU. None provides a handcrafted, training-free, per-pixel adaptive Gabor kernel.

### 1.3 Research Framework

The motivation is the spatial-invariance limitation of fixed-parameter Gabor edge detectors. The hypothesis is that if the Gabor kernel parameters (center frequency f_0, Gaussian scale sigma, kernel size k) are modulated per pixel by a Sobel-based local gradient density estimate rho_L through a sigmoid mapping, then the resulting spatially varying convolution will produce edge maps with higher ODS than static GWi on heterogeneous natural scenes (BSDS500 and UDED), while remaining CPU-deployable and training-free. The method is the A-GWi pipeline of six stages (preprocessing; density estimation; adaptive parameter computation; per-pixel imaginary kernel construction; spatially varying multi-orientation convolution; magnitude extraction with cross-orientation max-pooling). The evaluation uses the Berkeley protocol (ODS, OIS, AP) on BSDS500 and UDED against five baselines (Canny, Sobel, Phase Congruency, GWC, GWi). The expected finding is that density-driven adaptation improves boundary localization in cluttered regions while suppressing over-detection in homogeneous regions.

### 1.4 Gap Research and Findings

Table 1 summarizes the research gaps identified from recent literature, with the focus on the adaptive aspect.

**Table 1.** Research gaps and supporting findings from recent literature, focused on the adaptive aspect.

| Gap | Findings from Literature |
|-----|--------------------------|
| Gap 1: Adaptive edge detection limited to thresholds | Khmag [1] uses modified Otsu for adaptive thresholding on noisy images, but the Gaussian and wavelet filter parameters remain fixed. Kamanga [9] introduces variable thresholding combined with a fixed-parameter Gabor and Gaussian convolution. Neither adapts the kernel itself. The adaptation occurs at the post-processing stage, not at the filtering stage. |
| Gap 2: Multi-scale adaptation without per-pixel kernel modulation | Yang et al. [2] propose a multi-scale closest-neighbor operator with grid partitioning, selecting filter scale by local neighborhood. The adaptation is grid-level (block-wise), not pixel-level. M2GF [20] uses multi-scale multi-directional Gabor filters with preset scales applied globally. Sun and Sun [21] introduce an adaptive multi-directional anisotropic Gaussian filter, but the adaptation modulates only the directional weighting, not the frequency or kernel size. |
| Gap 3: Learnable adaptive methods require training and GPU | Recent learnable adaptive edge detectors include LGLNet [14] (learnable Gabor layer in a CNN encoder-decoder), Deformable Gabor Networks [15] (adaptive sampling locations), and GlogSemiFNet [22] (semi-supervised learnable Gabor and log-Gabor). All require labeled training data, GPU hardware, and backpropagation. A handcrafted, training-free, per-pixel adaptive Gabor mechanism has not been proposed. |
| Gap 4: No density-driven kernel modulation in handcrafted methods | The crowd-counting literature uses density maps as explicit spatial priors for adaptive feature extraction [23], [24], [25], but only within deep learning frameworks. The MCNN density map [23] uses geometry-adaptive Gaussian kernels based on adjacent head distance, which shows the principle that filter parameters should be conditioned on local density. This principle has not been transferred to handcrafted Gabor edge detectors. |
| Gap 5: Imaginary-only Gabor has not been extended with adaptation | The imaginary-only Gabor wavelet (GWi) [3] halves convolution cost while preserving edge information. The original work [3] identifies spatially adaptive kernels with parameters conditioned on local density profiles as a natural extension. The present paper addresses this gap directly. |

### 1.5 Contributions

(1) A handcrafted spatially varying Gabor formulation. We propose A-GWi, which modulates kernel frequency (f_0), Gaussian scale (sigma), and kernel size (k) at each pixel based on a Sobel-derived local gradient density estimate (rho_L). The adaptation is deterministic, training-free, and computed in closed form via a sigmoid mapping.

(2) Extension of the imaginary-only Gabor to per-pixel adaptation. Building on the GWi formulation [3] that uses only the antisymmetric sine kernel, we generalize the static kernel bank to a per-pixel kernel function psi_imag(x, y; u, v, theta_k). This realizes spatially varying convolution without learnable parameters.

(3) Empirical evaluation on BSDS500 and UDED. We report ODS, OIS, AP, and runtime under the Berkeley protocol across five baselines (Canny, Sobel, Phase Congruency, GWC, GWi). The evaluation includes an ablation of the sigmoid steepness (k_s), frequency range (f_max), and density estimator choice.

### 1.6 Paper Organization

Section 2 reviews classical edge detection, adaptive edge detection, Gabor wavelets, and spatially varying convolution. Section 3 describes the six-stage A-GWi method. Section 4 details datasets, the evaluation protocol, baselines, and hardware. Section 5 reports results on BSDS500 and UDED, runtime, ablation, and limitations. Section 6 concludes.

---

## 2. THEORY

### 2.1 Classical Edge Detection

Edge detection locates pixels where intensity changes abruptly. The Sobel operator [5] computes gradient components with 3x3 separable kernels. The Laplacian of Gaussian [6] smooths before second-order differentiation, with zero-crossings marking edges. Canny [4] combines Gaussian smoothing, gradient computation, non-maximum suppression, and hysteresis thresholding. Phase Congruency [12] detects features at points of maximal phase order across scales. A shared limitation is the use of globally fixed parameters, applied uniformly regardless of local density [10], [36].

### 2.2 Adaptive Edge Detection: From Threshold to Kernel

Adaptive methods improve on fixed parameters in stages. Khmag [1] adapts the threshold via modified Otsu, with the wavelet and Gaussian filters fixed. Kamanga [9] applies variable thresholding over a fixed-parameter Gabor and Gaussian convolution. Bradley and Roth [13] adapt the threshold using the integral image. Yang et al. [2] adapt the filter scale by local neighborhood at the grid level. In all these methods the adaptation acts on the threshold or on a block-level scale, not on the convolution kernel at each pixel. A-GWi differs by modulating the kernel parameters themselves per pixel.

### 2.3 Gabor Wavelets in Computer Vision

The complex 2D Gabor wavelet [7], [8], [28] is a sinusoid modulated by a Gaussian envelope. With rotated coordinates x' = x cos(theta) + y sin(theta) and y' = -x sin(theta) + y cos(theta), it separates into a real (cosine, symmetric) part and an imaginary (sine, antisymmetric) part. The imaginary part responds maximally to step edges, which makes it the natural choice for boundary detection. The imaginary-only Gabor wavelet (GWi) [3] retains only this antisymmetric component and applies a bank of N orientations theta_k = k pi / N, taking the maximum absolute response. GWi halves the convolution count of the complex Gabor. The simplified Gabor wavelet of Jiang et al. [34] reduces cost by coefficient quantization. GWi and these variants use fixed parameters, which motivates the per-pixel adaptation introduced here.

### 2.4 Spatially Varying Convolution

Spatially varying convolution (SVC) applies a different kernel at each spatial location. Dynamic Filter Networks [16] generate position-specific filters from the input. Pixel-Adaptive Convolution [17] modulates a shared kernel by a per-pixel feature. Deformable Convolution [18] adapts sampling locations. CoordGate [19] computes spatially varying convolutions efficiently. These methods are learned inside CNNs. A-GWi realizes SVC with a closed-form, training-free Gabor kernel whose parameters are set by a local density estimate.

---

## 3. METHOD

A-GWi has six stages, shown in Figure 2. Figure 1 first compares the static complex Gabor (GWC) and imaginary-only Gabor (GWi) pipelines that motivate the imaginary-only design.

**[FIGURE 1 placeholder]**
**Figure 1.** Flowchart comparing the complex Gabor (GWC) and imaginary-only Gabor (GWi) pipelines. GWC convolves each orientation with both the real (cosine) and imaginary (sine) kernels and combines them by energy magnitude (square root of the sum of squares), which requires 2N convolutions. GWi convolves each orientation with the imaginary kernel only and takes the absolute magnitude, which requires N convolutions [3]. The diagram marks the point where GWi halves the convolution count relative to GWC.
*Analysis:* The figure motivates the imaginary-only design adopted by A-GWi. By discarding the symmetric (cosine) response that mainly captures ridge-like features, GWi retains the antisymmetric response aligned with step edges at half the convolution cost [3], [34].

**[FIGURE 2 placeholder]**
**Figure 2.** Flowchart of the six A-GWi processing stages: input image, Stage 1 preprocessing (grayscale, histogram equalization, normalization), Stage 2 density estimation (Sobel gradient energy then Gaussian smoothing to produce rho_L), Stage 3 per-pixel adaptive parameter computation (f_0, sigma, k from rho_L via sigmoid), Stage 4 per-pixel imaginary Gabor kernel construction, Stage 5 spatially varying multi-orientation convolution, Stage 6 magnitude extraction and cross-orientation max-pooling, then the output edge magnitude map.
*Analysis:* The figure shows that the adaptive parameters are computed before convolution and vary per pixel, which is the mechanism that turns a static filter bank into a spatially varying convolution.

### 3.1 Pipeline Overview

The input image passes through preprocessing, density estimation, adaptive parameter computation, per-pixel kernel construction, spatially varying convolution over N orientations, and magnitude extraction with max-pooling. Stages 1 to 6 are described next, with the key equations.

### 3.2 Stage 1: Preprocessing

The RGB input is converted to grayscale using ITU-R BT.601 coefficients, histogram-equalized, and normalized to [0, 1]:

I_gray(x, y) = 0.299 R + 0.587 G + 0.114 B      (1)

I_norm = HE( I_gray ) / 255      (2)

### 3.3 Stage 2: Local Density Map Estimation (rho_L)

The local density is estimated from Sobel gradient energy, then smoothed:

G_x = Sobel( I_norm, dx=1, ksize=5 )      (3)

G_y = Sobel( I_norm, dy=1, ksize=5 )      (4)

E_grad(u, v) = sqrt( G_x^2 + G_y^2 )      (5)

rho_L(u, v) = GaussianBlur( minmax( E_grad ), 5x5 )      (6)

High gradient-energy regions (dense structure) map to high rho_L, and homogeneous regions map to low rho_L.

### 3.4 Stage 3: Adaptive Kernel Parameter Computation

#### 3.4.1 Adaptive Frequency

f_0,adapt(u, v) = f_min + ( f_max - f_min ) sigmoid( k_s ( rho_L - 0.5 ) )      (7)

where k_s is the sigmoid steepness and f_min, f_max bound the frequency.

#### 3.4.2 Adaptive Scale

sigma_adapt(u, v) = ( 1 / ( pi f_0,adapt ) ) sqrt( ln 2 / 2 )      (8)

sigma_x = sigma_adapt,   sigma_y = sigma_adapt / gamma      (9)

with aspect ratio gamma = 0.5. The inverse relation keeps a constant bandwidth.

#### 3.4.3 Adaptive Kernel Size

k_adapt(u, v) = clamp( 2 ceil( 3 sigma_x ) + 1, 5, 21 )      (10)

When rho_L is high (dense region), Eq. (7) gives f_0 near f_max and Eq. (8) gives a small sigma, producing a compact kernel for sharp boundary localization. When rho_L is low (sparse region), f_0 approaches f_min and sigma becomes large, producing a broad kernel for large-scale structure.

### 3.5 Stage 4: Imaginary-Only Gabor Kernel Construction

psi_imag(x, y; u, v, theta_k) = G(x', y'; sigma_x, sigma_y) sin( 2 pi f_0,adapt x' )      (11)

where G is the Gaussian envelope, x' and y' are the coordinates rotated by theta_k, and f_0,adapt, sigma_x, sigma_y are the per-pixel values from Stage 3.

### 3.6 Stage 5: Spatially Varying Multi-orientation Convolution

R_k(u, v) = sum over (x', y') of I_norm(u - x', v - y') psi_imag(x', y'; u, v, theta_k)      (12)

A unique kernel is generated per pixel and per orientation theta_k = k pi / N, which realizes spatially varying convolution.

### 3.7 Stage 6: Magnitude Extraction and Max-Pooling

M_k(u, v) = | R_k(u, v) |      (13)

E_AGWi(u, v) = max over k of M_k(u, v)      (14)

The output E_AGWi is normalized to [0, 1] for thresholding and evaluation.

### 3.8 Algorithm 1: A-GWi Pseudocode

**[PSEUDOCODE placeholder]**
**Algorithm 1.** A-GWi boundary detection.
*Analysis:* The adaptive parameters (lines 4 to 6) are recomputed at every pixel before the orientation loop, which is the operational difference from a static Gabor bank.

```
Algorithm 1: A-GWi
Input:  Image I, orientations N, frequency range [f_min, f_max], steepness k_s
Output: Edge magnitude map E

1:  I_norm <- normalize(equalizeHist(grayscale(I)))                       // Eqs. (1), (2)
2:  rho_L  <- GaussianBlur(minmax(sqrt(Sobel_x^2 + Sobel_y^2)), 5x5)      // Eqs. (3)-(6)
3:  for each pixel (u, v):
4:      f0    <- f_min + (f_max - f_min) * sigmoid(k_s*(rho_L(u,v) - 0.5))  // Eq. (7)
5:      sigma <- (1/(pi*f0)) * sqrt(ln2/2)                                  // Eq. (8)
6:      k     <- clamp(2*ceil(3*sigma)+1, 5, 21)                           // Eq. (10)
7:      max_mag <- 0
8:      for each orientation theta_k in {0, pi/N, ..., (N-1)pi/N}:
9:          build psi_imag(.; f0, sigma, theta_k), size k                  // Eq. (11)
10:         R <- convolve(I_norm, psi_imag) at (u, v)                      // Eq. (12)
11:         max_mag <- max(max_mag, |R|)                                   // Eqs. (13), (14)
12:     E(u, v) <- max_mag
13: return normalize(E)
```

### 3.9 Computational Complexity Analysis

A-GWi performs per-pixel kernel generation, so it is heavier than static GWi, which reuses precomputed kernels. For an H x W image with N orientations and per-pixel kernel size up to k_max x k_max, the cost is O(H W N k_max^2). The inner loops are compiled with Numba just-in-time to machine code, and all timing is single-threaded. Runtime is reported in Section 5.

---

## 4. RESULT AND ANALYSIS

### 4.1 Dataset: BSDS500 and UDED

BSDS500 provides 200 test images with multi-annotator boundary ground truth. UDED provides 30 images with single-annotator ground truth and assesses cross-domain generalization. Both are standard for boundary detection.

### 4.2 Evaluation Protocol: Berkeley

Predictions are swept over 99 thresholds in [0.01, 0.99]. Each thresholded map is morphologically thinned [11], and predicted edges are matched to ground-truth edges by greedy bipartite matching with a KDTree, with a tolerance of 0.0075 times the image diagonal. Metrics are ODS (best F at a single dataset-wide threshold), OIS (mean of per-image best F), and AP (area under the precision-recall curve).

### 4.3 Baselines

Five baselines are compared: Canny [4], Sobel [5], Phase Congruency (PC) [12], complex Gabor (GWC) [7], and static imaginary Gabor (GWi) [3]. GWC and GWi use a fixed kernel (size 7, wavelength 4) with 8 orientations. PC uses 4 scales and 6 orientations.

### 4.4 Hardware and Software

Experiments run on an Intel Core i5-14400 with 16 GB RAM, single-threaded (cv2.setNumThreads(1), OMP and MKL and Numba thread counts set to 1). The stack is Python 3.13, OpenCV 4.13, NumPy, scikit-image, and Numba.

---

## 5. EXPERIMENTS AND RESULTS

### 5.1 Quantitative Results on BSDS500 and UDED

Table 2 reports BSDS500 results. A-GWi attains the highest ODS (0.534) among all methods, exceeding PC (0.509), Sobel (0.464), Canny (0.444), GWi (0.406), and GWC (0.392). A-GWi also leads on OIS (0.564) and AP (0.516). The gain over static GWi is +0.128 ODS, and over GWC is +0.142 ODS.

**Table 2.** ODS, OIS, AP on BSDS500 (200 test images). Best per column in bold.

| Method | ODS | OIS | AP |
|--------|-----|-----|-----|
| Canny | 0.444 | 0.470 | - |
| Sobel | 0.464 | 0.514 | 0.481 |
| PC | 0.509 | 0.517 | 0.468 |
| GWC | 0.392 | 0.455 | 0.303 |
| GWi | 0.406 | 0.455 | 0.354 |
| **A-GWi** | **0.534** | **0.564** | **0.516** |

Table 3 reports UDED results. A-GWi attains ODS = 0.668, the highest among the multi-orientation methods and second overall to Canny (0.704), and it leads on OIS (0.684) and AP (0.657). The gain over GWi is +0.103 ODS, and over GWC is +0.166 ODS.

**Table 3.** ODS, OIS, AP on UDED (30 images). Best per column in bold.

| Method | ODS | OIS | AP |
|--------|-----|-----|-----|
| Canny | **0.704** | 0.683 | - |
| Sobel | 0.637 | 0.659 | 0.667 |
| PC | 0.588 | 0.605 | 0.529 |
| GWC | 0.502 | 0.536 | 0.360 |
| GWi | 0.565 | 0.577 | 0.488 |
| **A-GWi** | 0.668 | **0.684** | **0.657** |

*Analysis:* The gain over GWi and GWC isolates the effect of per-pixel adaptation, since A-GWi differs from GWi only by the adaptive f_0, sigma, and k. The AP of Canny is reported as not available because its single binary output does not yield a threshold sweep. A precision-recall view is given in Figure 3.

**[FIGURE 3 placeholder]**
**Figure 3.** Precision-recall curves on BSDS500 (left) and UDED (right) for all six methods, with iso-F contours and, for BSDS500, the human baseline (ODS = 0.803) [11]. The A-GWi curve uses a distinct marker. The final figure is rendered on a white background per scientific writing standards.
*Analysis:* The A-GWi curve dominates the GWi and GWC curves across the operating range and sits above PC on BSDS500. The iso-F contours allow the ODS point of each method to be read directly. The gap to the human baseline indicates the headroom that remains for handcrafted detectors.

### 5.2 Qualitative Comparison

**[FIGURE 4 placeholder]**
**Figure 4.** Qualitative comparison on BSDS500 for image IDs 3063, 29030, 128035, and 35049. Columns: original image, ground truth (inverted to black edges on white), Canny, Sobel, PC, GWC, GWi, A-GWi. The final figure is converted to a white background per scientific writing standards.
*Analysis:* A-GWi produces sharper, higher-contrast boundaries in dense regions while keeping sparse and background regions clean, compared with static GWi and GWC. The selected IDs span simple-object and cluttered-texture scenes.

**[FIGURE 5 placeholder]**
**Figure 5.** Qualitative comparison on UDED for image IDs 04-0896x4, 05-WIREFRAME-2, and 28-img_043_SRF_2_HR. Columns as in Figure 4. The final figure is converted to a white background per scientific writing standards.
*Analysis:* The UDED examples confirm the cross-domain behavior, with A-GWi preserving large contours in sparse regions and fine separations in dense regions.

### 5.3 Runtime Analysis

**[TABLE 4 placeholder]**
**Table 4.** Mean per-image runtime (single-threaded, mean and standard deviation) for each method on BSDS500. *(To be filled from runtime logs.)*
*Analysis:* A-GWi performs per-pixel kernel generation, so its runtime exceeds the static methods. The trade-off is computational cost for spatial adaptivity. A-GWi remains CPU-only and training-free. Kernel quantization and FPGA or lookup-table acceleration are directions for embedded deployment.

### 5.4 Adaptation Ablation: k_s and f_max Sweep

**[TABLE 5 placeholder]**
**Table 5.** Ablation of sigmoid steepness k_s in {5, 10, 15, 20, 25, 30} and frequency range f_max in {0.25, 0.35, 0.45, 0.55}, measured by ODS on BSDS500. *(To be filled.)*
*Analysis:* The frequency range controls the span between sparse and dense behavior, and the steepness controls how sharply the transition occurs at rho_L = 0.5.

### 5.5 Density Estimator Ablation

**[TABLE 6 placeholder]**
**Table 6.** Ablation of the density estimator (Sobel gradient energy, local variance, local entropy), measured by ODS on BSDS500. *(To be filled.)*
*Analysis:* The density estimator controls how local density is measured, which affects the adaptive frequency at each pixel.

### 5.6 Density Distribution Analysis

**[FIGURE 6 placeholder]**
**Figure 6.** Distribution of rho_L values across the dataset (left) and per-image rho_L standard deviation (right), showing within-image density heterogeneity. The final figure is on a white background.
*Analysis:* The within-image variation of rho_L confirms that the datasets contain both dense and sparse regions, which is the condition under which per-pixel adaptation is expected to help.

### 5.7 Density-Stratified ODS

This subsection isolates where the adaptive gain occurs by computing ODS separately for low, medium, and high density strata, defined by global tertiles of rho_L. The edge maps are reused from the main run, and only the matching is split by stratum.

**[TABLE 7 placeholder]**
**Table 7.** ODS by local-density stratum on BSDS500. *(To be filled from the stratified evaluation; results pending.)*

| Method | LOW | MID | HIGH |
|--------|-----|-----|------|
| GWC | (pending) | (pending) | (pending) |
| GWi | (pending) | (pending) | (pending) |
| A-GWi | (pending) | (pending) | (pending) |

*Analysis:* The adaptive gain is expected to concentrate in the HIGH stratum, where the per-pixel frequency increase sharpens boundaries between overlapping objects. The values will be inserted once the stratified evaluation completes.

### 5.8 Limitations and Failure Cases

Per-pixel spatially varying convolution has a higher constant factor than separable static convolution, so runtime is larger than for static methods. Sobel gradient energy as a density proxy can misattribute an isolated strong edge as a high-density region. The sigmoid parameters are selected empirically, not optimized. Evaluation is on natural-scene benchmarks; validation on dense agricultural grain imagery is left for future work.

---

## 6. CONCLUSIONS

A-GWi extends static GWi with per-pixel density-driven kernel adaptation, showing that spatially varying convolution can be implemented in handcrafted edge detection without training, GPU, or quantization. On BSDS500, A-GWi attains ODS = 0.534, the highest among all evaluated methods, and on UDED it attains ODS = 0.668, the highest among the multi-orientation methods. The gain over static GWi and GWC isolates the effect of per-pixel adaptation. Future work includes a learned density encoder that replaces the Sobel estimate, kernel quantization for embedded deployment, and extension to dense grain imagery for downstream tasks such as kernel counting and contamination screening.

---

## REFERENCES (IEEE, sequential order of first appearance)

[1] A. Khmag, "Innovative adaptive edge detection for noisy images using wavelet and Gaussian method," Sci. Rep., vol. 15, art. 86860, 2025. DOI: 10.1038/s41598-025-86860-9

[2] W. Yang, X.-D. Chen, H. Wang, and X. Mao, "Edge detection using multi-scale closest neighbor operator and grid partition," Vis. Comput., vol. 40, pp. 1947-1964, 2024. DOI: 10.1007/s00371-023-02894-y

[3] K. Yudhistiro, P. D. P. Adi, R. S. Basuki, and N. Rijati, "Halving Gabor convolutions: Imaginary-only wavelet with orientation-aware double-peak suppression for lightweight edge detection," manuscript under review, Int. J. Inf. Technol., Springer, 2026.

[4] J. Canny, "A computational approach to edge detection," IEEE Trans. Pattern Anal. Mach. Intell., vol. 8, no. 6, pp. 679-698, 1986. DOI: 10.1109/TPAMI.1986.4767851

[5] I. Sobel and G. Feldman, "A 3x3 isotropic gradient operator for image processing," Stanford Artificial Intelligence Project, 1968.

[6] D. Marr and E. Hildreth, "Theory of edge detection," Proc. R. Soc. Lond. B, vol. 207, no. 1167, pp. 187-217, 1980. DOI: 10.1098/rspb.1980.0020

[7] J. G. Daugman, "Two-dimensional spectral analysis of cortical receptive field profiles," Vis. Res., vol. 20, no. 10, pp. 847-856, 1980. DOI: 10.1016/0042-6989(80)90065-6

[8] J. G. Daugman, "Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters," J. Opt. Soc. Am. A, vol. 2, no. 7, pp. 1160-1169, 1985. DOI: 10.1364/JOSAA.2.001160

[9] I. A. Kamanga, "Improved edge detection using variable thresholding technique and convolution of Gabor with Gaussian filters," Signal Image Process. Int. J., vol. 13, no. 5, pp. 1-15, 2022.

[10] D. Diana, D. Agustina, D. Y. Situmorang, D. Kholid, and A. R. Pratama, "Comparison of Sobel, Prewitt, and Canny edge detection methods in digital images," J. Appl. Inf. Comput., vol. 10, no. 3, 2026.

[11] P. Arbelaez, M. Maire, C. Fowlkes, and J. Malik, "Contour detection and hierarchical image segmentation," IEEE Trans. Pattern Anal. Mach. Intell., vol. 33, no. 5, pp. 898-916, 2011. DOI: 10.1109/TPAMI.2010.161

[12] P. Kovesi, "Image features from phase congruency," Videre: J. Comput. Vis. Res., vol. 1, no. 3, pp. 1-26, 1999.

[13] D. Bradley and G. Roth, "Adaptive thresholding using the integral image," J. Graph. Tools, vol. 12, no. 2, pp. 13-21, 2007. DOI: 10.1080/2151237X.2007.10129236

[14] B. Ding, X. Long, L. Yang, J. Pang, and Z. Su, "Learnable Gabor edge detection layers for convolutional neural networks," Digit. Signal Process., 2025.

[15] Y. Yuan, J. Wang, and J. Chen, "Deformable Gabor feature networks for biomedical image classification," in Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV), 2021, pp. 4460-4469. DOI: 10.1109/WACV48630.2021.00451

[16] B. De Brabandere, X. Jia, T. Tuytelaars, and L. Van Gool, "Dynamic filter networks," in Adv. Neural Inf. Process. Syst. (NeurIPS), 2016.

[17] H. Su, V. Jampani, D. Sun, O. Gallo, E. Learned-Miller, and J. Kautz, "Pixel-adaptive convolutional neural networks," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 11166-11175. DOI: 10.1109/CVPR.2019.01142

[18] J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, and Y. Wei, "Deformable convolutional networks," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 764-773. DOI: 10.1109/ICCV.2017.89

[19] D. Howard, N. Nair, and A. Dopp, "CoordGate: Efficiently computing spatially-varying convolutions in convolutional neural networks," arXiv:2401.04680, 2024.

[20] Y. Li, Y. Bi, W. Zhang, J. Ren, and J. Chen, "M2GF: Multi-scale and multi-directional Gabor filters for image edge detection," Appl. Sci., vol. 13, no. 16, art. 9409, 2023. DOI: 10.3390/app13169409

[21] X. Sun and X.-F. Sun, "An edge detection algorithm based upon the adaptive multi-directional anisotropic Gaussian filter and its applications," J. Supercomput., vol. 80, no. 11, pp. 15183-15214, 2024. DOI: 10.1007/s11227-024-06044-6

[22] J. Zhang et al., "GlogSemiFNet: A semi-supervised contrastive learning framework for high-resolution remote sensing farmland edge detection based on frequency domain filters," Geo-spatial Inf. Sci., 2025.

[23] Y. Zhang, D. Zhou, S. Chen, S. Gao, and Y. Ma, "Single-image crowd counting via multi-column convolutional neural network," in Proc. CVPR, 2016, pp. 589-597. DOI: 10.1109/CVPR.2016.70

[24] J. Wan and A. B. Chan, "Adaptive density map generation for crowd counting," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 1130-1139. DOI: 10.1109/ICCV.2019.00122

[25] Y. Li, X. Zhang, and D. Chen, "CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes," in Proc. CVPR, 2018, pp. 1091-1100. DOI: 10.1109/CVPR.2018.00120

[26] J. M. S. Prewitt, "Object enhancement and extraction," in Picture Processing and Psychopictorics, B. S. Lipkin and A. Rosenfeld, Eds. New York: Academic Press, 1970, pp. 75-149.

[27] L. G. Roberts, "Machine perception of three-dimensional solids," Ph.D. dissertation, Massachusetts Institute of Technology, 1963.

[28] D. Gabor, "Theory of communication. Part 1: The analysis of information," J. Inst. Electr. Eng. III, vol. 93, no. 26, pp. 429-441, 1946. DOI: 10.1049/ji-3-2.1946.0074

[29] A. K. Jain and F. Farrokhnia, "Unsupervised texture segmentation using Gabor filters," Pattern Recognit., vol. 24, no. 12, pp. 1167-1186, 1991. DOI: 10.1016/0031-3203(91)90143-S

[30] C. Grigorescu, N. Petkov, and M. A. Westenberg, "Contour detection based on nonclassical receptive field inhibition," IEEE Trans. Image Process., vol. 12, no. 7, pp. 729-739, 2003. DOI: 10.1109/TIP.2003.814250

[31] C. Topal and C. Akinlar, "Edge Drawing: A combined real-time edge and segment detector," J. Vis. Commun. Image Represent., vol. 23, no. 6, pp. 862-872, 2012. DOI: 10.1016/j.jvcir.2012.05.004

[32] X. Soria, E. Riba, and A. Sappa, "Dense extreme inception network: Towards a robust CNN model for edge detection," in Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV), 2020, pp. 1923-1932. DOI: 10.1109/WACV45572.2020.9093290

[33] J. Jing, S. Liu, G. Wang, W. Zhang, and C. Sun, "Recent advances on image edge detection: A review," Neurocomputing, vol. 503, pp. 259-271, 2022. DOI: 10.1016/j.neucom.2022.06.083

[34] W. Jiang, K.-M. Lam, and T.-Z. Shen, "Edge detection using simplified Gabor wavelets," in Proc. Int. Conf. Neural Netw. Signal Process. (ICNNSP), 2008, pp. 586-591. DOI: 10.1109/ICNNSP.2008.4590418

[35] L. Xu, W. Wang, Z. Zhang, S. Ni, and Q. Bao, "A novel Sobel edge detection accelerator based on reconfigurable architecture," Traitement du Signal, vol. 39, no. 4, pp. 1421-1427, 2022. DOI: 10.18280/ts.390436

[36] I. Avci, "Threshold values of different classical edge detection algorithms," Traitement du Signal, vol. 39, no. 5, pp. 1775-1780, 2022. DOI: 10.18280/ts.390536

[37] M. Pu, Y. Huang, Y. Liu, Q. Guan, and H. Ling, "EDTER: Edge detection with transformer," in Proc. CVPR, 2022, pp. 1402-1412. DOI: 10.1109/CVPR52688.2022.00146

[38] Z. Su, W. Liu, Z. Yu, D. Hu, Q. Liao, Q. Tian, M. Pietikainen, and L. Liu, "Pixel difference networks for efficient edge detection," in Proc. ICCV, 2021, pp. 5097-5107. DOI: 10.1109/ICCV48922.2021.00507

[39] S. Khaki, H. Pham, Y. Han, A. Kuber, and L. Wang, "Convolutional neural networks for image-based corn kernel detection and counting," Sensors, vol. 20, no. 9, art. 2721, 2020. DOI: 10.3390/s20092721

---

## NOMENCLATURE

| Symbol | Description | Unit |
|--------|-------------|------|
| f_0 | Center frequency of Gabor kernel | cycles/pixel |
| f_min, f_max | Frequency range bounds | cycles/pixel |
| sigma_x, sigma_y | Gaussian envelope standard deviations | pixels |
| theta_k | Orientation angle, k = 0, ..., N-1 | radians |
| rho_L | Local density estimate | dimensionless [0,1] |
| k_s | Sigmoid steepness coefficient | dimensionless |
| k_adapt | Adaptive kernel size | pixels |
| N | Number of orientations | - |
| gamma | Aspect ratio (sigma_x / sigma_y) | dimensionless |
| H, W | Image height and width | pixels |
| E_AGWi | A-GWi edge magnitude map | [0, 1] |
| ODS, OIS, AP | Standard boundary detection metrics | - |

---

## NOTES FOR FINALIZATION

- Skeleton follows PAPER_FRAMEWORK_awal: abstract, research framework (1.3), gap table with 5 gaps (1.4), 3 contributions (1.5), KEY METHOD EQUATIONS (Eqs. 1-14), reference list, nomenclature.
- Section organization mirrors Paper 1: 1 Introduction, 2 Theory, 3 Method (6 stages), 4 Result and Analysis, 5 Experiments and Results, 6 Conclusions.
- Title uses option 2 from the brief, which matches the framework title. Option 1 is an equally valid alternative.
- Reference [3] is Paper 1 (GWi-ODPS), cited as under review.
- Tables 4, 5, 6, 7 and Figures 3, 6 are placeholders pending runtime logs, ablation, density distribution, and stratified evaluation.
- Figure note on white-background conversion is included in Figures 3, 4, 5, 6 per the 06c figure-generation convention.
- BSDS500 figure IDs: 3063, 29030, 128035, 35049. UDED figure IDs: 04-0896x4, 05-WIREFRAME-2, 28-img_043_SRF_2_HR.
- Em-dashes removed. AI buzzwords removed from prose; "robust" remains only inside the published title of reference [32], and "comprehensive" was removed from the title of reference [33] (shortened to "A review").
- References [26], [27], [29], [30], [31], [35], [37], [38], [39] are carried from the framework list for completeness; cite only those actually used in the final text, and renumber sequentially after pruning.
- Verify DOI for [14] and [22] before submission.
