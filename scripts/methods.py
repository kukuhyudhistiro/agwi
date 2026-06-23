"""
methods.py — All edge detection methods for the A-GWi paper (self-contained)
Author: Kukuh Yudhistiro, 2026

This paper is self-contained and does NOT depend on any Paper 1 files.
All six methods are implemented here:
    - A-GWi  (proposed): per-pixel adaptive imaginary Gabor (SVC)
    - GWi    (static imaginary Gabor)
    - GWC    (static complex Gabor)
    - Canny
    - Sobel
    - PC     (Phase Congruency, requires phasepack)

Conventions (kept internally consistent):
    - Gabor parameterized by wavelength (lambda); sigma = 0.56 * lambda
    - gamma = 0.5 aspect ratio
    - 8 orientations at 22.5 deg
    - imaginary kernels use sin; complex uses cos+sin
    - magnitude: |response| (GWi/A-GWi) or sqrt(re^2+im^2) (GWC)
    - max-pool across orientations
    - preprocessing: BT.601 grayscale, histogram equalization, /255

Numba is REQUIRED for A-GWi speed. Falls back to slow NumPy if absent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import cv2
from scipy import ndimage

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*a, **k):
        def deco(f):
            return f
        return deco
    prange = range

try:
    from phasepack import phasecong
    HAS_PHASEPACK = True
except ImportError:
    HAS_PHASEPACK = False


ORIENTATIONS_8 = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]


# ============================================================================
# Preprocessing
# ============================================================================
def preprocess(image_path):
    """BT.601 grayscale, histogram equalization, normalize to [0,1]."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    b, g, r = cv2.split(img)
    gray = (0.299 * r.astype(np.float64) +
            0.587 * g.astype(np.float64) +
            0.114 * b.astype(np.float64))
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    eq = cv2.equalizeHist(gray_u8)
    return eq.astype(np.float64) / 255.0, gray_u8


# ============================================================================
# Static Gabor kernels (GWi, GWC)
# ============================================================================
def gabor_imaginary(ksize, wavelength, theta_deg, gamma=0.5):
    sigma = 0.56 * wavelength
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    th = np.deg2rad(theta_deg)
    x_t = x * np.cos(th) + y * np.sin(th)
    y_t = -x * np.sin(th) + y * np.cos(th)
    gauss = np.exp(-0.5 * (x_t ** 2 + gamma ** 2 * y_t ** 2) / sigma ** 2)
    return gauss * np.sin(2 * np.pi * x_t / wavelength)


def gabor_complex(ksize, wavelength, theta_deg, gamma=0.5):
    sigma = 0.56 * wavelength
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    th = np.deg2rad(theta_deg)
    x_t = x * np.cos(th) + y * np.sin(th)
    y_t = -x * np.sin(th) + y * np.cos(th)
    gauss = np.exp(-0.5 * (x_t ** 2 + gamma ** 2 * y_t ** 2) / sigma ** 2)
    s = 2 * np.pi * x_t / wavelength
    return gauss * np.cos(s), gauss * np.sin(s)


# ============================================================================
# Baselines
# ============================================================================
def run_canny(gray_u8):
    t0 = time.perf_counter()
    v = np.median(gray_u8)
    lo = int(max(0, 0.67 * v))
    hi = int(min(255, 1.33 * v))
    blurred = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    edges = cv2.Canny(blurred, lo, hi)
    rt = time.perf_counter() - t0
    return edges.astype(np.float64) / 255.0, rt


def run_sobel(gray_f):
    t0 = time.perf_counter()
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    rt = time.perf_counter() - t0
    return mag, rt


def run_pc(gray_f, nscale=4, norient=6):
    if not HAS_PHASEPACK:
        raise ImportError("phasepack not installed. pip install phasepack")
    t0 = time.perf_counter()
    M = phasecong(gray_f, nscale=nscale, norient=norient)[0]
    rt = time.perf_counter() - t0
    return np.asarray(M, dtype=np.float64), rt


def run_gwc(gray_f, ksize=7, wavelength=4.0):
    t0 = time.perf_counter()
    mags = []
    for th in ORIENTATIONS_8:
        kr, ki = gabor_complex(ksize, wavelength, th)
        rr = ndimage.convolve(gray_f, kr, mode='constant')
        ri = ndimage.convolve(gray_f, ki, mode='constant')
        mags.append(np.sqrt(rr ** 2 + ri ** 2))
    mag = np.max(np.stack(mags), axis=0)
    rt = time.perf_counter() - t0
    return mag, rt


def run_gwi(gray_f, ksize=7, wavelength=4.0):
    t0 = time.perf_counter()
    mags = []
    for th in ORIENTATIONS_8:
        ki = gabor_imaginary(ksize, wavelength, th)
        ri = ndimage.convolve(gray_f, ki, mode='constant')
        mags.append(np.abs(ri))
    mag = np.max(np.stack(mags), axis=0)
    rt = time.perf_counter() - t0
    return mag, rt


# ============================================================================
# A-GWi (proposed): adaptive imaginary Gabor, per-pixel SVC
# f0-based formulation (matches Paper 2 draft AND user's proven experiment).
#   dense  (rho->1): f0->f_max (high freq), sigma small  -> sharp boundary
#   sparse (rho->0): f0->f_min (low freq),  sigma large  -> broad structure
#   sigma = (1/(pi*f0)) * sqrt(ln2/2)   [Gabor bandwidth relation]
# Fixed kernel size (default 7), aspect ratio gamma=0.5.
# ============================================================================
@dataclass
class AGWiParams:
    f_min: float = 0.05
    f_max: float = 0.45
    k_steepness: float = 25.0
    n_orientations: int = 8
    aspect: float = 0.5          # sigma_y = sigma_x / aspect
    kernel_size: int = 7
    sobel_ksize: int = 5
    density_blur: int = 5


def estimate_density(image_norm, sobel_ksize=5, blur_ksize=5):
    gx = cv2.Sobel(image_norm, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(image_norm, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    energy = np.sqrt(gx ** 2 + gy ** 2)
    rho = cv2.normalize(energy, None, 0, 1, cv2.NORM_MINMAX)
    rho = cv2.GaussianBlur(rho, (blur_ksize, blur_ksize), 0)
    return rho.astype(np.float64)


def estimate_density_variance(image_norm, win=5, blur_ksize=5):
    mean = cv2.boxFilter(image_norm, cv2.CV_64F, (win, win))
    mean_sq = cv2.boxFilter(image_norm ** 2, cv2.CV_64F, (win, win))
    var = np.maximum(mean_sq - mean ** 2, 0)
    rho = cv2.normalize(var, None, 0, 1, cv2.NORM_MINMAX)
    rho = cv2.GaussianBlur(rho, (blur_ksize, blur_ksize), 0)
    return rho.astype(np.float64)


@jit(nopython=True, cache=True, parallel=True)
def _agwi_svc(image_padded, rho_L, H, W, ksize, n_orient,
              k_s, f_min, f_max, aspect):
    """Per-pixel adaptive imaginary Gabor SVC (f0 formulation)."""
    half_k = ksize // 2
    output = np.zeros((H, W))
    step = np.pi / n_orient
    for r in prange(H):
        for c in range(W):
            rho = rho_L[r, c]
            # Adaptive frequency (sigmoid): dense -> high f0
            f_sig = 1.0 / (1.0 + np.exp(-k_s * (rho - 0.5)))
            f0 = f_min + (f_max - f_min) * f_sig
            # Adaptive scale: inverse to frequency (Gabor bandwidth)
            sigma_a = (1.0 / (np.pi * f0)) * np.sqrt(np.log(2.0) / 2.0)
            sigma_x = sigma_a
            sigma_y = sigma_a / aspect
            max_mag = 0.0
            for o in range(n_orient):
                theta = o * step
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                response = 0.0
                for xi in range(ksize):
                    for yi in range(ksize):
                        x = xi - half_k + 0.0
                        y = yi - half_k + 0.0
                        xp = x * cos_t + y * sin_t
                        yp = -x * sin_t + y * cos_t
                        gauss = np.exp(-0.5 * (xp ** 2 / sigma_x ** 2
                                              + yp ** 2 / sigma_y ** 2))
                        kval = gauss * np.sin(2.0 * np.pi * f0 * xp)
                        response += image_padded[r + xi, c + yi] * kval
                m = abs(response)
                if m > max_mag:
                    max_mag = m
            output[r, c] = max_mag
    return output


def run_agwi(gray_f, params=None, rho_L=None, density_method='sobel'):
    if params is None:
        params = AGWiParams()
    H, W = gray_f.shape
    t0 = time.perf_counter()
    if rho_L is None:
        if density_method == 'variance':
            rho_L = estimate_density_variance(gray_f, params.sobel_ksize,
                                              params.density_blur)
        else:
            rho_L = estimate_density(gray_f, params.sobel_ksize,
                                     params.density_blur)
    half_k = params.kernel_size // 2
    padded = cv2.copyMakeBorder(gray_f.astype(np.float64),
                                half_k, half_k, half_k, half_k,
                                cv2.BORDER_REFLECT)
    mag = _agwi_svc(padded, rho_L, H, W,
                    params.kernel_size, params.n_orientations,
                    params.k_steepness, params.f_min, params.f_max,
                    params.aspect)
    rt = time.perf_counter() - t0
    return mag, rt


def warmup_agwi(params=None):
    if NUMBA_AVAILABLE:
        _ = run_agwi(np.random.rand(16, 16), params)
        return True
    return False


# ============================================================================
# Dispatcher
# ============================================================================
def run_method(method_name, gray_f, gray_u8, agwi_params=None,
               density_method='sobel'):
    """Returns (magnitude_raw, runtime_s). Magnitude is NOT normalized."""
    if method_name == 'AGWi':
        return run_agwi(gray_f, agwi_params, density_method=density_method)
    elif method_name == 'GWi':
        return run_gwi(gray_f)
    elif method_name == 'GWC':
        return run_gwc(gray_f)
    elif method_name == 'Canny':
        return run_canny(gray_u8)
    elif method_name == 'Sobel':
        return run_sobel(gray_f)
    elif method_name == 'PC':
        return run_pc(gray_f)
    else:
        raise ValueError(f"Unknown method: {method_name}")
