#!/usr/bin/env python3
"""Numpy GLSL-equivalent lake detection algorithm library.

All functions are pure, stateless, and match the GLSL vertex shader behavior
in `src/shaders/earth.vert.glsl`. 1D simplification: dv = 0, terrainSteepness = |du|.

Reference functions implement the exact shader logic. Variant functions implement
the candidate algorithms from the benchmark matrix (docs/LAKE_BOUNDARY_BENCHMARK.md).
"""

import numpy as np


# ============================================================================
# Terrain-RGB encode/decode (exact match with scripts/tile_utils.py)
# ============================================================================

def terrain_rgb_decode_rgb(r, g, b):
    """Decode terrain-rgb float channels to elevation in meters."""
    return -10000.0 + (r * 65536.0 + g * 256.0 + b) * 0.1


def terrain_rgb_encode(elev):
    """Encode elevation (m) to terrain-rgb uint32 codes (0.1m quantization)."""
    encoded = np.clip((np.asarray(elev) + 10000.0) / 0.1, 0, 16777215).astype(np.uint32)
    return encoded


def terrain_rgb_decode(encoded):
    """Decode terrain-rgb uint32 codes back to elevation."""
    return -10000.0 + np.asarray(encoded, dtype=np.float64) * 0.1


# ============================================================================
# Interpolation kernels (exact GLSL match)
# ============================================================================

def cubic_cr(x):
    """Catmull-Rom cubic weight (a = -0.5), matches GLSL cubicCR()."""
    ax = np.abs(x)
    w = np.zeros_like(ax, dtype=np.float64)
    # ax < 1.0
    mask1 = ax < 1.0
    w[mask1] = (1.5 * ax[mask1] - 2.5) * ax[mask1] * ax[mask1] + 1.0
    # 1.0 <= ax < 2.0
    mask2 = (ax >= 1.0) & (ax < 2.0)
    w[mask2] = ((-0.5 * ax[mask2] + 2.5) * ax[mask2] - 4.0) * ax[mask2] + 2.0
    return w


def lanczos2(x):
    """Lanczos-2 windowed sinc, matches GLSL lanczos2()."""
    ax = np.abs(x)
    w = np.zeros_like(ax, dtype=np.float64)
    mask = ax < 2.0
    axm = ax[mask]
    pix = np.pi * axm
    # sin(x)/x with guard for x→0
    with np.errstate(divide="ignore", invalid="ignore"):
        w[mask] = np.where(
            axm < 0.001,
            1.0,
            np.sin(pix) * np.sin(pix * 0.5) / (pix * pix * 0.5),
        )
    return w


# ============================================================================
# GLSL smoothstep
# ============================================================================

def smoothstep(edge0, edge1, x):
    """GLSL smoothstep: Hermite interpolation between edge0 and edge1."""
    t = np.clip((np.asarray(x, dtype=np.float64) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ============================================================================
# 1D Elevation sampling (matches shader's sampleElevation)
# ============================================================================

def sample_nearest_1d(arr, uv_continuous):
    """Nearest-neighbor sampling at continuous coordinate uv_continuous.

    Args:
        arr: 1D numpy array of elevations
        uv_continuous: float, continuous index into arr (0..len-1)

    Returns:
        elevation value
    """
    idx = int(np.floor(uv_continuous + 0.5))
    idx = max(0, min(idx, len(arr) - 1))
    return arr[idx]


def sample_bilinear_1d(arr, uv_continuous):
    """Bilinear (linear) interpolation at continuous coordinate.

    Args:
        arr: 1D numpy array of elevations
        uv_continuous: float, continuous index into arr

    Returns:
        interpolated elevation
    """
    corner = uv_continuous - 0.5
    idx0 = int(np.floor(corner))
    idx1 = idx0 + 1
    frac = corner - idx0

    s0 = arr[max(0, min(idx0, len(arr) - 1))]
    s1 = arr[max(0, min(idx1, len(arr) - 1))]

    return s0 * (1.0 - frac) + s1 * frac


def sample_filtered_1d(arr, uv_continuous, mode=2.0):
    """4-tap filtered sampling (Catmull-Rom or Lanczos).

    Matches shader's sampleFiltered() for 1D. Uses 4-tap separable kernel.

    Args:
        arr: 1D numpy array of elevations
        uv_continuous: float, continuous index into arr
        mode: 2.0 = Catmull-Rom (bicubic), else = Lanczos-2

    Returns:
        filtered elevation
    """
    center = uv_continuous
    frac = center - np.floor(center)

    result = 0.0
    total_weight = 0.0
    for i in range(-1, 3):  # i = -1, 0, 1, 2
        if mode == 2.0:
            w = cubic_cr(float(i) - frac)
        else:
            w = lanczos2(float(i) - frac)

        tap_uv = uv_continuous + float(i)
        tap_idx = int(np.floor(tap_uv))
        tap_idx = max(0, min(tap_idx, len(arr) - 1))
        elev = arr[tap_idx]

        result += elev * w
        total_weight += w

    if total_weight > 0.001:
        return result / total_weight
    # Fallback: nearest
    idx = max(0, min(int(np.floor(uv_continuous)), len(arr) - 1))
    return arr[idx]


def sample_elevation_1d(arr, uv_continuous, interp_mode=2.0):
    """Sample elevation with specified interpolation mode.

    Args:
        arr: 1D numpy array
        uv_continuous: continuous index
        interp_mode: 0=nearest, 1=bilinear, 2=bicubic(Catmull-Rom), 3=lanczos2

    Returns:
        elevation
    """
    if interp_mode == 0.0:
        return sample_nearest_1d(arr, uv_continuous)
    elif interp_mode == 1.0:
        return sample_bilinear_1d(arr, uv_continuous)
    else:
        return sample_filtered_1d(arr, uv_continuous, interp_mode)


# ============================================================================
# Reference algorithm (current shader: G1 + N1 + S0 + M1 + D1 + E3)
# ============================================================================

def compute_lake_mask_reference(arr, flatness=100.0, lake_range=1.0, texture_size=256.0):
    """Reference lake detection: central difference + L2 + smoothstep.

    This is the exact 1D equivalent of the current vertex shader algorithm,
    with elevation guard removed (E3).

    Gradient is scaled by texture_size to match shader UV-space units
    (shader: lakeStep = range/textureSize, du = dh/(2*lakeStep)).

    Args:
        arr: 1D numpy array of elevations (m)
        flatness: uLakeFlatness parameter (default 100)
        lake_range: uLakeRange parameter in texel units (default 1.0)
        texture_size: uTextureSize equivalent (default 256.0, matches shader)

    Returns:
        lakeMask array (same length as arr), values in [0, 1]
    """
    n = len(arr)
    lake_mask = np.zeros(n, dtype=np.float64)

    # lakeStep in shader-equivalent UV units:
    # lakeStep_uv = lake_range / texture_size
    # In 1D index space: lake_step_indices = lake_range * (n / texture_size)
    scale = n / texture_size
    lake_step = lake_range * scale

    scale_factor = texture_size  # scales gradient from m/index to m/UV

    for i in range(n):
        uv_i = float(i)

        h_r = sample_filtered_1d(arr, uv_i + lake_step, 2.0)
        h_l = sample_filtered_1d(arr, uv_i - lake_step, 2.0)

        lake_du = (h_r - h_l) / (2.0 * lake_step) * scale_factor
        terrain_steepness = np.abs(lake_du)

        lo = flatness * 0.3
        hi = flatness
        lake_mask[i] = 1.0 - smoothstep(lo, hi, terrain_steepness)

    return lake_mask


# ============================================================================
# Gradient operators (G1-G6)
# ============================================================================

def grad_central_diff(arr, i, step, interp_mode=2.0):
    """G1: 2-point central difference."""
    uv = float(i)
    h_r = sample_elevation_1d(arr, uv + step, interp_mode)
    h_l = sample_elevation_1d(arr, uv - step, interp_mode)
    return (h_r - h_l) / (2.0 * step)


def grad_forward_diff(arr, i, step, interp_mode=2.0):
    """G2: 2-point forward difference."""
    uv = float(i)
    h = sample_elevation_1d(arr, uv, interp_mode)
    h_r = sample_elevation_1d(arr, uv + step, interp_mode)
    return (h_r - h) / step


def grad_3point(arr, i, step, interp_mode=2.0):
    """G3: 3-point stencil (-1, 0, 1)."""
    uv = float(i)
    h_r = sample_elevation_1d(arr, uv + step, interp_mode)
    h_l = sample_elevation_1d(arr, uv - step, interp_mode)
    h_c = sample_elevation_1d(arr, uv, interp_mode)
    return (h_r - h_l) / (2.0 * step)  # same as central, but with center tap available


def grad_5point(arr, i, step, interp_mode=2.0):
    """G4: 5-point stencil (-2, -1, 0, 1, 2)."""
    uv = float(i)
    h2_r = sample_elevation_1d(arr, uv + 2.0 * step, interp_mode)
    h_r = sample_elevation_1d(arr, uv + step, interp_mode)
    h_l = sample_elevation_1d(arr, uv - step, interp_mode)
    h2_l = sample_elevation_1d(arr, uv - 2.0 * step, interp_mode)
    return (-h2_r + 8.0 * h_r - 8.0 * h_l + h2_l) / (12.0 * step)


def grad_sobel(arr, i, step, interp_mode=2.0):
    """G5: 3-point Sobel-like weighted central difference."""
    uv = float(i)
    h_r = sample_elevation_1d(arr, uv + step, interp_mode)
    h_c = sample_elevation_1d(arr, uv, interp_mode)
    h_l = sample_elevation_1d(arr, uv - step, interp_mode)
    return (2.0 * h_r + h_r - 2.0 * h_l - h_l) / (6.0 * step)
    # Simplified: (3*h_r - 3*h_l) / (4*step) ≈ 0.75 * central_diff


def grad_prewitt(arr, i, step, interp_mode=2.0):
    """G6: Prewitt-like 3-point gradient."""
    uv = float(i)
    h_r = sample_elevation_1d(arr, uv + step, interp_mode)
    h_l = sample_elevation_1d(arr, uv - step, interp_mode)
    return (h_r - h_l) / (2.0 * step)  # In 1D, Prewitt = central difference


GRAD_FUNCTIONS = {
    "G1": grad_central_diff,
    "G2": grad_forward_diff,
    "G3": grad_3point,
    "G4": grad_5point,
    "G5": grad_sobel,
    "G6": grad_prewitt,
}


# ============================================================================
# Norm functions (N1-N3)
# ============================================================================

def norm_l2(grad_val):
    """N1: L2 norm."""
    return np.abs(grad_val)


def norm_l1(grad_val):
    """N2: L1 norm."""
    return np.abs(grad_val)


def norm_linf(grad_val):
    """N3: L∞ norm."""
    return np.abs(grad_val)


# In 1D, all norms reduce to abs(). They differ only in 2D.
NORM_FUNCTIONS = {"N1": norm_l2, "N2": norm_l1, "N3": norm_linf}


# ============================================================================
# Window statistics (W1-W4) — alternative to gradient
# ============================================================================

def window_variance(arr, i, window=3):
    """W1/W2: Local variance in window around i."""
    half = window // 2
    lo = max(0, i - half)
    hi = min(len(arr), i + half + 1)
    patch = arr[lo:hi]
    if len(patch) < 3:
        return 0.0
    return float(np.var(patch))


def window_range(arr, i, window=3):
    """W3: Local range (max - min) in window."""
    half = window // 2
    lo = max(0, i - half)
    hi = min(len(arr), i + half + 1)
    patch = arr[lo:hi]
    if len(patch) < 3:
        return 0.0
    return float(np.max(patch) - np.min(patch))


def window_rms_roughness(arr, i, window=3):
    """W4: RMS roughness after local linear detrend."""
    half = window // 2
    lo = max(0, i - half)
    hi = min(len(arr), i + half + 1)
    patch = arr[lo:hi]
    if len(patch) < 4:
        return 0.0
    xs = np.arange(len(patch), dtype=np.float64)
    coeffs = np.polyfit(xs, patch, 1)
    trend = np.polyval(coeffs, xs)
    residuals = patch - trend
    return float(np.sqrt(np.mean(residuals**2)))


WINDOW_FUNCTIONS = {
    "W1": (window_variance, 3),
    "W2": (window_variance, 5),
    "W3": (window_range, 3),
    "W4": (window_rms_roughness, 3),
}


# ============================================================================
# Pre-smoothing (S0-S2)
# ============================================================================

def smooth_box(arr, window=3):
    """S1: Box (moving average) smoothing."""
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="same")


def smooth_gaussian(arr, sigma=0.75):
    """S2: Approximate Gaussian smoothing (3-sigma window).

    sigma=0.75 texels approximates a 3×3 Gaussian in 1D.
    """
    half = int(np.ceil(sigma * 3))
    xs = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


# ============================================================================
# Decision functions (D1-D3)
# ============================================================================

def decision_smoothstep(steepness, flatness, lo_ratio=0.3):
    """D1/D3: smoothstep(lo_ratio * flatness, flatness, steepness)."""
    lo = flatness * lo_ratio
    hi = flatness
    return 1.0 - smoothstep(lo, hi, steepness)


def decision_hardstep(steepness, flatness):
    """D2: Hard threshold at flatness."""
    return np.where(steepness < flatness, 1.0, 0.0)


# ============================================================================
# Full algorithm runner — configurable all dimensions
# ============================================================================

def compute_lake_mask(arr, flatness=100.0, lake_range=1.0, texture_size=256.0,
                      grad_op="G1", norm="N1", window_stat=None,
                      pre_smooth="S0", multi_scale="M1",
                      decision_fn="D1", decision_k=0.3,
                      interp_mode=2.0):
    """Generic lake mask computation with configurable algorithm parameters.

    Args:
        arr: 1D numpy array of elevations (m), regularly spaced
        flatness: uLakeFlatness (default 100)
        lake_range: step size in texel units (default 1.0)
        texture_size: uTextureSize equivalent (default 256.0)
        grad_op: "G1".."G6" gradient operator
        norm: "N1".."N3" norm (all same in 1D)
        window_stat: None or "W1".."W4" — if set, replaces gradient with window stat
        pre_smooth: "S0"(none), "S1"(box), "S2"(gaussian)
        multi_scale: "M1"(single), "M2"(large range), "M3"(min), "M4"(product)
        decision_fn: "D1"(smoothstep), "D2"(hard), "D3"(variable smoothstep)
        decision_k: lo_ratio for smoothstep (D1=0.3, D3 can vary)
        interp_mode: 0=nearest, 1=bilinear, 2=bicubic, 3=lanczos2

    Returns:
        lakeMask array (same length as arr), values in [0, 1]
    """
    n = len(arr)
    scale = n / texture_size
    scale_factor = texture_size  # m/index → m/UV-equivalent

    # Pre-smoothing
    if pre_smooth == "S1":
        arr_smoothed = smooth_box(arr, 3)
    elif pre_smooth == "S2":
        arr_smoothed = smooth_gaussian(arr, 0.75)
    else:
        arr_smoothed = arr

    # Multi-scale
    if multi_scale == "M2":
        ranges = [lake_range]  # single large range
    elif multi_scale in ("M3", "M4"):
        ranges = [1.0, lake_range]  # small + large
    else:
        ranges = [1.0]  # M1: single scale

    # Compute steepness for each scale
    scale_steepnesses = []
    for r in ranges:
        step = r * scale  # texels → array indices
        steepness = np.zeros(n, dtype=np.float64)

        if window_stat is not None:
            # Use window statistic instead of gradient (scale-independent)
            win_fn, win_size = WINDOW_FUNCTIONS[window_stat]
            for i in range(n):
                steepness[i] = win_fn(arr_smoothed, i, win_size)
        else:
            # Gradient-based, scaled to shader-equivalent UV units
            grad_fn = GRAD_FUNCTIONS[grad_op]
            for i in range(n):
                g = grad_fn(arr_smoothed, i, step, interp_mode)
                steepness[i] = np.abs(g) * scale_factor

        scale_steepnesses.append(steepness)

    # Combine scales
    if multi_scale == "M3":
        combined = np.minimum(*scale_steepnesses)
    elif multi_scale == "M4":
        combined = scale_steepnesses[0] * scale_steepnesses[1]
    else:
        combined = scale_steepnesses[0]

    # Decision
    if decision_fn == "D2":
        lake_mask = decision_hardstep(combined, flatness)
    else:
        k = 0.3 if decision_fn == "D1" else decision_k
        lake_mask = decision_smoothstep(combined, flatness, lo_ratio=k)

    return lake_mask


# ============================================================================
# Algorithm config builder for full matrix
# ============================================================================

def build_algorithm_configs():
    """Build the list of all algorithm configurations to benchmark.

    Returns:
        list of (label, kwargs_dict)
    """
    configs = []

    # Baseline
    configs.append(("G1_N1_S0_M1_D1_E3", {
        "grad_op": "G1", "norm": "N1", "window_stat": None,
        "pre_smooth": "S0", "multi_scale": "M1",
        "decision_fn": "D1", "decision_k": 0.3,
        "flatness": 100.0, "lake_range": 1.0,
    }))

    # --- Dimension sweep: Gradient operators ---
    for g in ["G1", "G2", "G3", "G4", "G5", "G6"]:
        if g == "G1":
            continue  # already in baseline
        label = f"{g}_N1_S0_M1_D1_E3"
        configs.append((label, {
            "grad_op": g, "norm": "N1", "window_stat": None,
            "pre_smooth": "S0", "multi_scale": "M1",
            "decision_fn": "D1", "decision_k": 0.3,
            "flatness": 100.0, "lake_range": 1.0,
        }))

    # --- Dimension sweep: Norms (1D: all same, but include for completeness) ---
    for n in ["N2", "N3"]:
        label = f"G1_{n}_S0_M1_D1_E3"
        configs.append((label, {
            "grad_op": "G1", "norm": n, "window_stat": None,
            "pre_smooth": "S0", "multi_scale": "M1",
            "decision_fn": "D1", "decision_k": 0.3,
            "flatness": 100.0, "lake_range": 1.0,
        }))

    # --- Dimension sweep: Window statistics ---
    for w in ["W1", "W2", "W3", "W4"]:
        label = f"G1_N1_{w}_S0_M1_D1_E3"
        configs.append((label, {
            "grad_op": "G1", "norm": "N1", "window_stat": w,
            "pre_smooth": "S0", "multi_scale": "M1",
            "decision_fn": "D1", "decision_k": 0.3,
            "flatness": 100.0, "lake_range": 1.0,
        }))

    # --- Dimension sweep: Pre-smoothing ---
    for s in ["S1", "S2"]:
        label = f"G1_N1_S0_{s}_M1_D1_E3"
        configs.append((label, {
            "grad_op": "G1", "norm": "N1", "window_stat": None,
            "pre_smooth": s, "multi_scale": "M1",
            "decision_fn": "D1", "decision_k": 0.3,
            "flatness": 100.0, "lake_range": 1.0,
        }))

    # --- Dimension sweep: Multi-scale ---
    for m in ["M2", "M3", "M4"]:
        for lr in [1.0, 4.0, 8.0, 16.0]:
            label = f"G1_N1_S0_{m}_r{int(lr)}_D1_E3"
            configs.append((label, {
                "grad_op": "G1", "norm": "N1", "window_stat": None,
                "pre_smooth": "S0", "multi_scale": m,
                "decision_fn": "D1", "decision_k": 0.3,
                "flatness": 100.0, "lake_range": lr,
            }))

    # --- Dimension sweep: Decision functions ---
    for df in ["D1", "D2", "D3"]:
        ks = [0.3] if df == "D1" else ([0.5] if df == "D2" else [0.1, 0.3, 0.5, 0.7])
        for k in ks:
            label = f"G1_N1_S0_M1_{df}_k{k}_E3"
            configs.append((label, {
                "grad_op": "G1", "norm": "N1", "window_stat": None,
                "pre_smooth": "S0", "multi_scale": "M1",
                "decision_fn": df, "decision_k": k,
                "flatness": 100.0, "lake_range": 1.0,
            }))

    # --- Flatness parameter sweep ---
    for f in [20, 50, 100, 200, 500]:
        if f == 100:
            continue  # baseline
        label = f"G1_N1_S0_M1_D1_k0.3_f{f}_E3"
        configs.append((label, {
            "grad_op": "G1", "norm": "N1", "window_stat": None,
            "pre_smooth": "S0", "multi_scale": "M1",
            "decision_fn": "D1", "decision_k": 0.3,
            "flatness": float(f), "lake_range": 1.0,
        }))

    # --- Round 2: Combined optimums ---
    # Best dimensions from round 1: M4(multi-scale product) + G2(forward diff)
    combined = [
        # M4_r1 variants with different grad ops
        ("G2_S0_M4_r1_D1_E3", {"grad_op": "G2", "pre_smooth": "S0", "multi_scale": "M4",
         "decision_fn": "D1", "decision_k": 0.3, "flatness": 100.0, "lake_range": 1.0}),
        ("G4_S0_M4_r1_D1_E3", {"grad_op": "G4", "pre_smooth": "S0", "multi_scale": "M4",
         "decision_fn": "D1", "decision_k": 0.3, "flatness": 100.0, "lake_range": 1.0}),
        # M4_r1 with D3
        ("G1_S0_M4_r1_D3_k0.1_E3", {"grad_op": "G1", "pre_smooth": "S0", "multi_scale": "M4",
         "decision_fn": "D3", "decision_k": 0.1, "flatness": 100.0, "lake_range": 1.0}),
        ("G2_S0_M4_r1_D3_k0.1_E3", {"grad_op": "G2", "pre_smooth": "S0", "multi_scale": "M4",
         "decision_fn": "D3", "decision_k": 0.1, "flatness": 100.0, "lake_range": 1.0}),
        # M4 with intermediate ranges (r=2, r=3)
        ("G1_S0_M4_r2_D1_E3", {"grad_op": "G1", "pre_smooth": "S0", "multi_scale": "M4",
         "decision_fn": "D1", "decision_k": 0.3, "flatness": 100.0, "lake_range": 2.0}),
        ("G1_S0_M4_r3_D1_E3", {"grad_op": "G1", "pre_smooth": "S0", "multi_scale": "M4",
         "decision_fn": "D1", "decision_k": 0.3, "flatness": 100.0, "lake_range": 3.0}),
        ("G2_S0_M4_r2_D1_E3", {"grad_op": "G2", "pre_smooth": "S0", "multi_scale": "M4",
         "decision_fn": "D1", "decision_k": 0.3, "flatness": 100.0, "lake_range": 2.0}),
        # Tuned flatness for M4_r1
        ("G2_S0_M4_r1_D1_f50_E3", {"grad_op": "G2", "pre_smooth": "S0", "multi_scale": "M4",
         "decision_fn": "D1", "decision_k": 0.3, "flatness": 50.0, "lake_range": 1.0}),
        ("G2_S0_M4_r1_D1_f200_E3", {"grad_op": "G2", "pre_smooth": "S0", "multi_scale": "M4",
         "decision_fn": "D1", "decision_k": 0.3, "flatness": 200.0, "lake_range": 1.0}),
    ]
    for label, params in combined:
        params.setdefault("norm", "N1")
        params.setdefault("window_stat", None)
        configs.append((label, params))

    # --- Round 3: Fine-tune flatness for top combination ---
    for f_tune in [10, 20, 30, 40, 60, 80, 150]:
        label = f"G2_S0_M4_r1_D1_f{f_tune}_E3"
        configs.append((label, {
            "grad_op": "G2", "norm": "N1", "window_stat": None,
            "pre_smooth": "S0", "multi_scale": "M4",
            "decision_fn": "D1", "decision_k": 0.3,
            "flatness": float(f_tune), "lake_range": 1.0,
        }))

    return configs
