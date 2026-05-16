#!/usr/bin/env python3
"""Lake boundary benchmark runner.

Loads profiles from SQLite, runs all algorithm configs, computes metrics,
stores results back to SQLite. Supports synthetic profile generation with
noise models.

Usage:
  python scripts/benchmark_lake.py                    # run all profiles
  python scripts/benchmark_lake.py --profile-id 3     # single profile
  python scripts/benchmark_lake.py --dim sweep        # per-dimension sweep
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "tmp" / "lake_benchmark" / "lake_boundary.db"

# Add scripts dir for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lake_algorithms import (
    compute_lake_mask,
    build_algorithm_configs,
    terrain_rgb_encode,
    terrain_rgb_decode,
)

# ============================================================================
# Synthetic profile generation
# ============================================================================


def generate_lake_profile(water_level=0.0, slope=0.01, shoreline_idx=200,
                          micro_relief_sigma=1.0, profile_len=512):
    """Generate a synthetic 1D lake profile (single boundary).

    h(x) = waterLevel                 for x < shoreline
         = waterLevel + slope*(x-shoreline) + microRelief  for x >= shoreline

    Args:
        water_level: Elevation of the water surface (m)
        slope: Land slope in m per texel
        shoreline_idx: Sample index of the true shoreline
        micro_relief_sigma: Standard deviation of land micro-relief (m)
        profile_len: Total profile length in samples

    Returns:
        (elevations, ideal_lake_mask) arrays
    """
    np.random.seed(42)
    xs = np.arange(profile_len, dtype=np.float64)
    elev = np.full(profile_len, water_level, dtype=np.float64)

    # Land slope and micro-relief
    land_mask = xs >= shoreline_idx
    elev[land_mask] = water_level + slope * (xs[land_mask] - shoreline_idx)

    # Micro-relief: Gaussian random walk (pink-ish for realism)
    relief = np.zeros(profile_len, dtype=np.float64)
    white = np.random.randn(profile_len) * micro_relief_sigma
    # Simple low-pass: cumulative sum with decay
    alpha = 0.7
    relief[0] = white[0]
    for i in range(1, profile_len):
        relief[i] = alpha * relief[i - 1] + (1 - alpha) * white[i]
    elev[land_mask] += relief[land_mask]

    # Ideal mask: 1.0 for water, 0.0 for land
    ideal = np.where(xs < shoreline_idx, 1.0, 0.0)

    return elev, ideal


def generate_river_profile(water_level=0.0, slope=0.02,
                           bank1_idx=150, bank2_idx=300,
                           micro_relief_sigma=1.5, profile_len=512):
    """Generate a synthetic 1D river profile (two boundaries).

    h(x) = landElev + slope*(x)          for x < bank1
         = waterLevel                     for bank1 <= x <= bank2
         = landElev + slope*(x-bank2)     for x > bank2

    Args:
        water_level: River surface elevation (m)
        slope: Land slope in m per texel
        bank1_idx: Left bank sample index
        bank2_idx: Right bank sample index
        micro_relief_sigma: Std of land micro-relief (m)
        profile_len: Total profile length in samples

    Returns:
        (elevations, ideal_lake_mask) arrays
    """
    np.random.seed(43)
    xs = np.arange(profile_len, dtype=np.float64)
    elev = np.full(profile_len, water_level, dtype=np.float64)

    # Left bank land
    left = xs < bank1_idx
    elev[left] = water_level + slope * (bank1_idx - xs[left])

    # Right bank land
    right = xs > bank2_idx
    elev[right] = water_level + slope * (xs[right] - bank2_idx)

    # Micro-relief on land
    relief = np.zeros(profile_len, dtype=np.float64)
    white = np.random.randn(profile_len) * micro_relief_sigma
    alpha = 0.7
    relief[0] = white[0]
    for i in range(1, profile_len):
        relief[i] = alpha * relief[i - 1] + (1 - alpha) * white[i]
    land = left | right
    elev[land] += relief[land]

    # Ideal mask
    ideal = np.where((xs >= bank1_idx) & (xs <= bank2_idx), 1.0, 0.0)

    return elev, ideal


def add_white_noise(elev, sigma=0.05):
    """Add Gaussian white noise (σ meters)."""
    np.random.seed(44)
    return elev + np.random.randn(len(elev)) * sigma


def apply_terrain_rgb_chain(elev):
    """Apply terrain-rgb encode→decode chain (0.1m quantization)."""
    encoded = terrain_rgb_encode(elev)
    return terrain_rgb_decode(encoded)


def generate_synthetic_dataset(conn):
    """Generate synthetic profiles derived from real JAXA statistics.

    For each water body with profiles, extract statistics and generate
    matching synthetic profiles.
    """
    body_rows = conn.execute("""
        SELECT wb.id, wb.name, wb.type, COUNT(p.id)
        FROM water_bodies wb
        JOIN profiles p ON p.water_body_id = wb.id
        GROUP BY wb.id
    """).fetchall()

    for wb_id, name, wb_type, n_profiles in body_rows:
        if wb_type == "lake":
            # Get a representative profile for stats
            prof = conn.execute("""
                SELECT id FROM profiles WHERE water_body_id = ? LIMIT 1
            """, (wb_id,)).fetchone()
            if prof is None:
                continue
            profile_id = prof[0]

            samples = conn.execute("""
                SELECT elev_raw FROM elevation_samples
                WHERE profile_id = ? AND elev_raw IS NOT NULL
                ORDER BY idx
            """, (profile_id,)).fetchall()
            if not samples:
                continue

            elev_raw = np.array([s[0] for s in samples])

            # Find water segment (flattest contiguous region)
            water_level, shore_idx = _detect_water_from_elevation(elev_raw)

            # Land side stats
            land_elev = elev_raw[shore_idx + 10:shore_idx + 100]  # first 90 land samples
            if len(land_elev) < 20:
                continue
            xs = np.arange(len(land_elev), dtype=np.float64)
            slope, _ = np.polyfit(xs, land_elev, 1)
            trend = np.polyval([slope, np.mean(land_elev)], xs)
            micro_relief_sigma = float(np.std(land_elev - trend))

            print(f"  {name}: water_level={water_level:.1f}m, slope={slope:.4f}, micro_relief_σ={micro_relief_sigma:.2f}m")

            # Generate synthetic variants
            for var_idx, (var_name, noise_model) in enumerate([
                ("ideal", None),
                ("white_005", lambda x: add_white_noise(x, 0.05)),
                ("trgb_chain", apply_terrain_rgb_chain),
            ]):
                # 3 profiles with different shoreline positions
                for pi, offset in enumerate([0, -30, 40]):
                    shore = shore_idx + offset
                    shore = max(50, min(400, shore))
                    elev_syn, ideal = generate_lake_profile(
                        water_level=water_level,
                        slope=abs(slope) * 0.7,  # slightly gentler
                        shoreline_idx=shore,
                        micro_relief_sigma=max(0.1, micro_relief_sigma * 0.7),
                        profile_len=512,
                    )

                    if noise_model is not None:
                        elev_syn = noise_model(elev_syn)

                    # Compute terrain-rgb elev
                    elev_trgb = apply_terrain_rgb_chain(elev_syn)

                    label = f"{name}_syn_{var_name}_{pi + 1:02d}"

                    conn.execute("""
                        INSERT INTO profiles (water_body_id, label, direction, sample_count, source)
                        VALUES (?, ?, 'u', 512, 'synthetic')
                    """, (wb_id, label))
                    syn_prof_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                    samples = []
                    for i in range(512):
                        samples.append((
                            syn_prof_id, i, float(i),
                            0.0, 0.0,  # syn profiles: dist only, no lat/lon
                            float(elev_syn[i]),
                            float(elev_trgb[i]),
                        ))
                    conn.executemany(
                        "INSERT INTO elevation_samples (profile_id, idx, dist_m, lon, lat, elev_raw, elev_terrain_rgb) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        samples,
                    )

                    # Store synthetic params
                    conn.execute("""
                        INSERT INTO synthetic_params (profile_id, water_level, slope, micro_relief_sigma, source_profile_id)
                        VALUES (?, ?, ?, ?, ?)
                    """, (syn_prof_id, water_level, float(abs(slope)), micro_relief_sigma, profile_id))

                    # Ground truth: ideal mask boundary
                    conn.execute("""
                        INSERT INTO shore_labels (profile_id, idx_start, idx_end, source, confidence)
                        VALUES (?, ?, ?, 'synthetic', 1.0)
                    """, (syn_prof_id, 0, shore,))

        elif wb_type == "river":
            # Similar for rivers
            prof = conn.execute("""
                SELECT id FROM profiles WHERE water_body_id = ? LIMIT 1
            """, (wb_id,)).fetchone()
            if prof is None:
                continue
            profile_id = prof[0]

            samples = conn.execute("""
                SELECT elev_raw FROM elevation_samples
                WHERE profile_id = ? AND elev_raw IS NOT NULL
                ORDER BY idx
            """, (profile_id,)).fetchall()
            if not samples:
                continue

            elev_raw = np.array([s[0] for s in samples])

            # Detect river segment
            water_level, bank1, bank2 = _detect_river_from_elevation(elev_raw)
            if bank2 - bank1 < 5:
                continue

            land_elev = elev_raw[bank2 + 5:bank2 + 80]
            if len(land_elev) < 10:
                continue
            xs = np.arange(len(land_elev), dtype=np.float64)
            slope, _ = np.polyfit(xs, land_elev, 1)
            trend = np.polyval([slope, np.mean(land_elev)], xs)
            micro_relief_sigma = float(np.std(land_elev - trend))

            print(f"  {name}: water_level={water_level:.1f}m, width={bank2 - bank1}px, slope={slope:.4f}")

            for var_idx, (var_name, noise_model) in enumerate([
                ("ideal", None),
                ("white_005", lambda x: add_white_noise(x, 0.05)),
                ("trgb_chain", apply_terrain_rgb_chain),
            ]):
                for pi, (b1_off, b2_off) in enumerate([(0, 0), (-15, 15), (10, -5)]):
                    b1 = max(30, bank1 + b1_off)
                    b2 = min(470, bank2 + b2_off)
                    elev_syn, ideal = generate_river_profile(
                        water_level=water_level,
                        slope=abs(slope) * 0.7,
                        bank1_idx=b1,
                        bank2_idx=b2,
                        micro_relief_sigma=max(0.1, micro_relief_sigma * 0.7),
                        profile_len=512,
                    )

                    if noise_model is not None:
                        elev_syn = noise_model(elev_syn)

                    elev_trgb = apply_terrain_rgb_chain(elev_syn)
                    label = f"{name}_syn_{var_name}_{pi + 1:02d}"

                    conn.execute("""
                        INSERT INTO profiles (water_body_id, label, direction, sample_count, source)
                        VALUES (?, ?, 'u', 512, 'synthetic')
                    """, (wb_id, label))
                    syn_prof_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                    samples = []
                    for i in range(512):
                        samples.append((syn_prof_id, i, float(i), 0.0, 0.0, float(elev_syn[i]), float(elev_trgb[i])))
                    conn.executemany(
                        "INSERT INTO elevation_samples (profile_id, idx, dist_m, lon, lat, elev_raw, elev_terrain_rgb) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        samples,
                    )

                    conn.execute("""
                        INSERT INTO synthetic_params (profile_id, water_level, slope, micro_relief_sigma, source_profile_id)
                        VALUES (?, ?, ?, ?, ?)
                    """, (syn_prof_id, water_level, float(abs(slope)), micro_relief_sigma, profile_id))

                    conn.execute("""
                        INSERT INTO shore_labels (profile_id, idx_start, idx_end, source, confidence)
                        VALUES (?, ?, ?, 'synthetic', 1.0)
                    """, (syn_prof_id, b1, b2))

    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM profiles WHERE source='synthetic'").fetchone()[0]


def _detect_water_from_elevation(elev, flatness_thresh=0.5, min_water_len=10):
    """Detect lake water segment from elevation profile.

    Returns (water_level, shoreline_idx) where water_level is the mean
    elevation of the flattest segment.
    """
    n = len(elev)
    best_start = 0
    best_end = 0
    best_var = np.inf

    # Sliding window: find lowest-variance contiguous segment
    window = max(min_water_len, n // 8)
    step = max(1, window // 4)
    for start in range(0, n - window, step):
        end = start + window
        seg = elev[start:end]
        var = float(np.var(seg))
        if var < best_var:
            best_var = var
            best_start = start
            best_end = end

    # Expand the flat segment
    water_level = float(np.mean(elev[best_start:best_end]))
    # Walk right to find where elevation starts consistently rising
    shore_idx = best_end
    for i in range(best_end, min(n, best_end + window)):
        if elev[i] > water_level + flatness_thresh:
            shore_idx = i
            break

    return water_level, shore_idx


def _detect_river_from_elevation(elev, flatness_thresh=0.5, min_river_len=8):
    """Detect river water segment from elevation profile.

    Returns (water_level, bank1_idx, bank2_idx).
    """
    n = len(elev)
    window = max(min_river_len, n // 16)
    step = max(1, window // 4)
    best_start = 0
    best_end = 0
    best_var = np.inf

    for start in range(n // 8, n - n // 4 - window, step):
        end = start + window
        seg = elev[start:end]
        var = float(np.var(seg))
        if var < best_var:
            best_var = var
            best_start = start
            best_end = end

    water_level = float(np.mean(elev[best_start:best_end]))

    # Find both banks
    bank1 = best_start
    for i in range(best_start, 0, -1):
        if abs(elev[i] - water_level) > flatness_thresh:
            bank1 = i
            break

    bank2 = best_end
    for i in range(best_end, min(n, best_end + window)):
        if abs(elev[i] - water_level) > flatness_thresh:
            bank2 = i
            break

    return water_level, bank1, bank2


# ============================================================================
# Benchmark runner
# ============================================================================

def compute_metrics(lake_mask, dist_m, shore_start_idx, shore_end_idx, ground_truth_mask=None):
    """Compute the three benchmark metrics.

    Args:
        lake_mask: Algorithm output array [0, 1]
        dist_m: Distance along profile (meters)
        shore_start_idx: Start of water segment in ground truth
        shore_end_idx: End of water segment in ground truth
        ground_truth_mask: Optional ideal binary mask (for synthetic profiles)

    Returns:
        (boundary_offset_m, flood_ratio, fragment_count, rmse, transition_width_m)
    """
    n = len(lake_mask)

    # Ground truth binary mask
    if ground_truth_mask is not None:
        gt = ground_truth_mask
    else:
        gt = np.zeros(n, dtype=np.float64)
        gt[shore_start_idx:shore_end_idx + 1] = 1.0

    rmse = float(np.sqrt(np.mean((lake_mask - gt) ** 2)))

    # Convert lake_mask to binary for fragment/flood analysis
    binary = (lake_mask > 0.5).astype(int)

    # Boundary offset: find first crossing of 0.5 near the right shore
    # (lake → land transition)
    shore_mid = shore_end_idx
    if ground_truth_mask is not None:
        # For synthetic with known boundary
        # Find where gt transitions from 1 to 0
        diffs = np.diff(np.concatenate([[0], gt, [0]]))
        transitions = np.where(diffs != 0)[0]
        if len(transitions) >= 2:
            shore_mid = transitions[-1]  # last 1→0 transition (land side)
        elif len(transitions) == 1:
            shore_mid = transitions[0]

    # Find lake_mask 0.5 crossing nearest to shore_mid
    above = lake_mask > 0.5
    # Find transitions in predicted mask
    pred_diff = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    pred_starts = np.where(pred_diff > 0)[0]
    pred_ends = np.where(pred_diff < 0)[0]

    # The water→land transition closest to the shore
    boundary_offset_m = 0.0
    # Clamp indices to valid range (n-1) to avoid IndexError from diff padding
    pred_ends_clamped = np.clip(pred_ends, 0, n - 1)
    pred_starts_clamped = np.clip(pred_starts, 0, n - 1)
    shore_mid = min(shore_mid, n - 1)
    if len(pred_ends) > 0:
        distances = np.abs(pred_ends_clamped - shore_mid)
        closest = pred_ends_clamped[np.argmin(distances)]
        boundary_offset_m = float(dist_m[closest] - dist_m[shore_mid])
    elif len(pred_starts) > 0:
        distances = np.abs(pred_starts_clamped - shore_mid)
        closest = pred_starts_clamped[np.argmin(distances)]
        boundary_offset_m = float(dist_m[closest] - dist_m[shore_mid])

    # Flood ratio: land-side lakeMask > 0.5
    land_start = shore_end_idx + 1
    land_end = min(n, land_start + n // 4)  # check 25% of profile into land
    land_window = slice(land_start, land_end)
    n_land = max(1, land_end - land_start)
    flood_ratio = float(np.mean(lake_mask[land_window] > 0.5))

    # Fragment count: number of water blocks on land
    binary_land = binary[land_window]
    land_diff = np.diff(np.concatenate([[0], binary_land, [0]]))
    fragment_starts = np.where(land_diff > 0)[0]
    fragment_ends = np.where(land_diff < 0)[0]
    fragments = []
    for s, e in zip(fragment_starts, fragment_ends):
        if e > s:  # at least 1 sample
            fragments.append((s, e))
    fragment_count = len(fragments)

    # Transition width (distance between lakeMask=0.1 and 0.9 near boundary)
    transition_width_m = 0.0
    near_bound = slice(max(0, shore_mid - n // 8), min(n, shore_mid + n // 8))
    mask_near = lake_mask[near_bound]
    above_09 = np.where(mask_near > 0.9)[0]
    below_01 = np.where(mask_near < 0.1)[0]
    if len(above_09) > 0 and len(below_01) > 0:
        last_water_idx = max(0, shore_mid - n // 8) + above_09[-1]
        first_land_idx = max(0, shore_mid - n // 8) + below_01[0]
        if first_land_idx > last_water_idx:
            transition_width_m = float(dist_m[first_land_idx] - dist_m[last_water_idx])

    return boundary_offset_m, flood_ratio, fragment_count, rmse, transition_width_m


def run_benchmark(conn, profile_id=None, skip_synthetic=False):
    """Run all algorithm configs against all profiles.

    Args:
        conn: SQLite connection
        profile_id: Optional single profile ID to test
        skip_synthetic: If True, skip synthetic profiles
    """
    # Load algorithm configs
    algo_configs = build_algorithm_configs()
    print(f"Algorithm configs: {len(algo_configs)}")

    # Insert algorithm configs into DB
    for label, params in algo_configs:
        conn.execute("""
            INSERT OR IGNORE INTO algorithm_configs (label, grad_op, norm, window_stat, pre_smooth, multi_scale, decision_fn, elev_guard, extra_params)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'E3', ?)
        """, (
            label,
            params.get("grad_op"),
            params.get("norm"),
            params.get("window_stat"),
            params.get("pre_smooth"),
            params.get("multi_scale"),
            params.get("decision_fn"),
            json.dumps({k: v for k, v in params.items()
                       if k not in ("grad_op", "norm", "window_stat", "pre_smooth", "multi_scale", "decision_fn")}),
        ))
    conn.commit()

    # Load profiles
    where = "WHERE p.source != 'synthetic'" if skip_synthetic else ""
    if profile_id is not None:
        where = f"WHERE p.id = {profile_id}"
    profile_rows = conn.execute(f"""
        SELECT p.id, p.label, p.water_body_id, p.source, wb.type
        FROM profiles p
        JOIN water_bodies wb ON wb.id = p.water_body_id
        {where}
        ORDER BY p.id
    """).fetchall()

    print(f"Profiles: {len(profile_rows)}")

    total_runs = len(algo_configs) * len(profile_rows)
    run_count = 0
    t0 = time.time()

    for prof_id, prof_label, wb_id, source, wb_type in profile_rows:
        # Load elevation
        samples = conn.execute("""
            SELECT idx, dist_m, elev_raw, elev_terrain_rgb
            FROM elevation_samples
            WHERE profile_id = ?
            ORDER BY idx
        """, (prof_id,)).fetchall()

        if not samples:
            print(f"  SKIP profile {prof_id}: no samples")
            continue

        idxs = np.array([s[0] for s in samples])
        dists = np.array([s[1] for s in samples])
        elev_raw = np.array([s[2] for s in samples])
        elev_trgb = np.array([s[3] for s in samples])

        # Load shore labels
        labels = conn.execute("""
            SELECT idx_start, idx_end FROM shore_labels WHERE profile_id = ?
        """, (prof_id,)).fetchall()

        if not labels:
            print(f"  SKIP profile {prof_id}: no shore labels")
            continue

        shore_start = labels[0][0]
        shore_end = labels[0][1]

        # For synthetic, use the full elevation as both raw and trgb
        # For real, test against both raw and trgb

        for elev_arr, noise_label in [
            (elev_raw, "raw"),
            (elev_trgb, "trgb"),
        ]:
            if np.all(np.isnan(elev_arr)):
                continue
            # Remove NaN padding
            valid = ~np.isnan(elev_arr)
            if not valid.all():
                valid_idxs = np.where(valid)[0]
                arr = elev_arr[valid_idxs[0]:valid_idxs[-1] + 1]
            else:
                arr = elev_arr

            # Fill any interior NaN
            arr = np.array(arr)
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                arr[nan_mask] = np.interp(
                    np.flatnonzero(nan_mask),
                    np.flatnonzero(~nan_mask),
                    arr[~nan_mask],
                )

            for algo_label, algo_params in algo_configs:
                if profile_id is not None and run_count % 50 == 0:
                    print(f"  [{run_count}/{total_runs}] {prof_label} × {algo_label}")

                # Run algorithm
                try:
                    # Filter out params unknown to compute_lake_mask
                    valid_keys = {"flatness", "lake_range", "texture_size",
                                   "grad_op", "norm", "window_stat",
                                   "pre_smooth", "multi_scale", "decision_fn",
                                   "decision_k", "interp_mode"}
                    filtered_params = {k: v for k, v in algo_params.items() if k in valid_keys}
                    lake_mask = compute_lake_mask(arr, **filtered_params)
                except Exception as e:
                    print(f"  ERROR: {prof_label} × {algo_label}: {e}")
                    continue

                # Truncate lake_mask to match original lengths for metric computation
                mask_for_metrics = lake_mask[:len(dists)] if len(lake_mask) >= len(dists) else np.pad(lake_mask, (0, len(dists) - len(lake_mask)), 'edge')

                # For synthetic profiles, use the full ground truth
                gt = None
                if source == "synthetic":
                    gt = np.where(idxs < shore_start, 1.0, 0.0)
                    if wb_type == "river":
                        gt = np.where((idxs >= shore_start) & (idxs <= shore_end), 1.0, 0.0)

                offset_m, flood_r, frag_n, rmse, tw = compute_metrics(
                    mask_for_metrics, dists, shore_start, shore_end, gt,
                )

                # Get algorithm config ID
                algo_row = conn.execute(
                    "SELECT id FROM algorithm_configs WHERE label = ?", (algo_label,)
                ).fetchone()
                if algo_row is None:
                    continue
                algo_id = algo_row[0]

                # Insert benchmark run
                conn.execute("""
                    INSERT INTO benchmark_runs (algorithm_id, profile_id, noise_model, run_at)
                    VALUES (?, ?, ?, datetime('now'))
                """, (algo_id, prof_id, noise_label))
                run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                # Insert metrics
                conn.execute("""
                    INSERT INTO benchmark_metrics (run_id, boundary_offset_m, flood_ratio, fragment_count, lake_mask_rmse, transition_width_m)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (run_id, offset_m, flood_r, frag_n, rmse, tw))

                # Insert lake_mask detail (sample every 4th point to save space)
                step = max(1, len(idxs) // 256)
                detail_samples = []
                for i in range(0, len(idxs), step):
                    if i < len(mask_for_metrics):
                        detail_samples.append((
                            run_id, int(idxs[i]), float(dists[i]),
                            float(arr[min(i, len(arr) - 1)]),
                            float(mask_for_metrics[i]),
                        ))
                conn.executemany(
                    "INSERT INTO lake_mask_detail (run_id, idx, dist_m, elevation, lake_mask) VALUES (?, ?, ?, ?, ?)",
                    detail_samples,
                )

                run_count += 1

        conn.commit()

    elapsed = time.time() - t0
    print(f"\nDone. {run_count} runs in {elapsed:.1f}s ({run_count / elapsed:.1f}/s)")


def run_dimension_sweep(conn):
    """Run per-dimension sweep analysis.

    Groups benchmark results by dimension and computes summary statistics.
    """
    dimensions = {
        "grad_op": ["G1", "G2", "G3", "G4", "G5", "G6"],
        "pre_smooth": ["S0", "S1", "S2"],
        "multi_scale": ["M1", "M2_DEFAULT", "M3_DEFAULT", "M4_DEFAULT"],
        "decision_fn": ["D1", "D2", "D3"],
    }

    conn.execute("DELETE FROM dim_sweep_results")

    for dim_name, variants in dimensions.items():
        for variant in variants:
            # Query: get all runs with this dimension value (keeping others baseline)
            # Use label pattern matching
            if dim_name == "grad_op":
                pattern = f"{variant}_N1_S0_M1_D1_E3"
            elif dim_name == "pre_smooth":
                pattern = f"G1_N1_S0_{variant}_M1_D1_E3"
            elif dim_name == "multi_scale":
                # Match any M2, M3, M4 label
                ms_part = variant.split("_")[0]
                pattern = f"%_{ms_part}_%"
            elif dim_name == "decision_fn":
                pattern = f"G1_N1_S0_M1_{variant}_%_E3"
            else:
                continue

            rows = conn.execute("""
                SELECT AVG(bm.boundary_offset_m), AVG(bm.flood_ratio), AVG(bm.fragment_count)
                FROM benchmark_metrics bm
                JOIN benchmark_runs br ON br.id = bm.run_id
                JOIN algorithm_configs ac ON ac.id = br.algorithm_id
                WHERE ac.label LIKE ?
            """, (pattern,)).fetchone()

            if rows and rows[0] is not None:
                conn.execute("""
                    INSERT INTO dim_sweep_results (dim_name, variant_id, avg_offset_m, avg_flood_ratio, avg_fragment_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (dim_name, variant, rows[0], rows[1], rows[2]))

    conn.commit()

    # Compute ranks
    for dim_name in dimensions:
        rows = conn.execute("""
            SELECT id, avg_offset_m, avg_flood_ratio FROM dim_sweep_results
            WHERE dim_name = ? ORDER BY ABS(avg_offset_m) ASC
        """, (dim_name,)).fetchall()
        for rank, (row_id, _, _) in enumerate(rows, 1):
            conn.execute("UPDATE dim_sweep_results SET rank_offset = ? WHERE id = ?", (rank, row_id))

        rows = conn.execute("""
            SELECT id, avg_flood_ratio FROM dim_sweep_results
            WHERE dim_name = ? AND avg_flood_ratio IS NOT NULL ORDER BY avg_flood_ratio ASC
        """, (dim_name,)).fetchall()
        for rank, (row_id, _) in enumerate(rows, 1):
            conn.execute("UPDATE dim_sweep_results SET rank_flood = ? WHERE id = ?", (rank, row_id))

    conn.commit()


def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    profile_id = None
    skip_synthetic = False
    do_sweep = False

    for arg in sys.argv[1:]:
        if arg.startswith("--profile-id="):
            profile_id = int(arg.split("=")[1])
        elif arg == "--no-synthetic":
            skip_synthetic = True
        elif arg == "--dim=sweep":
            do_sweep = True

    if do_sweep:
        print("Running per-dimension sweep...")
        run_dimension_sweep(conn)
        print("Sweep complete.")
    else:
        # Check if synthetic profiles need generating
        syn_count = conn.execute("SELECT COUNT(*) FROM profiles WHERE source='synthetic'").fetchone()[0]
        if syn_count == 0 and not skip_synthetic:
            print("Generating synthetic profiles...")
            n_syn = generate_synthetic_dataset(conn)
            print(f"Generated {n_syn} synthetic profiles")

        print("Running benchmarks...")
        run_benchmark(conn, profile_id=profile_id, skip_synthetic=skip_synthetic)

        # Always run sweep after benchmark
        print("Running dimension sweep...")
        run_dimension_sweep(conn)

    conn.close()


if __name__ == "__main__":
    main()
