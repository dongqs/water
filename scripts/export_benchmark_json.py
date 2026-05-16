#!/usr/bin/env python3
"""Export lake boundary benchmark SQLite data to JSON for the frontend dashboard.

Outputs three files under tmp/lake_benchmark/export/:
  meta.json      — water_bodies, profiles, algorithm_configs
  metrics.json   — all benchmark_metrics joined with metadata (flat array)
  dim_sweep.json — per-dimension aggregated statistics

Usage:
  python scripts/export_benchmark_json.py
"""

import json
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "tmp" / "lake_benchmark" / "lake_boundary.db"
EXPORT_DIR = PROJECT_ROOT / "tmp" / "lake_benchmark" / "export"


def export_meta(conn):
    """Export water_bodies, profiles, algorithm_configs."""
    water_bodies = []
    for row in conn.execute("SELECT id, name, type, center_lat, center_lon FROM water_bodies"):
        water_bodies.append({
            "id": row[0], "name": row[1], "type": row[2],
            "center_lat": row[3], "center_lon": row[4],
        })

    profiles = []
    for row in conn.execute("""
        SELECT p.id, p.water_body_id, p.label, p.direction, p.sample_count, p.source
        FROM profiles p
    """):
        profiles.append({
            "id": row[0], "water_body_id": row[1], "label": row[2],
            "direction": row[3], "sample_count": row[4], "source": row[5],
        })

    # Get elevation samples for profile detail view
    # Store as {profile_id: {dist_m: [], elev_raw: [], shore_start: int, shore_end: int}}
    # We only need this for the profile detail charts
    profile_data = {}
    for row in conn.execute("SELECT id, water_body_id, source FROM profiles"):
        pid, wb_id, source = row
        samples = conn.execute(
            "SELECT dist_m, elev_raw FROM elevation_samples WHERE profile_id = ? ORDER BY idx",
            (pid,)
        ).fetchall()
        if not samples:
            continue
        labels = conn.execute(
            "SELECT idx_start, idx_end FROM shore_labels WHERE profile_id = ?",
            (pid,)
        ).fetchone()
        dists = [s[0] for s in samples]
        elevs = [s[1] if s[1] is not None else 0.0 for s in samples]
        profile_data[str(pid)] = {
            "water_body_id": wb_id,
            "source": source,
            "dist_m": dists,
            "elev_raw": elevs,
            "shore_start": labels[0] if labels else None,
            "shore_end": labels[1] if labels else None,
        }

    algo_configs = []
    for row in conn.execute("""
        SELECT id, label, grad_op, norm, window_stat, pre_smooth, multi_scale,
               decision_fn, elev_guard, extra_params
        FROM algorithm_configs
    """):
        algo_configs.append({
            "id": row[0], "label": row[1],
            "grad_op": row[2], "norm": row[3], "window_stat": row[4],
            "pre_smooth": row[5], "multi_scale": row[6],
            "decision_fn": row[7], "elev_guard": row[8],
            "extra_params": json.loads(row[9]) if row[9] else {},
        })

    meta = {
        "water_bodies": water_bodies,
        "profiles": profiles,
        "profile_data": profile_data,
        "algorithm_configs": algo_configs,
    }

    with open(EXPORT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"  meta.json: {len(water_bodies)} bodies, {len(profiles)} profiles, "
          f"{len(algo_configs)} algos, {len(profile_data)} profile_data entries")


def export_metrics(conn):
    """Export all benchmark_metrics as a flat joined array."""
    rows = conn.execute("""
        SELECT
          br.id AS run_id,
          ac.id AS algorithm_id, ac.label AS algorithm_label,
          ac.grad_op, ac.norm, ac.window_stat, ac.pre_smooth, ac.multi_scale,
          ac.decision_fn, ac.elev_guard,
          p.id AS profile_id, p.label AS profile_label, p.source,
          wb.id AS water_body_id, wb.name AS water_body_name, wb.type AS water_body_type,
          br.noise_model,
          bm.boundary_offset_m, bm.flood_ratio, bm.fragment_count,
          bm.lake_mask_rmse, bm.transition_width_m
        FROM benchmark_metrics bm
        JOIN benchmark_runs br ON br.id = bm.run_id
        JOIN algorithm_configs ac ON ac.id = br.algorithm_id
        JOIN profiles p ON p.id = br.profile_id
        JOIN water_bodies wb ON wb.id = p.water_body_id
        ORDER BY br.id
    """).fetchall()

    metrics = []
    for r in rows:
        metrics.append({
            "run_id": r[0],
            "algorithm_id": r[1], "algorithm_label": r[2],
            "grad_op": r[3], "norm": r[4], "window_stat": r[5],
            "pre_smooth": r[6], "multi_scale": r[7],
            "decision_fn": r[8], "elev_guard": r[9],
            "profile_id": r[10], "profile_label": r[11], "profile_source": r[12],
            "water_body_id": r[13], "water_body_name": r[14], "water_body_type": r[15],
            "noise_model": r[16],
            "boundary_offset_m": r[17], "flood_ratio": r[18], "fragment_count": r[19],
            "lake_mask_rmse": r[20], "transition_width_m": r[21],
        })

    with open(EXPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"  metrics.json: {len(metrics)} rows")


def compute_dim_sweep(metrics):
    """Compute per-dimension aggregated stats from metrics."""
    import statistics

    dimensions = {
        "grad_op": [("G1", "Central Diff"), ("G2", "Forward Diff"),
                     ("G3", "3-Point"), ("G4", "5-Point"),
                     ("G5", "Sobel"), ("G6", "Prewitt")],
        "pre_smooth": [("S0", "None"), ("S1", "Box 3x3"), ("S2", "Gaussian")],
        "multi_scale": [("M1", "Single"), ("M2", "Large Range"),
                         ("M3", "Min"), ("M4", "Product")],
        "decision_fn": [("D1", "Smoothstep"), ("D2", "Hard Step"), ("D3", "Var Smooth")],
    }

    # For baseline filtering: we want records where only the target dimension varies
    # Use synthetic profiles and raw noise for cleaner comparison
    base = [m for m in metrics if m["profile_source"] == "synthetic" and m["noise_model"] == "raw"]

    sweeps = []
    for dim_name, variants in dimensions.items():
        for variant_id, variant_label in variants:
            # Filter metrics matching this variant
            matching = [m for m in base if m[dim_name] == variant_id]
            if not matching:
                continue

            offsets = [abs(m["boundary_offset_m"]) for m in matching]
            floods = [m["flood_ratio"] for m in matching]
            frags = [m["fragment_count"] for m in matching]
            rmses = [m["lake_mask_rmse"] for m in matching]

            sweeps.append({
                "dim_name": dim_name,
                "variant_id": variant_id,
                "variant_label": variant_label,
                "count": len(matching),
                "avg_offset_m": round(statistics.mean(offsets), 1),
                "avg_flood_ratio": round(statistics.mean(floods), 4),
                "avg_fragment_count": round(statistics.mean(frags), 2),
                "avg_rmse": round(statistics.mean(rmses), 4),
            })

    # Sort each dimension by RMSE
    sweeps.sort(key=lambda s: (s["dim_name"], s["avg_rmse"]))

    with open(EXPORT_DIR / "dim_sweep.json", "w") as f:
        json.dump(sweeps, f)
    print(f"  dim_sweep.json: {len(sweeps)} entries")


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # not used, but good practice

    print(f"Exporting from {DB_PATH}")
    print(f"Exporting to {EXPORT_DIR}")

    export_meta(conn)
    export_metrics(conn)

    # Load metrics back to compute sweep
    with open(EXPORT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    compute_dim_sweep(metrics)

    conn.close()

    # Print file sizes
    for name in ["meta.json", "metrics.json", "dim_sweep.json"]:
        path = EXPORT_DIR / name
        size_kb = path.stat().st_size / 1024
        print(f"  {name}: {size_kb:.1f} KB")

    print("Done.")


if __name__ == "__main__":
    main()
