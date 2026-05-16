#!/usr/bin/env python3
"""Plot benchmark results from SQLite database.

Usage:
  python scripts/plot_benchmark.py                    # all charts
  python scripts/plot_benchmark.py --scatter          # overview scatter only
  python scripts/plot_benchmark.py --profile-id 3     # single profile detail
"""

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "tmp" / "lake_benchmark" / "lake_boundary.db"
CHARTS_DIR = PROJECT_ROOT / "tmp" / "lake_benchmark" / "charts"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    import sys
    sys.exit(1)

# Set Chinese font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def plot_scatter_overview(conn):
    """Overview scatter plot: boundary offset vs flood ratio for all algorithm configs."""
    rows = conn.execute("""
        SELECT ac.label, bm.boundary_offset_m, bm.flood_ratio, bm.fragment_count, bm.lake_mask_rmse
        FROM benchmark_metrics bm
        JOIN benchmark_runs br ON br.id = bm.run_id
        JOIN algorithm_configs ac ON ac.id = br.algorithm_id
        JOIN profiles p ON p.id = br.profile_id
        WHERE p.source = 'synthetic'
    """).fetchall()

    if not rows:
        print("  No data for scatter plot")
        return

    data = np.array([(abs(r[1]), r[2], r[3], r[4]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Offset vs Flood
    ax = axes[0]
    sc = ax.scatter(np.abs(data[:, 0]), data[:, 1], c=data[:, 3], cmap="RdYlGn_r", alpha=0.6, s=30)
    ax.set_xlabel("|Boundary Offset| (m)")
    ax.set_ylabel("Flood Ratio (land→water)")
    ax.set_title("Land Flood vs Boundary Offset")
    ax.axhline(0.2, color="red", linestyle="--", alpha=0.5, label="flood limit")
    ax.axvline(60, color="orange", linestyle="--", alpha=0.5, label="offset limit")
    ax.legend(fontsize="small")
    plt.colorbar(sc, ax=ax, label="RMSE")

    # 2. Offset vs Fragment count
    ax = axes[1]
    ax.scatter(np.abs(data[:, 0]), data[:, 2], alpha=0.6, s=30)
    ax.set_xlabel("|Boundary Offset| (m)")
    ax.set_ylabel("Fragment Count on Land")
    ax.set_title("Fragmentation vs Boundary Offset")
    ax.axhline(3, color="red", linestyle="--", alpha=0.5, label="fragment limit")

    # 3. Scatter by algorithm family
    ax = axes[2]
    families = {}
    for row in rows:
        label = row[0]
        # Extract algorithm family
        parts = label.split("_")
        if "W1" in label or "W2" in label or "W3" in label or "W4" in label:
            fam = "window_stat"
        elif "S1" in label or "S2" in label:
            fam = "pre_smooth"
        elif "M2" in label or "M3" in label or "M4" in label:
            fam = "multi_scale"
        elif "D2" in label or "D3" in label:
            fam = "decision"
        elif "f20" in label or "f50" in label or "f200" in label or "f500" in label:
            fam = "flatness"
        else:
            fam = "baseline"

        if fam not in families:
            families[fam] = {"x": [], "y": []}
        families[fam]["x"].append(abs(row[1]))
        families[fam]["y"].append(row[2])

    for fam_name, pts in families.items():
        ax.scatter(pts["x"], pts["y"], alpha=0.6, s=30, label=fam_name)

    ax.set_xlabel("|Boundary Offset| (m)")
    ax.set_ylabel("Flood Ratio")
    ax.set_title("By Algorithm Family")
    ax.legend(fontsize="xx-small", ncol=2)

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(CHARTS_DIR / "scatter_overview.png"), dpi=150)
    plt.close(fig)
    print("  scatter_overview.png saved")


def plot_dimension_chart(conn, dim_name, title, xlabel, variant_names):
    """Bar chart for a single dimension sweep."""
    rows = conn.execute("""
        SELECT variant_id, avg_offset_m, avg_flood_ratio, avg_fragment_count
        FROM dim_sweep_results
        WHERE dim_name = ?
        ORDER BY id
    """, (dim_name,)).fetchall()

    if not rows:
        print(f"  No data for dimension: {dim_name}")
        return

    variants = [r[0] for r in rows]
    offsets = [abs(r[1]) if r[1] else 0 for r in rows]
    floods = [r[2] if r[2] else 0 for r in rows]
    frags = [r[3] if r[3] else 0 for r in rows]

    x = np.arange(len(variants))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(12, 5))

    bars1 = ax1.bar(x - width, offsets, width, label="Avg |Offset| (m)", color="steelblue")
    ax1.set_ylabel("Avg |Boundary Offset| (m)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, floods, width, label="Avg Flood Ratio", color="coral")
    bars3 = ax2.bar(x + width, frags, width, label="Avg Fragment Count", color="seagreen")
    ax2.set_ylabel("Ratio / Count", color="dimgray")

    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=8)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize="small")

    plt.tight_layout()
    fig.savefig(str(CHARTS_DIR / f"dim_{dim_name}.png"), dpi=150)
    plt.close(fig)
    print(f"  dim_{dim_name}.png saved")


def plot_dimension_charts(conn):
    """Generate all per-dimension charts."""
    dims = [
        ("grad_op", "Gradient Operator Comparison", "Gradient Operator"),
        ("pre_smooth", "Pre-Smoothing Comparison", "Pre-Smooth Method"),
        ("decision_fn", "Decision Function Comparison", "Decision Function"),
    ]
    for dim_name, title, xlabel in dims:
        plot_dimension_chart(conn, dim_name, title, xlabel, [])


def plot_profile_detail(conn, profile_id):
    """Plot detailed algorithm comparison on a single profile."""
    # Load profile
    prof = conn.execute("""
        SELECT p.label, wb.name, wb.type, p.source
        FROM profiles p
        JOIN water_bodies wb ON wb.id = p.water_body_id
        WHERE p.id = ?
    """, (profile_id,)).fetchone()

    if not prof:
        print(f"  Profile {profile_id} not found")
        return

    prof_label, wb_name, wb_type, source = prof

    # Load elevation
    samples = conn.execute("""
        SELECT idx, dist_m, elev_raw FROM elevation_samples
        WHERE profile_id = ? ORDER BY idx
    """, (profile_id,)).fetchall()

    if not samples:
        return

    idxs = np.array([s[0] for s in samples])
    dists = np.array([s[1] for s in samples])
    elev = np.array([s[2] for s in samples])

    # Load shore labels
    labels = conn.execute("""
        SELECT idx_start, idx_end FROM shore_labels WHERE profile_id = ?
    """, (profile_id,)).fetchall()

    # Pick top 6 algorithms (lowest RMSE)
    top_runs = conn.execute("""
        SELECT br.id, ac.label, bm.boundary_offset_m, bm.flood_ratio, bm.lake_mask_rmse
        FROM benchmark_runs br
        JOIN algorithm_configs ac ON ac.id = br.algorithm_id
        JOIN benchmark_metrics bm ON bm.run_id = br.id
        WHERE br.profile_id = ? AND br.noise_model = 'raw'
        ORDER BY bm.lake_mask_rmse ASC
        LIMIT 6
    """, (profile_id,)).fetchall()

    if not top_runs:
        print(f"  No benchmark results for profile {profile_id}")
        return

    fig, (ax_elev, ax_mask) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Elevation profile
    ax_elev.plot(dists, elev, "k-", linewidth=1.0, alpha=0.7, label="Elevation")
    ax_elev.set_ylabel("Elevation (m)")
    ax_elev.set_title(f"{wb_name} — {prof_label} ({source})")
    ax_elev.legend(fontsize="small")

    # Mark shore labels
    for s, e in labels:
        if s < len(dists) and e < len(dists):
            ax_elev.axvspan(dists[s], dists[min(e, len(dists) - 1)], alpha=0.15, color="cyan", label="Water (ground truth)")

    # Lake masks
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_runs)))
    for (run_id, algo_label, offset, flood, rmse), color in zip(top_runs, colors):
        detail = conn.execute("""
            SELECT idx, lake_mask FROM lake_mask_detail
            WHERE run_id = ? ORDER BY idx
        """, (run_id,)).fetchall()

        if not detail:
            continue

        d_idx = np.array([d[0] for d in detail])
        d_dists = dists[np.minimum(d_idx, len(dists) - 1)]
        d_mask = np.array([d[1] for d in detail])

        short_label = algo_label[:30] if len(algo_label) > 30 else algo_label
        ax_mask.plot(d_dists, d_mask, "-", color=color, linewidth=1.0, alpha=0.8,
                     label=f"{short_label} (off={offset:.1f}m, RMSE={rmse:.3f})")

    ax_mask.set_ylabel("Lake Mask")
    ax_mask.set_xlabel("Distance (m)")
    ax_mask.set_ylim(-0.05, 1.05)
    ax_mask.legend(fontsize="xx-small", ncol=2)
    ax_mask.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(str(CHARTS_DIR / f"profile_{profile_id:03d}.png"), dpi=150)
    plt.close(fig)
    print(f"  profile_{profile_id:03d}.png saved ({wb_name} — {prof_label})")


def plot_all_profile_details(conn, max_profiles=20):
    """Plot detail charts for all synthetic profiles."""
    prof_rows = conn.execute("""
        SELECT DISTINCT p.id FROM profiles p
        JOIN benchmark_runs br ON br.profile_id = p.id
        WHERE p.source = 'synthetic'
        ORDER BY p.id
        LIMIT ?
    """, (max_profiles,)).fetchall()

    for (pid,) in prof_rows:
        plot_profile_detail(conn, pid)


def main():
    conn = sqlite3.connect(str(DB_PATH))
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    args = sys.argv[1:]
    if "--scatter" in args:
        print("Plotting overview scatter...")
        plot_scatter_overview(conn)
        plot_dimension_charts(conn)
        return

    if "--profile-id" in args:
        idx = args.index("--profile-id")
        pid = int(args[idx + 1]) if idx + 1 < len(args) else None
        if pid:
            plot_profile_detail(conn, pid)
            return

    # Full plot suite
    print("Plotting overview scatter...")
    plot_scatter_overview(conn)

    print("Plotting dimension charts...")
    plot_dimension_charts(conn)

    print("Plotting profile details...")
    plot_all_profile_details(conn)

    conn.close()
    print(f"\nAll charts saved to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
