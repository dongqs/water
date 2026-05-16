#!/usr/bin/env python3
"""Extract 64×64 elevation patches around water body center points.

No OSM dependency — uses local JAXA GeoTIFFs only.
Outputs .npz patches + JSON metadata for Canvas visualization.

Usage:
  python scripts/build_patch_dataset.py
"""

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "tmp" / "lake_boundary_dataset"
PATCHES_DIR = OUTPUT_DIR / "patches"
JAXA_STORAGE = Path("/home/dev/jaxa_storage")
CACHE_DIR = PROJECT_ROOT / "tmp" / "jaxa_cache"

PATCH_SIZE = 64
PATCH_RES_M = 30.0  # ~1 arc-second

WATER_BODIES = [
    ("西湖", "lake", 30.25, 120.14, 15, 15000),
    ("太湖", "lake", 31.20, 120.20, 15, 30000),
    ("青海湖", "lake", 36.90, 100.20, 15, 50000),
    ("洞庭湖", "lake", 29.30, 112.90, 15, 30000),
    ("钱塘江", "river", 30.20, 120.30, 15, 20000),
    ("黄浦江", "river", 31.23, 121.49, 15, 15000),
    ("长江-南京段", "river", 32.05, 118.75, 15, 30000),
    ("长江-武汉段", "river", 30.58, 114.30, 15, 30000),
]

# ---- Tile index ----
_TILE_INDEX = None

def build_tile_index():
    global _TILE_INDEX
    _TILE_INDEX = {}
    for entry in JAXA_STORAGE.iterdir():
        if not entry.is_dir():
            continue
        for f in entry.iterdir():
            name = f.name
            if name.endswith(".tif.gz"):
                tile_name = name[:-7]
            elif name.endswith(".tif"):
                tile_name = name[:-4]
            else:
                continue
            if tile_name not in _TILE_INDEX or str(f).endswith(".tif"):
                _TILE_INDEX[tile_name] = f


def get_tif_path(lat, lon):
    global _TILE_INDEX
    if _TILE_INDEX is None:
        build_tile_index()
    n = f"N{int(np.floor(lat)):03d}"
    e = f"E{int(np.floor(lon)):03d}" if lon >= 0 else f"W{int(-np.floor(lon)):03d}"
    return _TILE_INDEX.get(f"{n}{e}")


def open_tif(path):
    s = str(path)
    if s.endswith(".gz"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / Path(s).stem
        if not cache.exists():
            with gzip.open(s, "rb") as gz:
                with open(cache, "wb") as out:
                    out.write(gz.read())
        s = str(cache)
    return rasterio.open(s)


def extract_grid(lat_ctr, lon_ctr, n_patches):
    """Extract n_patches × n_patches grid of 64×64 elevation patches.

    Centers a grid on (lat_ctr, lon_ctr), each patch covers ~1.92km×1.92km.
    Returns list of (patch_2d, patch_lat, patch_lon, grid_row, grid_col).
    """
    half_deg = (PATCH_SIZE / 2) * PATCH_RES_M / 111320.0
    cos_lat = np.cos(np.radians(lat_ctr))
    step_deg = PATCH_SIZE * PATCH_RES_M / 111320.0  # ~0.017°

    patches = []
    offset = (n_patches - 1) / 2

    for gr in range(n_patches):
        for gc in range(n_patches):
            lat = lat_ctr + (offset - gr) * step_deg
            lon = lon_ctr + (gc - offset) * step_deg / cos_lat

            # Extract
            lat_min = lat - half_deg
            lat_max = lat + half_deg
            lon_min = lon - half_deg / cos_lat
            lon_max = lon + half_deg / cos_lat

            elev = np.full((PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)

            # Determine tile range needed
            tile_lats = [int(np.floor(lat_min)), int(np.floor(lat_max))]
            tile_lons = [int(np.floor(lon_min)), int(np.floor(lon_max))]

            for tl in range(min(tile_lats), max(tile_lats) + 1):
                for tlo in range(min(tile_lons), max(tile_lons) + 1):
                    tif_path = get_tif_path(tl + 0.5, tlo + 0.5)
                    if tif_path is None:
                        continue
                    try:
                        ds = open_tif(tif_path)
                    except Exception:
                        continue

                    r_min, c_min = rowcol(ds.transform, max(lon_min, tlo), min(lat_max, tl + 1))
                    r_max, c_max = rowcol(ds.transform, min(lon_max, tlo + 1), max(lat_min, tl))
                    r_min, c_min = max(0, int(r_min)), max(0, int(c_min))
                    r_max, c_max = min(ds.height, int(r_max) + 1), min(ds.width, int(c_max) + 1)
                    if r_max <= r_min or c_max <= c_min:
                        ds.close()
                        continue

                    data = ds.read(1, window=((r_min, r_max), (c_min, c_max)))
                    ds.close()

                    # Map to output grid
                    lats_tile = np.linspace(
                        ds.transform[5] + r_min * ds.transform[4],
                        ds.transform[5] + (r_max - 1) * ds.transform[4],
                        r_max - r_min,
                    ) if r_max > r_min else np.array([])
                    lons_tile = np.linspace(
                        ds.transform[2] + c_min * ds.transform[0],
                        ds.transform[2] + (c_max - 1) * ds.transform[0],
                        c_max - c_min,
                    ) if c_max > c_min else np.array([])

                    for tr in range(r_max - r_min):
                        out_r = int(round((lat_max - lats_tile[tr]) / (lat_max - lat_min) * (PATCH_SIZE - 1)))
                        if out_r < 0 or out_r >= PATCH_SIZE:
                            continue
                        for tc in range(c_max - c_min):
                            out_c = int(round((lons_tile[tc] - lon_min) / (lon_max - lon_min) * (PATCH_SIZE - 1)))
                            if out_c < 0 or out_c >= PATCH_SIZE:
                                continue
                            elev[out_r, out_c] = float(data[tr, tc])

            patches.append({
                "lat": lat, "lon": lon,
                "grid_row": gr, "grid_col": gc,
                "elevation": elev if not np.all(np.isnan(elev)) else None,
            })

    return patches


def main():
    build_tile_index()
    print(f"Tile index: {len(_TILE_INDEX)} tiles")

    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    all_meta = []

    for wb_idx, (wb_name, wb_type, c_lat, c_lon, n_patches, _) in enumerate(WATER_BODIES):
        t0 = time.time()
        patches = extract_grid(c_lat, c_lon, n_patches)
        saved = 0
        for pi, p in enumerate(patches):
            if p["elevation"] is None:
                continue
            # ASCII-safe file names (avoid URL encoding issues)
            wb_id = wb_name  # keep for metadata
            fname_npz = f"wb{wb_idx:02d}_{pi:03d}.npz"
            fname_bin = f"wb{wb_idx:02d}_{pi:03d}.bin"
            elev_f32 = p["elevation"].astype(np.float32)

            # Save .npz (Python use)
            np.savez_compressed(
                PATCHES_DIR / fname_npz,
                elevation=elev_f32,
                lat=np.float32(p["lat"]),
                lon=np.float32(p["lon"]),
            )

            # Save .bin (JS use: raw float32 little-endian, row-major)
            with open(PATCHES_DIR / fname_bin, "wb") as f:
                f.write(elev_f32.tobytes())

            all_meta.append({
                "file_bin": fname_bin, "file_npz": fname_npz, "wb_id": wb_idx,
                "water_body": wb_name, "wb_type": wb_type,
                "lat": float(p["lat"]), "lon": float(p["lon"]),
                "grid_row": p["grid_row"], "grid_col": p["grid_col"],
            })
            saved += 1
        print(f"  {wb_name}: {saved}/{n_patches*n_patches} patches in {time.time()-t0:.1f}s")

    # Save metadata
    meta_path = OUTPUT_DIR / "patch_meta.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, ensure_ascii=False)

    size_mb = sum(f.stat().st_size for f in PATCHES_DIR.glob("*.npz")) / 1024 / 1024
    print(f"\nDone: {len(all_meta)} patches, {size_mb:.1f}MB")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
