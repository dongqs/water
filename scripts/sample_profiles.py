#!/usr/bin/env python3
"""Extract 1D elevation profiles from JAXA AW3D30 DEM data and OSM water boundaries.

Reads raw GeoTIFFs from /home/dev/jaxa_storage/ (decompressing .tif.gz as needed).
Queries OSM Overpass API for water body boundaries.
Stores everything in tmp/lake_benchmark/lake_boundary.db SQLite.

Usage:
  python scripts/sample_profiles.py              # extract all profiles
  python scripts/sample_profiles.py --dry-run    # show what would be extracted
"""

import gzip
import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "tmp" / "lake_benchmark" / "lake_boundary.db"
JAXA_STORAGE = Path("/home/dev/jaxa_storage")
CACHE_DIR = PROJECT_ROOT / "tmp" / "jaxa_cache"  # decompressed .tif cache

# OSM Overpass API endpoint (public, rate-limited)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# --- Water body definitions ---
# Each: (name, type, center_lat, center_lon, profile_count, profile_length_deg)
# profile_length_deg: how many degrees wide each u-axis profile is (centered on body)
WATER_BODIES = [
    # Lakes
    ("西湖", "lake", 30.25, 120.14, 3, 0.08),
    ("太湖", "lake", 31.20, 120.20, 3, 0.30),
    ("青海湖", "lake", 36.90, 100.20, 3, 0.60),
    ("洞庭湖", "lake", 29.30, 112.90, 3, 0.35),
    # Rivers
    ("钱塘江", "river", 30.20, 120.30, 2, 0.06),
    ("黄浦江", "river", 31.23, 121.49, 2, 0.05),
    ("长江-南京段", "river", 32.05, 118.75, 2, 0.10),
    ("长江-武汉段", "river", 30.58, 114.30, 2, 0.12),
]


_TILE_INDEX = None  # lazy-built index: tile_name -> Path


def _build_tile_index():
    """Scan JAXA_STORAGE and build {tile_name: file_path} index."""
    global _TILE_INDEX
    _TILE_INDEX = {}
    for entry in sorted(JAXA_STORAGE.iterdir()):
        if not entry.is_dir():
            continue
        for f in entry.iterdir():
            if f.suffix in (".tif", ".gz"):
                name = f.name
                # Strip .gz extension to get tile name
                if name.endswith(".tif.gz"):
                    tile_name = name[:-7]
                elif name.endswith(".tif"):
                    tile_name = name[:-4]
                else:
                    continue
                # Prefer uncompressed .tif over .tif.gz
                if tile_name not in _TILE_INDEX or not str(f).endswith(".gz"):
                    _TILE_INDEX[tile_name] = f


def get_tif_path(lat, lon):
    """Find the JAXA 1°×1° tile containing (lat, lon).

    JAXA AW3D30 tiles are named N{lat}E{lon} for the NW corner of the 1°×1° cell.
    """
    global _TILE_INDEX
    if _TILE_INDEX is None:
        _build_tile_index()

    n_part = f"N{int(np.floor(lat)):03d}"
    if lon >= 0:
        e_part = f"E{int(np.floor(lon)):03d}"
    else:
        e_part = f"W{int(-np.floor(lon)):03d}"
    tile_name = f"{n_part}{e_part}"

    path = _TILE_INDEX.get(tile_name)
    return path, tile_name


def get_elevation(tif_path, lats, lons):
    """Sample elevations at given lat/lon points from a JAXA GeoTIFF.

    Args:
        tif_path: Path to .tif or .tif.gz file
        lats, lons: Arrays of latitude/longitude coordinates (same length)

    Returns:
        numpy array of elevations in meters, NaN for out-of-bounds

    Handles .tif.gz by decompressing to cache first.
    """
    path_str = str(tif_path)

    if path_str.endswith(".gz"):
        # Decompress to cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_name = Path(path_str).stem  # removes .gz, keeps .tif
        cache_path = CACHE_DIR / cache_name
        if not cache_path.exists():
            with gzip.open(path_str, "rb") as gz_f:
                with open(cache_path, "wb") as out_f:
                    out_f.write(gz_f.read())
        path_str = str(cache_path)

    with rasterio.open(path_str) as ds:
        # rasterio.transform: (lon, lat) -> (col, row)
        rows, cols = rasterio.transform.rowcol(ds.transform, lons, lats)
        rows = np.array(rows)
        cols = np.array(cols)

        # Mask out-of-bounds indices
        valid = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)

        elevations = np.full(len(lats), np.nan, dtype=np.float64)
        if valid.any():
            data = ds.read(1)  # band 1
            elevations[valid] = data[rows[valid], cols[valid]]

        return elevations


def get_nearby_profiles(center_lat, center_lon, count, length_deg, direction="u"):
    """Generate profile lines near a water body center.

    For u-axis profiles (east-west): vary latitude to cross different parts.
    For rivers, profiles span less to stay near the river.

    Args:
        center_lat, center_lon: Center point of the water body
        count: Number of profiles to generate
        length_deg: Total length of each profile in degrees
        direction: 'u' for east-west profiles

    Returns:
        list of (start_lat, start_lon, end_lat, end_lon, label)
    """
    profiles = []
    half_len = length_deg / 2.0
    name = "u" if direction == "u" else direction

    if count == 1:
        profiles.append((
            center_lat,
            center_lon - half_len,
            center_lat,
            center_lon + half_len,
            f"{name}_01",
        ))
    else:
        # Space profiles across ±30% of the length
        offsets = np.linspace(-half_len * 0.3, half_len * 0.3, count)
        for i, off in enumerate(offsets):
            lat = center_lat + off
            profiles.append((
                lat,
                center_lon - half_len,
                lat,
                center_lon + half_len,
                f"{name}_{i + 1:02d}",
            ))

    return profiles


def sample_profile(tif_cache, start_lat, start_lon, end_lat, end_lon, num_samples=512):
    """Extract elevation along a straight-line profile.

    Bilinear interpolation between GeoTIFF samples.

    Args:
        tif_cache: dict mapping tile_name -> rasterio dataset handle
        start_lat, start_lon: Profile start point
        end_lat, end_lon: Profile end point
        num_samples: Number of sample points along the profile

    Returns:
        (distances_m, lats, lons, elevations_raw) arrays
    """
    lats = np.linspace(start_lat, end_lat, num_samples)
    lons = np.linspace(start_lon, end_lon, num_samples)

    # Compute distances along profile (approximate, using spherical earth)
    # For short profiles (< 1°), simple approximation is fine
    dlat = lats - lats[0]
    dlon = lons - lons[0]
    mid_lat = np.radians(np.mean(lats))
    dx = dlon * 111320.0 * np.cos(mid_lat)  # meters per degree longitude
    dy = dlat * 111320.0  # meters per degree latitude
    distances = np.sqrt(dx**2 + dy**2)

    # Sample elevation point by point (could batch per tile but simpler this way)
    elevations = np.full(num_samples, np.nan, dtype=np.float64)

    for i in range(num_samples):
        tif_path, tile_name = get_tif_path(lats[i], lons[i])
        if tif_path is None:
            continue

        path_str = str(tif_path)
        if path_str.endswith(".gz"):
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_name = Path(path_str).stem
            cache_path = CACHE_DIR / cache_name
            if not cache_path.exists():
                with gzip.open(path_str, "rb") as gz_f:
                    with open(cache_path, "wb") as out_f:
                        out_f.write(gz_f.read())
            path_str = str(cache_path)

        try:
            with rasterio.open(path_str) as ds:
                row, col = rasterio.transform.rowcol(ds.transform, lons[i], lats[i])
                row, col = int(row), int(col)
                if 0 <= row < ds.height and 0 <= col < ds.width:
                    elevations[i] = float(ds.read(1)[row, col])
        except Exception:
            continue

    return distances, lats, lons, elevations


def query_osm_water(lat, lon, name_hint=None, radius_m=50000):
    """Query OSM Overpass API for water bodies near a point.

    Returns a list of water body geometries (polygons as lat/lon rings).
    """
    # Overpass QL: find water bodies within radius of point
    query = f"""
    [out:json][timeout:15];
    (
      way["natural"="water"](around:{radius_m},{lat},{lon});
      relation["natural"="water"](around:{radius_m},{lat},{lon});
      way["waterway"="river"](around:{radius_m},{lat},{lon});
      relation["waterway"="river"](around:{radius_m},{lat},{lon});
      way["waterway"="riverbank"](around:{radius_m},{lat},{lon});
    );
    (._;>;);
    out body;
    """
    try:
        req = urllib.request.Request(
            OVERPASS_URL,
            data=query.encode("utf-8"),
            headers={"User-Agent": "niao-lake-benchmark/0.1"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  WARNING: OSM query failed: {e}")
        return []

    return _parse_overpass_response(data, name_hint)


def _parse_overpass_response(data, name_hint=None):
    """Parse Overpass API JSON into list of (name, polygon_rings)."""
    elements = data.get("elements", [])

    # Separate nodes and ways
    nodes = {}
    ways = []
    relations = []

    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])
        elif el["type"] == "way":
            ways.append(el)
        elif el["type"] == "relation":
            relations.append(el)

    results = []

    # Process ways
    for way in ways:
        tags = way.get("tags", {})
        name = tags.get("name", "unnamed")
        if name_hint and name_hint not in name:
            continue

        # Build polygon ring from node refs
        coords = []
        for node_id in way.get("nodes", []):
            if node_id in nodes:
                coords.append(nodes[node_id])
        if len(coords) >= 3:
            results.append((name, [coords]))

    # Process relations (multipolygons)
    for rel in relations:
        tags = rel.get("tags", {})
        name = tags.get("name", "unnamed")
        if name_hint and name_hint not in name:
            if "water" not in tags and "waterway" not in tags:
                continue

        rings = []
        for member in rel.get("members", []):
            if member["type"] == "way" and member.get("ref"):
                # Find the referenced way
                for way in ways:
                    if way["id"] == member["ref"]:
                        coords = []
                        for node_id in way.get("nodes", []):
                            if node_id in nodes:
                                coords.append(nodes[node_id])
                        if len(coords) >= 3:
                            rings.append(coords)
                        break
        if rings:
            results.append((name, rings))

    return results


def find_shore_intersections(profile_lats, profile_lons, water_polygons):
    """Find where a profile crosses water body boundaries.

    For each water polygon, determine which sample indices are inside/outside.
    Returns list of (idx_start, idx_end, source, confidence) for each crossing.

    Uses ray casting (even-odd rule) for point-in-polygon test.
    """
    from matplotlib.path import Path as MplPath

    shore_segments = []

    for name, rings in water_polygons:
        # Use the largest ring (outer boundary)
        outer_ring = max(rings, key=len)
        vertices = np.array(outer_ring)

        # Build matplotlib path for fast point-in-polygon
        try:
            path = MplPath(vertices[:, ::-1])  # lat,lon -> lon,lat? No: mpl uses (x,y)
            # Actually mpl Path uses (x,y) where x=lon, y=lat
            path = MplPath(np.column_stack([vertices[:, 1], vertices[:, 0]]))
        except Exception:
            continue

        # Test all profile points
        points = np.column_stack([profile_lons, profile_lats])
        inside = path.contains_points(points)

        if not inside.any():
            continue

        # Find contiguous inside segments
        changes = np.diff(np.concatenate([[False], inside, [False]]).astype(int))
        starts = np.where(changes > 0)[0]
        ends = np.where(changes < 0)[0] - 1

        for s, e in zip(starts, ends):
            if e - s > 2:  # at least 3 consecutive points
                shore_segments.append((int(s), int(e), f"osm:{name}", 0.9))

    return shore_segments


def terrain_rgb_encode(elev):
    """Encode elevation to terrain-rgb values (0.1m quantization)."""
    encoded = np.clip((elev + 10000.0) / 0.1, 0, 16777215).astype(np.uint32)
    return encoded


def terrain_rgb_decode(encoded):
    """Decode terrain-rgb back to elevation."""
    return -10000.0 + encoded.astype(np.float64) * 0.1


def create_schema(conn):
    """Create SQLite schema if not exists."""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS water_bodies (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      type TEXT NOT NULL,
      osm_id INTEGER,
      osm_name TEXT,
      center_lat REAL NOT NULL,
      center_lon REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS profiles (
      id INTEGER PRIMARY KEY,
      water_body_id INTEGER REFERENCES water_bodies(id),
      label TEXT NOT NULL,
      start_lat REAL, start_lon REAL,
      end_lat REAL, end_lon REAL,
      direction TEXT NOT NULL,
      sample_count INTEGER,
      source TEXT DEFAULT 'jaxa'
    );

    CREATE TABLE IF NOT EXISTS elevation_samples (
      profile_id INTEGER REFERENCES profiles(id),
      idx INTEGER,
      dist_m REAL,
      lon REAL, lat REAL,
      elev_raw REAL,
      elev_terrain_rgb REAL,
      PRIMARY KEY (profile_id, idx)
    );

    CREATE TABLE IF NOT EXISTS shore_labels (
      profile_id INTEGER REFERENCES profiles(id),
      idx_start INTEGER,
      idx_end INTEGER,
      source TEXT,
      confidence REAL
    );

    CREATE TABLE IF NOT EXISTS algorithm_configs (
      id INTEGER PRIMARY KEY,
      label TEXT UNIQUE NOT NULL,
      grad_op TEXT,
      norm TEXT,
      window_stat TEXT,
      pre_smooth TEXT,
      multi_scale TEXT,
      decision_fn TEXT,
      elev_guard TEXT,
      extra_params TEXT
    );

    CREATE TABLE IF NOT EXISTS synthetic_params (
      profile_id INTEGER PRIMARY KEY REFERENCES profiles(id),
      water_level REAL,
      slope REAL,
      micro_relief_sigma REAL,
      source_profile_id INTEGER REFERENCES profiles(id)
    );

    CREATE TABLE IF NOT EXISTS benchmark_runs (
      id INTEGER PRIMARY KEY,
      algorithm_id INTEGER REFERENCES algorithm_configs(id),
      profile_id INTEGER REFERENCES profiles(id),
      noise_model TEXT,
      run_at TEXT
    );

    CREATE TABLE IF NOT EXISTS benchmark_metrics (
      run_id INTEGER PRIMARY KEY REFERENCES benchmark_runs(id),
      boundary_offset_m REAL,
      flood_ratio REAL,
      fragment_count INTEGER,
      lake_mask_rmse REAL,
      transition_width_m REAL
    );

    CREATE TABLE IF NOT EXISTS lake_mask_detail (
      run_id INTEGER REFERENCES benchmark_runs(id),
      idx INTEGER,
      dist_m REAL,
      elevation REAL,
      lake_mask REAL,
      PRIMARY KEY (run_id, idx)
    );

    CREATE TABLE IF NOT EXISTS dim_sweep_results (
      id INTEGER PRIMARY KEY,
      dim_name TEXT,
      variant_id TEXT,
      avg_offset_m REAL,
      avg_flood_ratio REAL,
      avg_fragment_count REAL,
      rank_offset INTEGER,
      rank_flood INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_elevation_profile
      ON elevation_samples(profile_id, idx);

    CREATE INDEX IF NOT EXISTS idx_lakemask_run
      ON lake_mask_detail(run_id, idx);

    CREATE INDEX IF NOT EXISTS idx_benchmark_profile
      ON benchmark_runs(profile_id);

    CREATE INDEX IF NOT EXISTS idx_benchmark_algo
      ON benchmark_runs(algorithm_id);
    """)


def main():
    dry_run = "--dry-run" in sys.argv

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    create_schema(conn)

    print(f"DB: {DB_PATH}")
    print(f"JAXA: {JAXA_STORAGE}")
    print(f"Water bodies: {len(WATER_BODIES)}")

    for wb_idx, (name, wb_type, center_lat, center_lon, n_profiles, length_deg) in enumerate(WATER_BODIES):
        print(f"\n{'='*60}")
        print(f"[{wb_idx + 1}/{len(WATER_BODIES)}] {name} ({wb_type})")

        # Check JAXA tile availability
        tif_path, tile_name = get_tif_path(center_lat, center_lon)
        if tif_path is None:
            print(f"  SKIP: no JAXA tile found for ({center_lat}, {center_lon})")
            continue
        print(f"  Tile: {tile_name} ({'OK' if tif_path else 'MISSING'})")

        if dry_run:
            continue

        # Query OSM for water boundary
        print(f"  Querying OSM...")
        radius = int(length_deg * 111320 * 2)  # rough conversion
        water_polygons = query_osm_water(center_lat, center_lon, name_hint=name[:2], radius_m=radius)
        print(f"  Found {len(water_polygons)} water polygon(s)")

        # Insert water body
        conn.execute(
            "INSERT INTO water_bodies (name, type, center_lat, center_lon) VALUES (?, ?, ?, ?)",
            (name, wb_type, center_lat, center_lon),
        )
        water_body_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Generate profiles
        profiles = get_nearby_profiles(center_lat, center_lon, n_profiles, length_deg)

        for pi, (s_lat, s_lon, e_lat, e_lon, label) in enumerate(profiles):
            print(f"  Profile {label}: ({s_lat:.4f}, {s_lon:.4f}) -> ({e_lat:.4f}, {e_lon:.4f})")

            # Sample elevation
            distances, lats, lons, elev_raw = sample_profile(
                {}, s_lat, s_lon, e_lat, e_lon, num_samples=512
            )

            valid = ~np.isnan(elev_raw)
            if not valid.any():
                print(f"    SKIP: all NaN elevations")
                continue

            # Compute terrain-rgb quantized elevation
            encoded = terrain_rgb_encode(np.nan_to_num(elev_raw, nan=0))
            elev_trgb = terrain_rgb_decode(encoded)
            elev_trgb[np.isnan(elev_raw)] = np.nan

            # Find shore intersections from OSM
            shore_segs = find_shore_intersections(lats, lons, water_polygons)
            print(f"    Shore segments: {len(shore_segs)}")

            # Insert profile
            conn.execute(
                "INSERT INTO profiles (water_body_id, label, start_lat, start_lon, end_lat, end_lon, direction, sample_count, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (water_body_id, label, float(s_lat), float(s_lon), float(e_lat), float(e_lon), "u", int(valid.sum()), "jaxa"),
            )
            profile_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Insert elevation samples
            samples = []
            for i in range(len(distances)):
                if not np.isnan(elev_raw[i]):
                    samples.append((
                        profile_id, i, float(distances[i]),
                        float(lons[i]), float(lats[i]),
                        float(elev_raw[i]), float(elev_trgb[i]),
                    ))
            conn.executemany(
                "INSERT INTO elevation_samples (profile_id, idx, dist_m, lon, lat, elev_raw, elev_terrain_rgb) VALUES (?, ?, ?, ?, ?, ?, ?)",
                samples,
            )

            # Insert shore labels
            for s_start, s_end, source, conf in shore_segs:
                conn.execute(
                    "INSERT INTO shore_labels (profile_id, idx_start, idx_end, source, confidence) VALUES (?, ?, ?, ?, ?)",
                    (profile_id, s_start, s_end, source, conf),
                )

            conn.commit()
            time.sleep(1)  # respect OSM rate limits

        conn.commit()

    # Summary
    n_bodies = conn.execute("SELECT COUNT(*) FROM water_bodies").fetchone()[0]
    n_profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    n_samples = conn.execute("SELECT COUNT(*) FROM elevation_samples").fetchone()[0]
    n_labels = conn.execute("SELECT COUNT(*) FROM shore_labels").fetchone()[0]
    print(f"\n{'='*60}")
    print(f"Done. {n_bodies} water bodies, {n_profiles} profiles, {n_samples} elevation samples, {n_labels} shore labels")
    print(f"DB: {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
