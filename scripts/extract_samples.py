#!/usr/bin/env python3
"""Extract 8×8 JAXA elevation patches for West Lake boundary sample points.

Reads OSM boundary from boundaries.geojson, samples points at 1km intervals
along the shoreline, extracts 8×8 elevation from JAXA GeoTIFFs.

Output: tmp/lake_boundary_dataset/samples/
  sample_NN.bin  — 8×8 float32 elevation (256 bytes)
  sample_meta.json — sample point coordinates + file mapping
"""

import gzip, json, math, sys, time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import rowcol
from matplotlib.path import Path as MplPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "tmp" / "lake_boundary_dataset" / "samples"
JAXA_STORAGE = Path("/home/dev/jaxa_storage")
CACHE_DIR = PROJECT_ROOT / "tmp" / "jaxa_cache"

PATCH_PX = 16         # 16×16 JAXA pixels
ARC_SEC = 1 / 3600    # 1 arc-second ≈ 30m
SAMPLE_SPACING_M = 1000
LAKES = ["西湖", "太湖", "洞庭湖"]

# ---- Tile index ----
_TILE_INDEX = None

def build_tile_index():
    global _TILE_INDEX
    _TILE_INDEX = {}
    for entry in JAXA_STORAGE.iterdir():
        if not entry.is_dir(): continue
        for f in entry.iterdir():
            name = f.name
            if name.endswith(".tif.gz"): tile_name = name[:-7]
            elif name.endswith(".tif"): tile_name = name[:-4]
            else: continue
            if tile_name not in _TILE_INDEX or str(f).endswith(".tif"):
                _TILE_INDEX[tile_name] = f

def get_tif_path(lat, lon):
    global _TILE_INDEX
    if _TILE_INDEX is None: build_tile_index()
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
                with open(cache, "wb") as out: out.write(gz.read())
        s = str(cache)
    return rasterio.open(s)

# ---- Boundary loading ----
def load_boundary_ring(geojson_path, water_body="西湖"):
    with open(geojson_path) as f:
        data = json.load(f)
    outer_segs = []
    for feat in data["features"]:
        if feat["properties"].get("water_body") != water_body: continue
        if feat["properties"].get("role") == "inner": continue
        ring = feat["geometry"]["coordinates"][0]
        if len(ring) >= 2:
            outer_segs.append([[p[0], p[1]] for p in ring])
    return outer_segs

def assemble_rings(segments):
    """Same algorithm as JS assembleRings (closed only)."""
    if not segments: return []
    used = set()
    result = []
    RING_EPS = 1e-5; CLOSE_EPS = 1e-4
    def pteq(a, b): return abs(a[0]-b[0]) < RING_EPS and abs(a[1]-b[1]) < RING_EPS
    def near(a, b): return abs(a[0]-b[0]) < CLOSE_EPS and abs(a[1]-b[1]) < CLOSE_EPS
    def pick_next():
        for i, s in enumerate(segments):
            if i not in used: used.add(i); return s[:]
        return None
    def find_connected(pt):
        for eq in [pteq, near]:
            for i, s in enumerate(segments):
                if i in used: continue
                if len(s) < 2: continue
                if eq(s[-1], pt): return i, s[:-1][::-1]
                if eq(s[0], pt): return i, s[1:]
        return None
    chain = pick_next()
    while chain:
        changed = True
        while changed:
            changed = False
            end = chain[-1]
            nxt = find_connected(end)
            if nxt: used.add(nxt[0]); chain.extend(nxt[1]); changed = True; continue
            start = chain[0]
            nxt = find_connected(start)
            if nxt: used.add(nxt[0]); chain[0:0] = nxt[1]; changed = True
        s, e = chain[0], chain[-1]
        if len(chain) >= 3 and not pteq(s, e) and near(s, e): chain.append([s[0], s[1]])
        if len(chain) >= 3 and (pteq(s, e) or near(s, e)): result.append(chain)
        chain = pick_next()
    return result

def sample_along_ring(ring, spacing_m):
    if len(ring) < 2: return []
    dists = [0]
    mid_lat = sum(p[1] for p in ring) / len(ring)
    m_per_deg_lon = 111320 * math.cos(mid_lat * math.pi / 180)
    for i in range(1, len(ring)):
        dlat = (ring[i][1] - ring[i-1][1]) * 111320
        dlon = (ring[i][0] - ring[i-1][0]) * m_per_deg_lon
        dists.append(dists[-1] + math.sqrt(dlat*dlat + dlon*dlon))
    total = dists[-1]
    if total < spacing_m:
        mid = ring[len(ring)//2]
        return [(mid[1], mid[0])]
    result = []
    for d in np.arange(0, total, spacing_m):
        seg = 1
        while seg < len(dists) and dists[seg] < d: seg += 1
        if seg >= len(dists): seg = len(dists) - 1
        t = (d - dists[seg-1]) / max(dists[seg] - dists[seg-1], 0.001)
        lat = ring[seg-1][1] + t * (ring[seg][1] - ring[seg-1][1])
        lon = ring[seg-1][0] + t * (ring[seg][0] - ring[seg-1][0])
        result.append((lat, lon))
    return result

def extract_patch(lat_ctr, lon_ctr, tif_cache=None):
    """Extract PATCH_PX × PATCH_PX elevation around (lat_ctr, lon_ctr)."""
    half = PATCH_PX / 2 * ARC_SEC
    cache = tif_cache or {}
    elev = np.full((PATCH_PX, PATCH_PX), np.nan, dtype=np.float32)

    # Group pixels by tile for batch reading
    pixels_by_tile = {}
    for row in range(PATCH_PX):
        for col in range(PATCH_PX):
            lat = lat_ctr + half - row * ARC_SEC
            lon = lon_ctr - half + col * ARC_SEC
            tif_path = get_tif_path(lat, lon)
            if tif_path is None: continue
            key = str(tif_path)
            if key not in pixels_by_tile:
                pixels_by_tile[key] = []
            pixels_by_tile[key].append((row, col, lat, lon))

    for key, px_list in pixels_by_tile.items():
        if key not in cache:
            try:
                cache[key] = open_tif(Path(key))
            except Exception:
                continue
        ds = cache[key]
        for row, col, lat, lon in px_list:
            try:
                r, c = rowcol(ds.transform, lon, lat)
                r, c = int(r), int(c)
                if 0 <= r < ds.height and 0 <= c < ds.width:
                    elev[row, col] = float(ds.read(1, window=((r, r+1), (c, c+1)))[0, 0])
            except Exception: continue
    return elev

def main():
    build_tile_index()
    print(f"Tile index: {len(_TILE_INDEX)} tiles")

    geojson_path = OUTPUT_DIR.parent / "boundaries.geojson"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_meta = []
    global_idx = 0

    for lake_name in LAKES:
        segments = load_boundary_ring(str(geojson_path), lake_name)
        if not segments:
            print(f"{lake_name}: no boundary data, skip")
            continue
        rings = assemble_rings(segments)
        print(f"{lake_name}: {len(segments)} segs → {len(rings)} assembled rings")

        samples = []
        for ring in rings:
            pts = sample_along_ring(ring, SAMPLE_SPACING_M)
            samples.extend(pts)
        print(f"  {len(samples)} sample points @ {SAMPLE_SPACING_M}m")

        for lat, lon in samples:
            elev = extract_patch(lat, lon)
            # ASCII-safe prefix: West Lake=wl, Taihu=th, Dongting=dt
            prefix = {"西湖":"xihu", "太湖":"taihu", "洞庭湖":"dongtinghu"}.get(lake_name, "unknown")
            fname = f"{prefix}{global_idx:03d}.bin"
            with open(OUTPUT_DIR / fname, "wb") as f:
                f.write(elev.astype(np.float32).tobytes())
            all_meta.append({"index": global_idx, "lake": lake_name, "lat": lat, "lon": lon, "file": fname})
            if global_idx % 20 == 0:
                print(f"  [{global_idx}] {lake_name}: ({lat:.5f}, {lon:.5f})")
            global_idx += 1

        # ---- Extension: every 5th shoreline point, sample into water + land ----
        print(f"  Generating extension samples...")
        ext_count = 0
        # Pre-build paths for each ring (once)
        ring_paths = []
        for ring in rings:
            verts = np.array([(p[0], p[1]) for p in ring])
            ring_paths.append(MplPath(verts))
        if not ring_paths:
            ring_paths = [MplPath(np.array([(0,0)]))]

        for ri, ring in enumerate(rings):
            shore_pts = sample_along_ring(ring, SAMPLE_SPACING_M)
            path = ring_paths[ri] if ri < len(ring_paths) else ring_paths[0]
            for si, (lat, lon) in enumerate(shore_pts):
                if si % 5 != 0:
                    continue

                # Compute local normal: tangent from neighbors along assembled ring
                prev_idx = (si - 1) % len(shore_pts)
                next_idx = (si + 1) % len(shore_pts)
                plat, plon = shore_pts[prev_idx]
                nlat, nlon = shore_pts[next_idx]
                cos_mid = math.cos(math.radians(lat))
                tx = (nlon - plon) * 111320 * cos_mid
                ty = (nlat - plat) * 111320
                tlen = math.sqrt(tx*tx + ty*ty) or 1
                nx = ty / tlen
                ny = -tx / tlen

                # Determine water side: test small step in each direction
                water_sign = 1
                for test_sign in [-1, 1]:
                    tlat = lat + test_sign * ny * 10 / 111320
                    tlon = lon + test_sign * nx * 10 / (111320 * cos_mid)
                    if path.contains_point((tlon, tlat)):
                        water_sign = test_sign
                        break

                # Generate 1 water + 1 land extension
                def rand_dist():
                    r = np.random.random()
                    if r < 0.5: return np.random.uniform(150, 600)   # near shore
                    elif r < 0.8: return np.random.uniform(600, 1800)  # medium
                    else: return np.random.uniform(1800, 3000)        # far

                for direction, label_suffix in [(water_sign, "water"), (-water_sign, "land")]:
                    dist = rand_dist()
                    elat = lat + direction * ny * dist / 111320
                    elon = lon + direction * nx * dist / (111320 * cos_mid)

                    elev = extract_patch(elat, elon)
                    prefix = {"西湖":"xihu", "太湖":"taihu", "洞庭湖":"dongtinghu"}.get(lake_name, "xx")
                    fname = f"{prefix}{global_idx:03d}.bin"
                    with open(OUTPUT_DIR / fname, "wb") as f:
                        f.write(elev.astype(np.float32).tobytes())
                    all_meta.append({
                        "index": global_idx, "lake": lake_name,
                        "lat": elat, "lon": elon, "file": fname,
                        "sample_type": label_suffix,
                        "parent_idx": global_idx - ext_count - 1 + si,  # approximate
                    })
                    if global_idx % 50 == 0:
                        print(f"  [{global_idx}] {lake_name}/{label_suffix}: ({elat:.5f}, {elon:.5f}) d={dist:.0f}m")
                    global_idx += 1
                    ext_count += 1
        print(f"  {ext_count} extension samples")

    # ---- Evaluation: GT mask + lakeMask accuracy ----
    print("Computing GT masks and accuracy...")

    flatness_values = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    EVAL_PX = 8  # inner evaluation area
    SUB = 4      # sub-pixel resolution

    for s in all_meta:
        lat, lon = s["lat"], s["lon"]
        elev = np.fromfile(OUTPUT_DIR / s["file"], dtype=np.float32).reshape(PATCH_PX, PATCH_PX)
        elev = np.nan_to_num(elev, nan=np.nanmedian(elev[np.isfinite(elev)]) if np.any(np.isfinite(elev)) else 0)

        # --- GT mask: 4x4 sub-pixel ray casting against OSM boundary ---
        # The inner 8x8 area is centered in the 16x16 patch
        half_deg = EVAL_PX / 2 * ARC_SEC  # 4 arc-seconds
        outer_half = PATCH_PX / 2 * ARC_SEC  # 8 arc-seconds

        # Build MplPath from assembled outer rings (already computed above for this lake)
        # Find which lake this sample belongs to
        lake_name = s["lake"]
        lake_segments = load_boundary_ring(str(geojson_path), lake_name)
        lake_rings = assemble_rings(lake_segments)
        if not lake_rings:
            s["gt_mask"] = None
            s["accuracy"] = {}
            continue

        # Use the largest outer ring
        outer_ring = max(lake_rings, key=len)
        # Convert to MplPath vertices: (lon, lat)
        verts = np.array([(p[0], p[1]) for p in outer_ring])
        path = MplPath(verts)

        # Generate sub-pixel centers for 8x8 grid
        gt_mask = np.zeros((EVAL_PX, EVAL_PX), dtype=np.uint8)
        for er in range(EVAL_PX):
            for ec in range(EVAL_PX):
                # Center of this eval pixel in degrees
                elat = lat + half_deg - (er + 0.5) * ARC_SEC
                elon = lon - half_deg + (ec + 0.5) * ARC_SEC
                # 4x4 sub-pixels
                sub_water = 0
                for sr in range(SUB):
                    for sc in range(SUB):
                        slat = elat + (sr / SUB - 0.5 + 0.5/SUB) * ARC_SEC
                        slon = elon + (sc / SUB - 0.5 + 0.5/SUB) * ARC_SEC
                        if path.contains_point((slon, slat)):
                            sub_water += 1
                gt_mask[er, ec] = 1 if sub_water > SUB*SUB/2 else 0

        s["gt_mask"] = gt_mask.flatten().tolist()

        # --- LakeMask accuracy for each flatness ---
        # Use the full 16x16 for gradient computation, evaluate only inner 8x8
        # Inner 8x8 corresponds to rows 4..11, cols 4..11 in the 16x16 array
        r0 = (PATCH_PX - EVAL_PX) // 2
        c0 = (PATCH_PX - EVAL_PX) // 2
        accuracy = {}
        for flatness in flatness_values:
            lake_mask = np.zeros((PATCH_PX, PATCH_PX), dtype=np.float64)
            for r in range(1, PATCH_PX - 1):
                for c in range(1, PATCH_PX - 1):
                    du = elev[r, c+1] - elev[r, c-1]
                    dv = elev[r+1, c] - elev[r-1, c]
                    steep = np.sqrt(du*du + dv*dv) / 2
                    lo, hi = flatness * 0.3, flatness
                    if steep <= lo:
                        lake_mask[r, c] = 1.0
                    elif steep >= hi:
                        lake_mask[r, c] = 0.0
                    else:
                        t = (steep - lo) / (hi - lo)
                        lake_mask[r, c] = 1.0 - t*t*(3.0 - 2.0*t)
            # Evaluate inner 8x8
            pred = lake_mask[r0:r0+EVAL_PX, c0:c0+EVAL_PX] > 0.5
            correct = np.sum(pred == gt_mask)
            accuracy[str(flatness)] = round(float(correct) / (EVAL_PX * EVAL_PX), 4)

        s["accuracy"] = accuracy

        # Quality filter (same as before)
        rounded = np.round(elev[np.isfinite(elev)]).astype(np.int32)
        if len(rounded) >= 64:
            values, counts = np.unique(rounded, return_counts=True)
            total = len(rounded)
            idx = np.argsort(-counts)
            top1 = counts[idx[0]] / total
            top2 = counts[idx[1]] / total if len(idx) > 1 else 0
            if top1 >= 0.25 and top2 < 0.25:
                s["quality"] = "good"
                s["water_elev"] = float(values[idx[0]])
                s["water_frac"] = float(top1)
            else:
                s["quality"] = "noisy"
        else:
            s["quality"] = "noisy"

    qualified_count = sum(1 for s in all_meta if s.get("quality") == "good")
    print(f"Quality: {qualified_count}/{len(all_meta)} good patches")
    print(f"Flatness sweep: {flatness_values}")
    # Show best flatness
    best_f = max(flatness_values, key=lambda f: np.mean([s["accuracy"][str(f)] for s in all_meta if s["accuracy"].get(str(f))]))
    best_acc = np.mean([s["accuracy"][str(best_f)] for s in all_meta if s["accuracy"].get(str(best_f))])
    print(f"Best flatness: {best_f} (avg accuracy: {best_acc:.4f})")

    # ---- Export terrain-rgb PNGs + compact meta ----
    from PIL import Image
    png_dir = OUTPUT_DIR / "png"
    png_dir.mkdir(exist_ok=True)
    compact_meta = []
    for s in all_meta:
        elev = np.fromfile(OUTPUT_DIR / s["file"], dtype=np.float32).reshape(PATCH_PX, PATCH_PX)
        elev = np.nan_to_num(elev, nan=0)
        # Encode terrain-rgb: elev+10000, /0.1, split into RGB
        encoded = np.clip((elev + 10000.0) / 0.1, 0, 16777215).astype(np.uint32)
        R = (encoded // 65536).astype(np.uint8)
        G = ((encoded % 65536) // 256).astype(np.uint8)
        B = (encoded % 256).astype(np.uint8)
        rgb = np.dstack([R, G, B])
        png_name = s["file"].replace(".bin", ".png")
        Image.fromarray(rgb).save(png_dir / png_name, optimize=True)
        # Compact meta (drop verbose fields)
        cm = {"i": s["index"], "k": s["lake"], "t": s.get("sample_type", "shore"),
              "lat": round(s["lat"], 5), "lon": round(s["lon"], 5),
              "png": png_name}
        if s.get("quality") == "good":
            cm["q"] = 1; cm["we"] = s.get("water_elev"); cm["wf"] = s.get("water_frac")
        else:
            cm["q"] = 0
        if s.get("gt_mask"):
            cm["gm"] = s["gt_mask"]
        if s.get("accuracy"):
            cm["ac"] = {k: v for k, v in s["accuracy"].items()}
        compact_meta.append(cm)

    with open(OUTPUT_DIR / "sample_meta.json", "w") as f:
        json.dump(compact_meta, f, ensure_ascii=False)
    # Also keep full meta for Python use
    with open(OUTPUT_DIR / "sample_meta_full.json", "w") as f:
        json.dump(all_meta, f, ensure_ascii=False)

    png_size = sum(f.stat().st_size for f in png_dir.glob("*.png")) / 1024 / 1024
    meta_size = (OUTPUT_DIR / "sample_meta.json").stat().st_size / 1024
    print(f"Terrain-RGB PNGs: {png_size:.1f}MB, compact meta: {meta_size:.0f}KB")
    print(f"Done: {len(all_meta)} total samples in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
