"""Shared utilities for terrain-rgb tile generation."""

import math
import os
import sys
import time

import numpy as np
from PIL import Image

TILE_SIZE = 257  # pixel-as-point: 257 vertices, 256 quads
MERCATOR_EXTENT = 20037508.342789244


# ---- terrain-rgb encoding ----

def decode(rgb):
    R = rgb[:, :, 0].astype(np.float64)
    G = rgb[:, :, 1].astype(np.float64)
    B = rgb[:, :, 2].astype(np.float64)
    return -10000.0 + ((R * 65536.0 + G * 256.0 + B) * 0.1)


def encode(elev):
    encoded = np.clip((elev + 10000.0) / 0.1, 0, 16777215).astype(np.uint32)
    R = (encoded // 65536).astype(np.uint8)
    G = ((encoded % 65536) // 256).astype(np.uint8)
    B = (encoded % 256).astype(np.uint8)
    return np.dstack([R, G, B])


# ---- tile file I/O ----

def save_tile(z, x, y, rgb, out_dir):
    out_path = os.path.join(out_dir, str(z), str(x), f"{y}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(rgb).save(out_path)
    return out_path


def load_tile(z, x, y, src_dir):
    path = os.path.join(src_dir, str(z), str(x), f"{y}.png")
    try:
        return np.array(Image.open(path))
    except FileNotFoundError:
        return None


def tile_exists(z, x, y, out_dir):
    return os.path.exists(os.path.join(out_dir, str(z), str(x), f"{y}.png"))


# ---- Mercator ↔ lat/lon ----

def mercator_xmin(x, z):
    return -MERCATOR_EXTENT + x * (2 * MERCATOR_EXTENT / (2 ** z))


def mercator_ymax(y, z):
    return MERCATOR_EXTENT - y * (2 * MERCATOR_EXTENT / (2 ** z))


def tile_latlon_bounds(z, x, y):
    """Return (north, south, west, east) in degrees for a Mercator tile."""
    n = 2 ** z
    u_min, u_max = x / n, (x + 1) / n
    v_min, v_max = y / n, (y + 1) / n
    west = u_min * 360.0 - 180.0
    east = u_max * 360.0 - 180.0
    merc_y_top = math.pi - v_min * 2.0 * math.pi
    merc_y_bot = math.pi - v_max * 2.0 * math.pi
    north = math.degrees(2.0 * math.atan(math.exp(merc_y_top)) - math.pi / 2.0)
    south = math.degrees(2.0 * math.atan(math.exp(merc_y_bot)) - math.pi / 2.0)
    return north, south, west, east


# ---- lat/lon → tile x/y ----

def latlon_to_xy_range(lat_min, lat_max, lon_min, lon_max, z):
    """Convert lat/lon bounding box to Mercator x/y tile ranges.

    Args:
        lat_min, lat_max: latitude bounds in degrees.
        lon_min, lon_max: longitude bounds in degrees.
        z: zoom level.

    Returns:
        (x_min, x_max, y_min, y_max) — inclusive floor ranges.
    """
    n = 2 ** z
    x_min = int(math.floor((lon_min + 180.0) / 360.0 * n))
    x_max = int(math.floor((lon_max + 180.0) / 360.0 * n))

    # Mercator y: y=0 at north, y=n-1 at south
    merc_north = math.log(math.tan(math.pi / 4.0 + math.radians(lat_max) / 2.0))
    merc_south = math.log(math.tan(math.pi / 4.0 + math.radians(lat_min) / 2.0))
    v_north = (math.pi - merc_north) / (2.0 * math.pi)
    v_south = (math.pi - merc_south) / (2.0 * math.pi)
    y_min = int(math.floor(v_north * n))
    y_max = int(math.floor(v_south * n))

    return x_min, x_max, y_min, y_max


# ---- half-pixel extension ----

def half_pixel_extension(x, z, west, east, north, south):
    """Compute half-pixel extension for pixel-as-point alignment.
    At antimeridian edges, don't extend outward (source data doesn't wrap).
    """
    pixel_size_lon = (east - west) / (TILE_SIZE - 1)
    pixel_size_lat = (north - south) / (TILE_SIZE - 1)
    max_tile = (2 ** z) - 1
    ext_left = 0.0 if x == 0 else 0.5 * pixel_size_lon
    ext_right = 0.0 if x == max_tile else 0.5 * pixel_size_lon
    return ext_left, ext_right, pixel_size_lat


# ---- progress bar ----

def progress_bar(count, total, t0):
    elapsed = time.time() - t0
    rate = count / elapsed if elapsed > 0 else 0
    eta = (total - count) / rate if rate > 0 else 0

    pct = count / total * 100
    bar_width = 30
    filled = int(bar_width * count / total)
    bar = "█" * filled + "░" * (bar_width - filled)

    eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta / 60:.1f}m"
    rate_str = f"{rate:.1f}/s" if rate >= 0.1 else f"{rate:.2f}/s"
    sys.stderr.write(
        f"\r  [{bar}] {pct:5.1f}%  {count}/{total}  "
        f"{rate_str}  ETA {eta_str}  "
    )
    sys.stderr.flush()


# ---- CLI helpers ----

def parse_range(s):
    if ".." in s:
        parts = s.split("..")
        return range(int(parts[0]), int(parts[1]) + 1)
    return [int(s)]


# ---- resampling per zoom ----
# Per docs/ELEVATION.md: >8:1 average, 2-8:1 cubic, <2:1 lanczos

def resample_method(z):
    if z <= 9:
        return "average"
    elif z <= 11:
        return "cubic"
    else:
        return "lanczos"
