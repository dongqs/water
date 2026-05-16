#!/usr/bin/env python3
"""Fetch OSM water body boundaries as GeoJSON for Canvas visualization.

Output: tmp/lake_boundary_dataset/boundaries.geojson
"""

import json, math, sys, time, urllib.request, urllib.parse, urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "tmp" / "lake_boundary_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

# (display_name, type, center_lat, center_lon, [osm_names...], bbox_deg)
WATER_BODIES = [
    ("西湖", "lake", 30.25, 120.14, ["西湖"], 0.3),
    ("太湖", "lake", 31.20, 120.20, ["太湖", "Lake Tai"], 1.0),
    ("青海湖", "lake", 36.90, 100.20, ["青海湖", "Qinghai Lake", "Koko Nor"], 1.5),
    ("洞庭湖", "lake", 29.30, 112.90, ["洞庭湖", "Dongting Lake"], 1.0),
    ("钱塘江", "river", 30.20, 120.30, ["钱塘江"], 0.3),
    ("黄浦江", "river", 31.23, 121.49, ["黄浦江"], 0.2),
    ("长江-南京段", "river", 32.05, 118.75, ["长江"], 0.5),
    ("长江-武汉段", "river", 30.58, 114.30, ["长江"], 0.5),
]


def overpass_query(query, timeout=30):
    for attempt in range(3):
        try:
            data = urllib.parse.urlencode({"data": query}).encode("utf-8")
            req = urllib.request.Request(OVERPASS_URL, data=data,
                headers={"User-Agent": "niao/0.2", "Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"    attempt {attempt + 1}/3: {e}")
            if attempt < 2:
                time.sleep(3)
    return None


def build_polygons(elements):
    """Reconstruct polygon rings from OSM elements (out geom mode)."""
    # Index standalone ways
    way_index = {}
    for el in elements:
        if el["type"] == "way":
            geom = el.get("geometry")
            if geom and len(geom) >= 3:
                way_index[el["id"]] = [[p["lon"], p["lat"]] for p in geom]

    def collect_rings(rel, visited=None):
        if visited is None:
            visited = set()
        if rel["id"] in visited:
            return [], []
        visited.add(rel["id"])
        outers, inners = [], []
        for m in rel.get("members", []):
            mtype = m.get("type")
            mref = m.get("ref")
            mrole = m.get("role", "outer")
            if mtype == "way":
                geom = m.get("geometry")
                if geom and len(geom) >= 3:
                    ring = [[p["lon"], p["lat"]] for p in geom]
                elif mref in way_index:
                    ring = way_index[mref]
                else:
                    continue
                if mrole == "inner":
                    inners.append(ring)
                else:
                    outers.append(ring)
            elif mtype == "relation":
                for sub in elements:
                    if sub["type"] == "relation" and sub["id"] == mref:
                        sub_o, sub_i = collect_rings(sub, visited)
                        outers.extend(sub_o)
                        inners.extend(sub_i)
                        break
        return outers, inners

    polygons = []
    for el in elements:
        if el["type"] != "relation":
            continue
        tags = el.get("tags", {})
        if "natural" not in tags and "waterway" not in tags:
            continue
        if "type" not in tags:
            continue
        name = tags.get("name", "unnamed")
        outers, inners = collect_rings(el)
        if outers:
            polygons.append({
                "name": name, "osm_id": el["id"],
                "type": tags.get("waterway", tags.get("natural", "")),
                "outer_rings": outers, "inner_rings": inners,
            })
    return polygons


def fetch_water_body(names, wb_type, c_lat, c_lon, bbox_deg):
    """Try OSM names until one returns data."""
    bbox = f"{c_lat - bbox_deg},{c_lon - bbox_deg},{c_lat + bbox_deg},{c_lon + bbox_deg}"
    tag_filter = '["natural"="water"]' if wb_type == "lake" else '["waterway"~"river|riverbank"]'

    for osm_name in names:
        query = f"""
        [out:json][timeout:25][bbox:{bbox}];
        relation["name"="{osm_name}"]{tag_filter};
        (._;>;);
        out geom;
        """.replace("\n", " ")
        data = overpass_query(query)
        if data and data.get("elements"):
            polys = build_polygons(data.get("elements", []))
            if polys:
                print(f"    matched OSM name: {osm_name}")
                return polys
        time.sleep(1)
    return []


def main():
    all_features = []

    for wb_name, wb_type, c_lat, c_lon, osm_names, bbox_deg in WATER_BODIES:
        print(f"\n{wb_name} ({wb_type}) @ {c_lat:.2f}, {c_lon:.2f}")
        polys = fetch_water_body(osm_names, wb_type, c_lat, c_lon, bbox_deg)
        if not polys:
            print("  FAILED: no OSM data found")
            continue

        for poly in polys:
            for ring in poly["outer_rings"]:
                all_features.append({
                    "type": "Feature",
                    "properties": {"name": poly["name"], "type": poly["type"],
                                   "water_body": wb_name, "wb_type": wb_type,
                                   "role": "outer", "osm_id": poly["osm_id"]},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                })
            for ring in poly["inner_rings"]:
                all_features.append({
                    "type": "Feature",
                    "properties": {"name": poly["name"], "type": poly["type"],
                                   "water_body": wb_name, "wb_type": wb_type,
                                   "role": "inner", "osm_id": poly["osm_id"]},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                })

            total_pts = sum(len(r) for r in poly["outer_rings"])
            lats = [p[1] for r in poly["outer_rings"] for p in r]
            lons = [p[0] for r in poly["outer_rings"] for p in r]
            print(f"  {poly['name']}: {len(poly['outer_rings'])}o+{len(poly['inner_rings'])}i, {total_pts}pts")
            if lats:
                print(f"    lat[{min(lats):.4f}..{max(lats):.4f}] lon[{min(lons):.4f}..{max(lons):.4f}]")

        time.sleep(1)

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = OUTPUT_DIR / "boundaries.geojson"
    with open(output_path, "w") as f:
        json.dump(geojson, f, ensure_ascii=False)
    print(f"\nDone: {len(all_features)} features, {output_path.stat().st_size / 1024:.0f}KB")


if __name__ == "__main__":
    main()
