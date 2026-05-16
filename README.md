# Water Boundary Detection Research

Lake water-land boundary detection algorithm research using OSM polygons as ground truth and JAXA AW3D30 elevation data.

## Quick Start

```bash
npm install
npm run dev        # → http://localhost:3000/boundary.html
```

## Pages

| URL | Description |
|-----|------------|
| `boundary.html` | OSM polygon overlay + 16×16 JAXA sample patches with classification |
| `benchmark.html` | Algorithm benchmark results dashboard |

## Data

```
data/lake_boundary_dataset/
├── boundaries.geojson       # OSM water polygons (西湖/太湖/洞庭湖 + rivers)
└── samples/
    ├── sample_meta.json     # 760 sample points with GT masks + accuracy
    └── png/                 # 16×16 terrain-rgb PNG patches (171KB)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `fetch_osm_boundaries.py` | Fetch water body boundaries from OSM Overpass API |
| `extract_samples.py` | Sample elevation patches along lake shoreline + evaluation |
| `build_patch_dataset.py` | Build 64×64 elevation patch grid |
| `lake_algorithms.py` | Numpy GLSL-equivalent lake detection algorithms |
| `benchmark_lake.py` | Full benchmark runner with synthetic profiles |
| `plot_benchmark.py` | Matplotlib benchmark charts |
| `export_benchmark_json.py` | SQLite benchmark → JSON export |

## Key Findings

- **Best flatness**: 1.1–1.5 (68.3% accuracy plateau at 30m/px scale)
- **Water**: 90.4% accuracy (algorithm correctly identifies flat water)
- **Land**: 62.4% (flat farmland misclassified as water)
- **Shoreline**: 65.0% (transition zone is the hard part)
- **Single flatness ceiling**: ~68% due to conflicting water vs land optimal values

## Keyboard Shortcuts (boundary.html)

| Key | Action |
|-----|--------|
| `O` | Toggle OSM overlay |
| `S` | Toggle sample patches |
| `Q` | Filter good-quality only |
| `C` | Toggle classification overlay |
| `[`/`]` | Cycle flatness value |
| `F` | Fit all bodies |
| `R` | Re-center on selected |
| Scroll | Zoom |
| Drag | Pan |
