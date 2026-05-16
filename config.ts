// ---------------------------------------------------------------------------
// Load config.toml via smol-toml parser
// ---------------------------------------------------------------------------

import { parse as parseToml } from 'smol-toml';
import tomlSrc from '../config.toml?raw';

const cfg = parseToml(tomlSrc) as Record<string, Record<string, unknown>>;
const n = (s: string, k: string): number => cfg[s][k] as number;

// --- Zoom ---
export const ZOOM_MAX = n('zoom', 'max');

// --- LOD ---
export const LOD_THRESHOLD_DEFAULT = n('lod', 'threshold_default');
export const LOD_THRESHOLD_MIN = n('lod', 'threshold_min');
export const LOD_THRESHOLD_MAX = n('lod', 'threshold_max');
export const LOD_THRESHOLD_STEP = n('lod', 'threshold_step');
export const LOD_CULL_FREQUENCY = n('lod', 'cull_frequency');

// --- Displacement ---
export const DISPLACEMENT_SCALE_DEFAULT = n('displacement', 'scale_default');
export const DISPLACEMENT_SCALE_MIN = n('displacement', 'scale_min');
export const DISPLACEMENT_SCALE_MAX = n('displacement', 'scale_max');
export const DISPLACEMENT_SCALE_STEP = n('displacement', 'scale_step');
export const DISPLACEMENT_LOG_BIAS = n('displacement', 'log_bias');

// --- Hillshade ---
export const HILLSHADE_STRENGTH_DEFAULT = n('hillshade', 'strength_default');
export const HILLSHADE_STRENGTH_MIN = n('hillshade', 'strength_min');
export const HILLSHADE_STRENGTH_MAX = n('hillshade', 'strength_max');
export const HILLSHADE_STRENGTH_STEP = n('hillshade', 'strength_step');
export const HILLSHADE_LIGHT = {
  x: n('hillshade', 'light_x'),
  y: n('hillshade', 'light_y'),
  z: n('hillshade', 'light_z'),
};

// --- Snow line ---
export const SNOW_LINE_DEFAULT = n('snow', 'line_default');
export const SNOW_LINE_MIN = n('snow', 'line_min');
export const SNOW_LINE_MAX = n('snow', 'line_max');
export const SNOW_LINE_STEP = n('snow', 'line_step');
export const SNOW_LINE_BAND = n('snow', 'band');

// --- Snow steepness ---
export const SNOW_STEEPNESS_DEFAULT = n('snow_steepness', 'default');
export const SNOW_STEEPNESS_MIN = n('snow_steepness', 'min');
export const SNOW_STEEPNESS_MAX = n('snow_steepness', 'max');
export const SNOW_STEEPNESS_STEP = n('snow_steepness', 'step');
export const SNOW_STEEPNESS_ROCK_FACTOR = n('snow_steepness', 'rock_factor');
export const SNOW_STEEPNESS_SNOW_FACTOR = n('snow_steepness', 'snow_factor');

// --- Lake flatness ---
export const LAKE_FLATNESS_DEFAULT = n('lake', 'flatness_default');
export const LAKE_FLATNESS_MIN = n('lake', 'flatness_min');
export const LAKE_FLATNESS_MAX = n('lake', 'flatness_max');
export const LAKE_FLATNESS_STEP = n('lake', 'flatness_step');
export const LAKE_RANGE_DEFAULT = n('lake', 'range_default');
export const LAKE_RANGE_MIN = n('lake', 'range_min');
export const LAKE_RANGE_MAX = n('lake', 'range_max');
export const LAKE_RANGE_STEP = n('lake', 'range_step');

// --- Tile ---
export const TILE_PIXELS = n('tile', 'quads');

// --- Data source per zoom ---
export const SOURCE: Record<number, string> = (cfg['source'] ?? {}) as Record<number, string>;

// --- Region filter ---
export type RegionBBox = [number, number, number, number]; // [lonMin, latMin, lonMax, latMax]

function parseRegions(raw: unknown): Record<string, RegionBBox> {
  const map = (raw ?? {}) as Record<string, number[]>;
  for (const [name, bbox] of Object.entries(map)) {
    if (!Array.isArray(bbox) || bbox.length !== 4) {
      throw new Error(`Region "${name}": expected [lonMin, latMin, lonMax, latMax], got ${JSON.stringify(bbox)}`);
    }
    const [lonMin, latMin, lonMax, latMax] = bbox;
    if (lonMin > lonMax) {
      throw new Error(`Region "${name}": lonMin (${lonMin}) > lonMax (${lonMax}) — cross-antimeridian not supported`);
    }
    if (latMin < -90 || latMax > 90) {
      throw new Error(`Region "${name}": latitude out of range [-90, 90]`);
    }
  }
  // Validate none sentinel
  const none = map['none'];
  if (none) {
    if (none[0] !== 0 || none[1] !== 0 || none[2] !== 0 || none[3] !== 0) {
      throw new Error('Region "none": must be [0, 0, 0, 0]');
    }
  }
  return map as Record<string, RegionBBox>;
}

export const REGIONS: Record<string, RegionBBox> = parseRegions(cfg['regions']);

function parseZoomRegions(raw: unknown, label: string): Record<number, string[]> {
  const map = (raw ?? {}) as Record<number, string[]>;
  const maxZoom = ZOOM_MAX;

  // Check every level 1..maxZoom is present
  for (let z = 1; z <= maxZoom; z++) {
    if (!(z in map)) {
      throw new Error(`[${label}]: missing zoom level ${z} (required: 1-${maxZoom})`);
    }
  }
  // Check no extra keys beyond maxZoom
  for (const key of Object.keys(map)) {
    const z = Number(key);
    if (z < 1 || z > maxZoom) {
      throw new Error(`[${label}]: zoom ${z} out of range (1-${maxZoom})`);
    }
  }

  // Coverage check: coarser(z) must individually contain each finer(z+1) region
  for (let z = 1; z < maxZoom; z++) {
    const coarseNames = map[z];
    const fineNames = map[z + 1];
    for (const fName of fineNames) {
      if (fName === 'none') continue; // none = empty, always contained
      const fBbox = REGIONS[fName];
      if (!fBbox) throw new Error(`[${label}]: region "${fName}" at zoom ${z + 1} not found in [regions]`);
      let contained = false;
      for (const cName of coarseNames) {
        if (cName === 'none') continue;
        const cBbox = REGIONS[cName];
        if (!cBbox) throw new Error(`[${label}]: region "${cName}" at zoom ${z} not found in [regions]`);
        if (bboxContains(cBbox, fBbox)) {
          contained = true;
          break;
        }
      }
      if (!contained) {
        throw new Error(
          `[${label}]: zoom ${z + 1} region "${fName}" ` +
          `([${fBbox}]) is not contained by any zoom ${z} region`,
        );
      }
    }
  }

  return map;
}

function bboxContains(parent: RegionBBox, child: RegionBBox): boolean {
  return parent[0] <= child[0] && parent[1] <= child[1] &&
         parent[2] >= child[2] && parent[3] >= child[3];
}

const ZOOM_REGIONS_PROD = parseZoomRegions(cfg['zoom_regions'], 'zoom_regions');
const ZOOM_REGIONS_DEV = parseZoomRegions(cfg['zoom_regions_dev'], 'zoom_regions_dev');
export const ZOOM_REGIONS: Record<number, string[]> = (
  import.meta.env.DEV ? ZOOM_REGIONS_DEV : ZOOM_REGIONS_PROD
);

// --- Cloud ---
export const CLOUD_SUBDIVISIONS = n('cloud', 'subdivisions');
export const CLOUD_ALTITUDE = n('cloud', 'altitude');
export const CLOUD_DISP_SCALE = n('cloud', 'disp_scale');
export const CLOUD_DISP_BIAS = n('cloud', 'disp_bias');
export const CLOUD_OCTAVES = n('cloud', 'octaves');
export const CLOUD_BASE_FREQ = n('cloud', 'base_freq');
export const CLOUD_LACUNARITY = n('cloud', 'lacunarity');
export const CLOUD_GAIN = n('cloud', 'gain');
export const CLOUD_DRIFT_U = n('cloud', 'drift_u');
export const CLOUD_DRIFT_V = n('cloud', 'drift_v');
export const CLOUD_ALPHA_THRESHOLD = n('cloud', 'alpha_threshold');

// --- Biome ---
export const BIOME_LAPSE_RATE = n('biome', 'lapse_rate');
export const BIOME_LAT_TEMP_FACTOR = n('biome', 'lat_temp_factor');
export const BIOME_T_MIN = n('biome', 't_lut_min');
export const BIOME_T_MAX = n('biome', 't_lut_max');
export const BIOME_P_MIN = n('biome', 'p_lut_min');
export const BIOME_P_MAX = n('biome', 'p_lut_max');

// --- Snow latitude ---
export const SNOW_LAT_FACTOR = n('snow_latitude', 'factor');
export const SNOW_LAT_POWER = n('snow_latitude', 'power');

// --- Snow season ---
export const SNOW_SEASON_AMPLITUDE = n('snow_season', 'amplitude');
export const SNOW_SEASON_LAT_POWER = n('snow_season', 'lat_power');
export const SNOW_SEASON_SPEED_DEFAULT = n('snow_season', 'speed_default');

// --- Sun ---
export const SUN_DIRECTION = {
  x: n('sun', 'direction_x'),
  y: n('sun', 'direction_y'),
  z: n('sun', 'direction_z'),
};
export const SUN_INTENSITY = n('sun', 'intensity');

// --- Fallback ---
export const FALLBACK_ELEV_MIN = n('fallback', 'elev_min');
export const FALLBACK_ELEV_MAX = n('fallback', 'elev_max');
export const TILE_FALLBACK_AMPLITUDE = n('fallback', 'amplitude');
