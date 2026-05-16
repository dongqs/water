/** Data loading and filtering for the benchmark dashboard. */

// ---- Types ----
export interface WaterBody {
  id: number; name: string; type: string;
  center_lat: number; center_lon: number;
}
export interface ProfileMeta {
  id: number; water_body_id: number; label: string;
  direction: string; sample_count: number; source: string;
}
export interface ProfileData {
  water_body_id: number; source: string;
  dist_m: number[]; elev_raw: number[];
  shore_start: number | null; shore_end: number | null;
}
export interface AlgoConfig {
  id: number; label: string;
  grad_op: string; norm: string; window_stat: string | null;
  pre_smooth: string; multi_scale: string;
  decision_fn: string; elev_guard: string;
  extra_params: Record<string, unknown>;
}
export interface MetricRow {
  run_id: number;
  algorithm_id: number; algorithm_label: string;
  grad_op: string; norm: string; window_stat: string | null;
  pre_smooth: string; multi_scale: string;
  decision_fn: string; elev_guard: string;
  profile_id: number; profile_label: string; profile_source: string;
  water_body_id: number; water_body_name: string; water_body_type: string;
  noise_model: string;
  boundary_offset_m: number; flood_ratio: number; fragment_count: number;
  lake_mask_rmse: number; transition_width_m: number;
}
export interface DimSweepEntry {
  dim_name: string; variant_id: string; variant_label: string;
  count: number; avg_offset_m: number; avg_flood_ratio: number;
  avg_fragment_count: number; avg_rmse: number;
}
export interface MetaJSON {
  water_bodies: WaterBody[];
  profiles: ProfileMeta[];
  profile_data: Record<string, ProfileData>;
  algorithm_configs: AlgoConfig[];
}

// ---- Global state ----
let meta: MetaJSON | null = null;
let metrics: MetricRow[] = [];
let dimSweep: DimSweepEntry[] = [];

export async function loadData(): Promise<void> {
  const [metaRes, metricsRes, dimRes] = await Promise.all([
    fetch('/export/meta.json'),
    fetch('/export/metrics.json'),
    fetch('/export/dim_sweep.json'),
  ]);
  meta = await metaRes.json();
  metrics = await metricsRes.json();
  dimSweep = await dimRes.json();
}

export function getMeta(): MetaJSON { return meta!; }
export function getAllMetrics(): MetricRow[] { return metrics; }
export function getDimSweep(): DimSweepEntry[] { return dimSweep; }

// ---- Filtering ----
export interface Filters {
  noise?: string;
  type?: string;
  source?: string;
  family?: string;
  waterBodyId?: number;
}

export function classifyFamily(row: MetricRow): string {
  const l = row.algorithm_label;
  if (l.includes("_W1_") || l.includes("_W2_") || l.includes("_W3_") || l.includes("_W4_")) return "window";
  if (l.includes("_S1_") || l.includes("_S2_")) return "smooth";
  if (l.includes("_M2_") || l.includes("_M3_") || l.includes("_M4_")) return "multiscale";
  if (l.includes("_D2_") || l.includes("_D3_")) return "decision";
  if (l.includes("_f") && (l.includes("_f20_") || l.includes("_f50_") || l.includes("_f200_") || l.includes("_f500_") || l.includes("_f10_") || l.includes("_f30_") || l.includes("_f40_") || l.includes("_f60_") || l.includes("_f80_") || l.includes("_f150_"))) return "flatness";
  if (l.includes("G2_") || l.includes("G4_") || (l.startsWith("G") && (l.includes("_M4_") || l.includes("_D3_")))) return "combined";
  if (l.startsWith("G2_") || l.startsWith("G3_") || l.startsWith("G4_") || l.startsWith("G5_") || l.startsWith("G6_")) return "grad";
  return "baseline";
}

export function aggregateByAlgorithm(rows: MetricRow[]): MetricRow[] {
  // Group by algorithm_label, average across all profiles
  const groups = new Map<string, { rows: MetricRow[] }>();
  for (const r of rows) {
    if (!groups.has(r.algorithm_label)) groups.set(r.algorithm_label, { rows: [] });
    groups.get(r.algorithm_label)!.rows.push(r);
  }
  const result: MetricRow[] = [];
  for (const [label, group] of groups) {
    const agg = { ...group.rows[0] };
    const n = group.rows.length;
    agg.boundary_offset_m = group.rows.reduce((s, r) => s + r.boundary_offset_m, 0) / n;
    agg.flood_ratio = group.rows.reduce((s, r) => s + r.flood_ratio, 0) / n;
    agg.fragment_count = group.rows.reduce((s, r) => s + r.fragment_count, 0) / n;
    agg.lake_mask_rmse = group.rows.reduce((s, r) => s + r.lake_mask_rmse, 0) / n;
    agg.transition_width_m = group.rows.reduce((s, r) => s + r.transition_width_m, 0) / n;
    result.push(agg);
  }
  return result;
}

export function filterMetrics(filters: Filters = {}): MetricRow[] {
  let rows = metrics;
  if (filters.noise && filters.noise !== "all") rows = rows.filter(r => r.noise_model === filters.noise);
  if (filters.type && filters.type !== "all") rows = rows.filter(r => r.water_body_type === filters.type);
  if (filters.source && filters.source !== "all") rows = rows.filter(r => r.profile_source === filters.source);
  if (filters.waterBodyId && filters.waterBodyId > 0) rows = rows.filter(r => r.water_body_id === filters.waterBodyId);
  if (filters.family) {
    if (filters.family === "all") { /* no filter */ }
    else if (filters.family === "baseline") rows = rows.filter(r => classifyFamily(r) === "baseline");
    else if (filters.family === "grad") rows = rows.filter(r => classifyFamily(r) === "grad");
    else if (filters.family === "window") rows = rows.filter(r => classifyFamily(r) === "window");
    else if (filters.family === "smooth") rows = rows.filter(r => classifyFamily(r) === "smooth");
    else if (filters.family === "multiscale") rows = rows.filter(r => classifyFamily(r) === "multiscale");
    else if (filters.family === "decision") rows = rows.filter(r => classifyFamily(r) === "decision");
    else if (filters.family === "flatness") rows = rows.filter(r => classifyFamily(r) === "flatness");
    else if (filters.family === "combined") rows = rows.filter(r => classifyFamily(r) === "combined");
  }
  return rows;
}

export function getAlgorithmFamilyColor(family: string): string {
  const colors: Record<string, string> = {
    baseline: "#3498db",
    grad: "#2ecc71",
    window: "#e74c3c",
    smooth: "#f39c12",
    multiscale: "#9b59b6",
    decision: "#1abc9c",
    flatness: "#e67e22",
    combined: "#ff6b6b",
  };
  return colors[family] || "#95a5a6";
}
