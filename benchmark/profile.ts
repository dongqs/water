/** Profile detail chart — elevation + lakeMask overlay for selected algorithms. */

import { getMeta, getAlgorithmFamilyColor, type ProfileData } from './data';
import { computeLakeMask, computeLakeMaskConfig, type LakeParams } from './algorithms';

declare const echarts: any;

let chart: any = null;
let selectedAlgos: string[] = [];

export function initProfile(container: HTMLElement): void {
  chart = echarts.init(container);
}

export function setSelectedAlgos(algos: string[]): void {
  selectedAlgos = algos;
}

export function showProfile(profileId: number): void {
  const meta = getMeta();
  const pd: ProfileData | undefined = meta.profile_data[String(profileId)];
  if (!pd) return;

  const profileMeta = meta.profiles.find(p => p.id === profileId);
  const waterBody = meta.water_bodies.find(wb => wb.id === profileMeta?.water_body_id);

  // Show panel
  const panel = document.getElementById("panel-profile")!;
  panel.classList.add("active");
  document.getElementById("profile-title")!.textContent =
    `${waterBody?.name ?? "?"} — ${profileMeta?.label ?? "?"} (${pd.source})`;
  document.getElementById("profile-info")!.textContent =
    `Shore: [${pd.shore_start}, ${pd.shore_end}], ${pd.dist_m.length} samples`;

  // Build algo selector
  const selector = document.getElementById("algo-selector")!;
  const topAlgos = meta.algorithm_configs
    .filter(a => !a.label.includes("_W1_") && !a.label.includes("_W2_") && !a.label.includes("_W3_") && !a.label.includes("_W4_"))
    .slice(0, 30);

  selector.innerHTML = "";
  const defaultSelected = meta.algorithm_configs
    .filter(a => a.label === "G1_N1_S0_M1_D1_E3" || a.label.startsWith("G2_S0_M4_r1_D1_f10"))
    .map(a => a.label);

  if (selectedAlgos.length === 0) selectedAlgos = defaultSelected;

  for (const ac of topAlgos) {
    const lbl = document.createElement("label");
    lbl.className = selectedAlgos.includes(ac.label) ? "checked" : "";
    lbl.innerHTML = `<input type="checkbox" value="${ac.label}"> ${ac.label.slice(0, 35)}`;
    lbl.addEventListener("click", () => {
      const idx = selectedAlgos.indexOf(ac.label);
      if (idx >= 0) selectedAlgos.splice(idx, 1);
      else if (selectedAlgos.length < 8) selectedAlgos.push(ac.label);
      // Re-render
      showProfile(profileId);
    });
    selector.appendChild(lbl);
  }

  // Build chart
  renderProfileChart(pd, profileId);
}

function renderProfileChart(pd: ProfileData, profileId: number): void {
  if (!chart) return;
  const meta = getMeta();

  // Elevation curve
  const elevSeries: any = {
    name: 'Elevation',
    type: 'line',
    data: pd.dist_m.map((d, i) => [d, pd.elev_raw[i]]),
    lineStyle: { color: '#000', width: 1.5 },
    symbol: 'none',
  };

  // Water region highlight
  let waterMark: any = null;
  if (pd.shore_start != null && pd.shore_end != null) {
    waterMark = {
      name: 'Water (GT)',
      type: 'line',
      data: pd.dist_m.map((d, i) => [
        d,
        (i >= pd.shore_start! && i <= pd.shore_end!) ? pd.elev_raw[i] : NaN,
      ]),
      lineStyle: { color: '#3498db', width: 4, opacity: 0.3 },
      symbol: 'none',
      areaStyle: { color: 'rgba(52,152,219,0.1)' },
    };
  }

  // Lake mask curves
  const arr = new Float64Array(pd.elev_raw.map(v => v ?? 0));
  const maskSeries: any[] = [];
  const colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#ff6b6b', '#3498db'];

  for (let ai = 0; ai < selectedAlgos.length; ai++) {
    const ac = meta.algorithm_configs.find(a => a.label === selectedAlgos[ai]);
    if (!ac) continue;

    const params: LakeParams = {};
    if (ac.extra_params && typeof ac.extra_params === 'object') {
      const ep = ac.extra_params as Record<string, unknown>;
      if (typeof ep.flatness === 'number') params.flatness = ep.flatness;
      if (typeof ep.lake_range === 'number') params.lake_range = ep.lake_range;
      if (typeof ep.grad_op === 'string') params.grad_op = ep.grad_op;
      if (typeof ep.multi_scale === 'string') params.multi_scale = ep.multi_scale;
      if (typeof ep.decision_fn === 'string') params.decision_fn = ep.decision_fn;
      if (typeof ep.decision_k === 'number') params.decision_k = ep.decision_k;
    }

    const mask = computeLakeMask(arr, params);
    const shortLabel = ac.label.length > 35 ? ac.label.slice(0, 33) + '...' : ac.label;

    maskSeries.push({
      name: shortLabel,
      type: 'line',
      data: pd.dist_m.map((d, i) => [d, mask[i] ?? 0]),
      lineStyle: { color: colors[ai % colors.length], width: 1.5 },
      symbol: 'none',
    });
  }

  const yAxisMax = Math.max(...pd.elev_raw.filter(v => !isNaN(v))) * 1.2;

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(22,33,62,0.95)',
      borderColor: '#0f3460',
      textStyle: { color: '#e0e0e0', fontSize: 10 },
    },
    legend: {
      data: ['Elevation', 'Water (GT)', ...maskSeries.map(s => s.name)],
      bottom: 0,
      textStyle: { color: '#e0e0e0', fontSize: 9 },
      type: 'scroll',
    },
    grid: { left: 80, right: 30, top: 20, bottom: 70 },
    xAxis: {
      type: 'value',
      name: 'Distance (m)',
      nameLocation: 'center', nameGap: 40,
      nameTextStyle: { color: '#aaa', fontSize: 10 },
      axisLabel: { color: '#aaa', fontSize: 9 },
      axisLine: { lineStyle: { color: '#0f3460' } },
    },
    yAxis: {
      type: 'value',
      name: 'Elev / Mask',
      nameLocation: 'center', nameGap: 55,
      nameTextStyle: { color: '#aaa', fontSize: 10 },
      min: 0,
      max: yAxisMax,
      axisLabel: { color: '#aaa', fontSize: 9 },
      axisLine: { lineStyle: { color: '#0f3460' } },
    },
    series: [elevSeries, waterMark, ...maskSeries].filter(Boolean),
  }, true);
}

export function resizeProfile(): void {
  chart?.resize();
}
