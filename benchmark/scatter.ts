/** Overview scatter plot: boundary offset vs flood ratio, colored by algorithm family. */

import {
  filterMetrics, aggregateByAlgorithm, classifyFamily,
  getAlgorithmFamilyColor, type MetricRow, type Filters,
} from './data';

declare const echarts: any;

let chart: any = null;
let currentFilters: Filters = {};
let onPointClick: ((algoLabel: string) => void) | null = null;

export function initScatter(container: HTMLElement): void {
  chart = echarts.init(container);
}

export function onScatterPointClick(cb: (algoLabel: string) => void): void {
  onPointClick = cb;
}

export function updateScatter(filters: Filters): void {
  if (!chart) return;
  currentFilters = filters;

  const rows = filterMetrics(filters);
  const aggregated = aggregateByAlgorithm(rows);

  // Group by family for legend
  const families = [...new Set(aggregated.map(r => classifyFamily(r)))];
  const familyData: Record<string, [number, number, number, string, string][]> = {};
  for (const fam of families) familyData[fam] = [];

  for (const r of aggregated) {
    const fam = classifyFamily(r);
    familyData[fam].push([
      Math.abs(r.boundary_offset_m),
      r.flood_ratio,
      Math.max(2, 20 / (r.lake_mask_rmse + 0.1)), // point size, bounded
      r.algorithm_label,
      r.lake_mask_rmse.toFixed(4),
    ]);
  }

  const series = families.map(fam => ({
    name: fam,
    type: 'scatter',
    data: familyData[fam],
    symbolSize: (val: number[]) => val[2],
    itemStyle: { color: getAlgorithmFamilyColor(fam), opacity: 0.7 },
    emphasis: { itemStyle: { opacity: 1.0, borderWidth: 2, borderColor: '#fff' } },
  }));

  chart.setOption({
    tooltip: {
      trigger: 'item',
      formatter: (p: any) => {
        const d = p.data;
        return `<b>${d[3]}</b><br/>|Offset|: ${d[0].toFixed(1)}m<br/>Flood: ${(d[1] * 100).toFixed(1)}%<br/>RMSE: ${d[4]}`;
      },
      backgroundColor: 'rgba(22,33,62,0.95)',
      borderColor: '#0f3460',
      textStyle: { color: '#e0e0e0', fontSize: 11 },
    },
    legend: {
      bottom: 0, textStyle: { color: '#e0e0e0', fontSize: 10 },
      data: families,
    },
    grid: { left: 75, right: 30, top: 20, bottom: 55 },
    xAxis: {
      name: '|Boundary Offset| (m)',
      nameLocation: 'center', nameGap: 35,
      nameTextStyle: { color: '#aaa', fontSize: 10 },
      axisLine: { lineStyle: { color: '#0f3460' } },
      axisLabel: { color: '#aaa', fontSize: 9 },
    },
    yAxis: {
      name: 'Flood Ratio',
      nameLocation: 'center', nameGap: 50,
      nameTextStyle: { color: '#aaa', fontSize: 10 },
      axisLine: { lineStyle: { color: '#0f3460' } },
      axisLabel: { color: '#aaa', fontSize: 9 },
    },
    series,
  }, true);

  chart.off('click');
  chart.on('click', (params: any) => {
    if (params.data && onPointClick) {
      onPointClick(params.data[3]); // algorithm_label
    }
  });
}

export function resizeScatter(): void {
  chart?.resize();
}
