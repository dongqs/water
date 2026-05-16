/** Dimension sweep bar chart — grouped bars per variant. */

import { getDimSweep, type DimSweepEntry } from './data';

declare const echarts: any;

let chart: any = null;

export function initDimension(container: HTMLElement): void {
  chart = echarts.init(container);
}

export function updateDimension(dimName: string): void {
  if (!chart) return;
  const entries = getDimSweep().filter(e => e.dim_name === dimName);

  const labels = entries.map(e => e.variant_label);
  const offsets = entries.map(e => Math.abs(e.avg_offset_m));
  const floods = entries.map(e => e.avg_flood_ratio);
  const frags = entries.map(e => e.avg_fragment_count);
  const rmses = entries.map(e => e.avg_rmse);

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(22,33,62,0.95)',
      borderColor: '#0f3460',
      textStyle: { color: '#e0e0e0', fontSize: 11 },
    },
    legend: {
      data: ['Avg |Offset| (m)', 'Avg Flood Ratio', 'Avg Fragments', 'Avg RMSE'],
      bottom: 0,
      textStyle: { color: '#e0e0e0', fontSize: 10 },
    },
    grid: { left: 80, right: 85, top: 20, bottom: 60 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: '#aaa', fontSize: 9, rotate: 25 },
      axisLine: { lineStyle: { color: '#0f3460' } },
      name: 'Variant',
      nameLocation: 'center', nameGap: 40,
      nameTextStyle: { color: '#aaa', fontSize: 10 },
    },
    yAxis: [
      {
        type: 'value',
        name: 'Meters',
        nameLocation: 'center', nameGap: 55,
        nameTextStyle: { color: '#aaa', fontSize: 10 },
        axisLabel: { color: '#aaa', fontSize: 9 },
        axisLine: { lineStyle: { color: '#0f3460' } },
      },
      {
        type: 'value',
        name: 'Ratio/Count',
        nameLocation: 'center', nameGap: 55,
        nameTextStyle: { color: '#aaa', fontSize: 10 },
        axisLabel: { color: '#aaa', fontSize: 9 },
        axisLine: { lineStyle: { color: '#0f3460' } },
      },
    ],
    series: [
      {
        name: 'Avg |Offset| (m)', type: 'bar',
        data: offsets,
        itemStyle: { color: '#3498db' },
      },
      {
        name: 'Avg Flood Ratio', type: 'bar',
        data: floods,
        itemStyle: { color: '#e74c3c' },
      },
      {
        name: 'Avg Fragments', type: 'bar',
        data: frags,
        itemStyle: { color: '#2ecc71' },
      },
      {
        name: 'Avg RMSE', type: 'line',
        yAxisIndex: 1,
        data: rmses,
        lineStyle: { color: '#f39c12', width: 2 },
        itemStyle: { color: '#f39c12' },
        symbol: 'diamond',
      },
    ],
  }, true);
}

export function resizeDimension(): void {
  chart?.resize();
}
