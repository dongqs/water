/** Sortable algorithm ranking table. */

import {
  filterMetrics, aggregateByAlgorithm, classifyFamily,
  getAlgorithmFamilyColor, type MetricRow, type Filters,
} from './data';

let currentFilters: Filters = {};
let currentRows: MetricRow[] = [];
let sortCol = "lake_mask_rmse";
let sortAsc = true;
let onRowClick: ((algoLabel: string) => void) | null = null;

export function onTableRowClick(cb: (algoLabel: string) => void): void {
  onRowClick = cb;
}

function renderTable(rows: MetricRow[]): void {
  const tbody = document.querySelector("#algo-table tbody")!;
  tbody.innerHTML = "";

  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");
    const fam = classifyFamily(r);
    tr.style.borderLeft = `3px solid ${getAlgorithmFamilyColor(fam)}`;
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td title="${r.algorithm_label}">${r.algorithm_label.length > 40 ? r.algorithm_label.slice(0, 38) + '...' : r.algorithm_label}</td>
      <td>${r.lake_mask_rmse.toFixed(4)}</td>
      <td>${Math.abs(r.boundary_offset_m).toFixed(1)}</td>
      <td>${(r.flood_ratio * 100).toFixed(1)}%</td>
      <td>${r.fragment_count.toFixed(1)}</td>
      <td>${r.transition_width_m.toFixed(1)}</td>
    `;
    tr.addEventListener("click", () => {
      document.querySelectorAll("#algo-table tr.selected").forEach(el => el.classList.remove("selected"));
      tr.classList.add("selected");
      if (onRowClick) onRowClick(r.algorithm_label);
    });
    tbody.appendChild(tr);
  });

  // Update sort header indicators
  document.querySelectorAll("#algo-table th").forEach(th => {
    const col = (th as HTMLElement).dataset.sort;
    if (col === sortCol) {
      (th as HTMLElement).style.color = "#3498db";
      (th as HTMLElement).textContent = `${(th as HTMLElement).textContent!.replace(/ [▲▼]$/, "")} ${sortAsc ? "▲" : "▼"}`;
    } else {
      (th as HTMLElement).style.color = "";
    }
  });
}

export function updateTable(filters: Filters): void {
  currentFilters = filters;
  const rows = filterMetrics(filters);
  const aggregated = aggregateByAlgorithm(rows);
  currentRows = aggregated;
  sortAndRender();
}

function sortAndRender(): void {
  const sorted = [...currentRows].sort((a, b) => {
    let va: number, vb: number;
    if (sortCol === "algorithm_label") {
      return sortAsc ? a.algorithm_label.localeCompare(b.algorithm_label) : b.algorithm_label.localeCompare(a.algorithm_label);
    }
    if (sortCol === "boundary_offset_m_abs") {
      va = Math.abs(a.boundary_offset_m);
      vb = Math.abs(b.boundary_offset_m);
    } else if (sortCol === "rank") {
      va = a.lake_mask_rmse;
      vb = b.lake_mask_rmse;
    } else {
      va = (a as any)[sortCol] ?? 0;
      vb = (b as any)[sortCol] ?? 0;
    }
    return sortAsc ? va - vb : vb - va;
  });
  renderTable(sorted);
}

export function initTable(): void {
  document.querySelectorAll("#algo-table th").forEach(th => {
    const el = th as HTMLElement;
    el.addEventListener("click", () => {
      const col = el.dataset.sort;
      if (col === sortCol) {
        sortAsc = !sortAsc;
      } else {
        sortCol = col || "lake_mask_rmse";
        sortAsc = col === "lake_mask_rmse" || col === "boundary_offset_m_abs" || col === "flood_ratio" || col === "rank";
      }
      sortAndRender();
    });
  });
}
