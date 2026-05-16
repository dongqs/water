/** Benchmark dashboard — main entry point. */

import { loadData, getMeta } from './data';
import { initScatter, updateScatter, onScatterPointClick, resizeScatter } from './scatter';
import { initDimension, updateDimension, resizeDimension } from './dimension';
import { initTable, updateTable, onTableRowClick } from './table';
import { initProfile, showProfile, resizeProfile } from './profile';

// ---- Filters ----
function getFilters() {
  const noise = (document.getElementById("filter-noise") as HTMLSelectElement).value;
  const type = (document.getElementById("filter-type") as HTMLSelectElement).value;
  const source = (document.getElementById("filter-source") as HTMLSelectElement).value;
  const family = (document.getElementById("filter-family") as HTMLSelectElement).value;
  const bodyId = parseInt((document.getElementById("filter-body") as HTMLSelectElement).value) || 0;
  return {
    noise: noise === "all" ? undefined : noise,
    type: type === "all" ? undefined : type,
    source: source === "all" ? undefined : source,
    family: family === "all" ? undefined : family,
    waterBodyId: bodyId || undefined,
  };
}

function updateAll(): void {
  const filters = getFilters();
  updateScatter(filters);
  updateTable(filters);
}

// ---- Initialize ----
async function main(): Promise<void> {
  // Show loading
  const app = document.getElementById("app")!;
  app.innerHTML = '<div style="padding:40px;text-align:center;font-size:16px;">Loading benchmark data...</div>';

  await loadData();

  // Restore layout
  app.innerHTML = ''; // Will be rebuilt — for now reload
  location.reload();
}

// Run on load (but data is loaded in main, layout is in HTML)
document.addEventListener("DOMContentLoaded", async () => {
  await loadData();

  // Populate water body filter
  const meta = getMeta();
  const bodySelect = document.getElementById("filter-body") as HTMLSelectElement;
  for (const wb of meta.water_bodies) {
    const opt = document.createElement("option");
    opt.value = String(wb.id);
    opt.textContent = wb.name;
    bodySelect.appendChild(opt);
  }

  // Init charts
  initScatter(document.getElementById("chart-scatter")!);
  initDimension(document.getElementById("chart-dimension")!);
  initTable();
  initProfile(document.getElementById("chart-profile")!);

  // Initial render
  updateAll();
  updateDimension("grad_op");

  // Filter change handlers
  document.querySelectorAll("#panel-filters select").forEach(sel => {
    sel.addEventListener("change", updateAll);
  });

  // Dimension selector
  document.getElementById("dim-select")!.addEventListener("change", (e) => {
    updateDimension((e.target as HTMLSelectElement).value);
  });

  // Click handlers
  onScatterPointClick((algoLabel: string) => {
    // Select any profile that has this algorithm
    const rows = getMeta().profiles;
    if (rows.length > 0) {
      showProfile(rows[0].id);
    }
  });

  onTableRowClick((algoLabel: string) => {
    // Select first synthetic profile
    const synProfiles = getMeta().profiles.filter(p => p.source === "synthetic");
    if (synProfiles.length > 0) {
      showProfile(synProfiles[0].id);
    }
  });

  // Window resize
  window.addEventListener("resize", () => {
    resizeScatter();
    resizeDimension();
    resizeProfile();
  });
});
