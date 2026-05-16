/** Canvas OSM water body boundary viewer with pan/zoom + sampling grid. */

interface PatchMeta {
  file_bin: string; water_body: string; wb_type: string;
  lat: number; lon: number; grid_row: number; grid_col: number; wb_id: number;
}
interface OSMFeature {
  type: string; properties: { name: string; water_body: string; type: string; role?: string; wb_type?: string };
  geometry: { type: string; coordinates: number[][][][] | number[][][] };
}

let meta: PatchMeta[] = [];
let geojson: any = null;
let waterBodies: string[] = [];
let currentWB = "";

let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;

let viewX = 0, viewY = 0, viewScale = 2;
let dragging = false;
let dragStartX = 0, dragStartY = 0, dragViewX = 0, dragViewY = 0;
let mouseX = 0, mouseY = 0;
let showOSM = true, showSamples = true, goodOnly = false, showClassify = true, flatnessVal = 1.1;
const FLATNESS_LIST = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 5.0];
let refLat = 30, refLon = 120;
const TOPBAR_H = 36;
const loggedWBs = new Set<string>();

const RING_EPS = 1e-5, CLOSE_EPS = 1e-4;
const ptEq = (a: number[], b: number[]) => Math.abs(a[0] - b[0]) < RING_EPS && Math.abs(a[1] - b[1]) < RING_EPS;
const nearlySame = (a: number[], b: number[]) => Math.abs(a[0] - b[0]) < CLOSE_EPS && Math.abs(a[1] - b[1]) < CLOSE_EPS;

// Water body colors
const WB_COLORS = [
  ['#3498db', 'rgba(52,152,219,0.15)'], ['#2ecc71', 'rgba(46,204,113,0.15)'],
  ['#e74c3c', 'rgba(231,76,60,0.15)'],   ['#f39c12', 'rgba(243,156,18,0.15)'],
  ['#9b59b6', 'rgba(155,89,182,0.15)'],  ['#1abc9c', 'rgba(26,188,156,0.15)'],
  ['#e67e22', 'rgba(230,126,34,0.15)'],
];
function wbColor(wb: string) { const i = waterBodies.indexOf(wb); return WB_COLORS[i >= 0 ? i % WB_COLORS.length : 0]; }

// ---- Ring assembly ----
function assembleRings(segments: number[][][], requireClosed = true): number[][][] {
  if (segments.length === 0) return [];
  const used = new Set<number>();
  const result: number[][][] = [];
  const pickNext = (): number[][] | null => {
    for (let i = 0; i < segments.length; i++) { if (!used.has(i)) { used.add(i); return segments[i].slice(); } }
    return null;
  };
  const findConnected = (pt: number[]) => {
    for (const eq of [ptEq, nearlySame]) {
      for (let i = 0; i < segments.length; i++) {
        if (used.has(i)) continue;
        const s = segments[i]; if (!s || s.length < 2) continue;
        const first = s[0], last = s[s.length - 1];
        if (eq(last, pt)) return { i, pts: s.slice(0, -1).reverse() };
        if (eq(first, pt)) return { i, pts: s.slice(1) };
      }
    }
    return null;
  };
  let chain = pickNext();
  while (chain) {
    let changed = true;
    while (changed) {
      changed = false;
      const end = chain[chain.length - 1];
      let next = findConnected(end);
      if (next) { used.add(next.i); chain.push(...next.pts); changed = true; continue; }
      const start = chain[0];
      next = findConnected(start);
      if (next) { used.add(next.i); chain.unshift(...next.pts); changed = true; }
    }
    if (chain.length >= 2) {
      const [s, e] = [chain[0], chain[chain.length - 1]];
      if (!requireClosed) { result.push(chain); }
      else if (ptEq(s, e) || nearlySame(s, e)) { result.push(chain); }
      else if (chain.length >= 3 && nearlySame(s, e)) { chain.push([s[0], s[1]]); result.push(chain); }
    }
    chain = pickNext();
  }
  return result;
}

// Split a ring at gaps > maxGapDeg (avoids long straight lines across map)
function splitAtGaps(ring: number[][], maxGapDeg = 0.05): number[][][] {
  if (ring.length < 2) return [ring];
  const result: number[][][] = [];
  let chunk: number[][] = [ring[0]];
  for (let i = 1; i < ring.length; i++) {
    const dlat = Math.abs(ring[i][1] - ring[i - 1][1]);
    const dlon = Math.abs(ring[i][0] - ring[i - 1][0]);
    if (dlat > maxGapDeg || dlon > maxGapDeg) {
      if (chunk.length >= 2) result.push(chunk);
      chunk = [ring[i]];
    } else { chunk.push(ring[i]); }
  }
  if (chunk.length >= 2) result.push(chunk);
  return result;
}

// ---- Sampling points along lake boundary ----
interface SamplePt {
  lat: number; lon: number;
  index: number;
}

function sampleAlongRing(ring: number[][], spacingM: number, startIdx: number): SamplePt[] {
  if (ring.length < 2) return [];
  // Compute cumulative distances along the ring
  const dists: number[] = [0];
  const midLat = ring.reduce((s, p) => s + p[1], 0) / ring.length;
  const mPerDegLon = 111320 * Math.cos(midLat * Math.PI / 180);
  for (let i = 1; i < ring.length; i++) {
    const dlat = (ring[i][1] - ring[i - 1][1]) * 111320;
    const dlon = (ring[i][0] - ring[i - 1][0]) * mPerDegLon;
    dists.push(dists[i - 1] + Math.sqrt(dlat * dlat + dlon * dlon));
  }
  const totalLen = dists[dists.length - 1];
  if (totalLen < spacingM) {
    // Short ring: take midpoint
    const mid = ring[Math.floor(ring.length / 2)];
    return [{ lat: mid[1], lon: mid[0], index: startIdx }];
  }

  // Place points at intervals
  const result: SamplePt[] = [];
  for (let d = 0; d < totalLen; d += spacingM) {
    // Find segment containing distance d
    let seg = 1;
    while (seg < dists.length && dists[seg] < d) seg++;
    if (seg >= dists.length) seg = dists.length - 1;
    const t = (d - dists[seg - 1]) / Math.max(dists[seg] - dists[seg - 1], 0.001);
    const lat = ring[seg - 1][1] + t * (ring[seg][1] - ring[seg - 1][1]);
    const lon = ring[seg - 1][0] + t * (ring[seg][0] - ring[seg - 1][0]);
    result.push({ lat, lon, index: startIdx + result.length });
  }
  return result;
}

function generateSamples(features: OSMFeature[]): SamplePt[] {
  // Assemble outer rings
  const outerSegs: number[][][] = [];
  for (const feat of features) {
    if (feat.properties.role === 'inner') continue;
    const ring = feat.geometry.type === 'Polygon' ? feat.geometry.coordinates[0] as number[][]
      : (feat.geometry.coordinates as number[][][][])[0][0];
    if (ring && ring.length >= 2) outerSegs.push(ring);
  }
  const assembled = assembleRings(outerSegs, true);

  // Sample along each ring
  let allSamples: SamplePt[] = [];
  let idx = 0;
  for (const ring of assembled) {
    const samples = sampleAlongRing(ring, 1000, idx);
    allSamples = allSamples.concat(samples);
    idx += samples.length;
  }
  return allSamples;
}

// ---- Drawing helpers ----
function geoToScreen(lat: number, lon: number): [number, number] {
  const cosLat = Math.cos(refLat * Math.PI / 180);
  const dx = (lon - refLon) * 111320 * cosLat;
  const dy = (refLat - lat) * 111320;
  const W = canvas.width, H = canvas.height;
  return [W / 2 + dx / 30 * viewScale + viewX, H / 2 + dy / 30 * viewScale + viewY];
}

function drawRing(ring: number[][], close = true) {
  const pts: [number, number][] = [];
  for (const p of ring) {
    if (isFinite(p[0]) && isFinite(p[1]) && Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90) {
      pts.push(geoToScreen(p[1], p[0]));
    }
  }
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  if (close) ctx.closePath();
}

// Cache sample points and elevation data
const sampleCache = new Map<string, SamplePt[]>();
const elevCache = new Map<number, Float32Array>();
interface SampleRec {
  i: number; k: string; t: string; lat: number; lon: number; png: string;
  q?: number; we?: number; wf?: number; gm?: number[]; ac?: Record<string, number>;
}
let sampleMeta: SampleRec[] | null = null;

function decodeTerrainRGB(img: HTMLImageElement): Float32Array {
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const cx = c.getContext('2d')!;
  cx.drawImage(img, 0, 0);
  const data = cx.getImageData(0, 0, img.width, img.height).data;
  const elev = new Float32Array(img.width * img.height);
  for (let i = 0; i < elev.length; i++) {
    const R = data[i * 4], G = data[i * 4 + 1], B = data[i * 4 + 2];
    elev[i] = -10000.0 + (R * 65536.0 + G * 256.0 + B) * 0.1;
  }
  return elev;
}

function loadPNG(url: string): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(decodeTerrainRGB(img));
    img.onerror = reject;
    img.src = url;
  });
}

async function loadSamples() {
  try {
    const resp = await fetch('/export/samples/sample_meta.json');
    sampleMeta = await resp.json();
    const batchSize = 50;
    let loaded = 0;
    for (let i = 0; i < sampleMeta!.length; i += batchSize) {
      const batch = sampleMeta!.slice(i, i + batchSize);
      await Promise.all(batch.map(async (s) => {
        const elev = await loadPNG(`/export/samples/png/${s.png}`);
        elevCache.set(s.i, elev);
      }));
      loaded += batch.length;
      if (loaded % 200 === 0) console.log(`Loaded ${loaded}/${sampleMeta!.length} PNGs`);
    }
    console.log(`Loaded ${sampleMeta!.length} samples`);
  } catch (e) { console.warn('Sample data not available:', e); }
}

function elevColor(m: number): [number, number, number] {
  if (isNaN(m)) return [10, 10, 30];
  // Absolute elevation, color cycle every 32m: blue→cyan→green→yellow→red
  // Water < 0m = deep blue; each 32m band repeats the rainbow
  const band = 32;
  const t = ((m % band) + band) % band / band;  // 0..1 within each 32m band
  const h = 240 * (1 - t);  // blue=240°, red=0°
  const s = 0.85;
  const l = 0.35 + t * 0.25;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((h / 60) % 2 - 1));
  const m2 = l - c / 2;
  let r2: number, g2: number, b2: number;
  if (h < 60) { r2 = c; g2 = x; b2 = 0; }
  else if (h < 120) { r2 = x; g2 = c; b2 = 0; }
  else if (h < 180) { r2 = 0; g2 = c; b2 = x; }
  else if (h < 240) { r2 = 0; g2 = x; b2 = c; }
  else { r2 = x; g2 = 0; b2 = c; }
  // Darken water (< 0m) significantly
  const brightness = m < 0 ? 0.4 : 1.0;
  return [
    Math.floor((r2 + m2) * 255 * brightness),
    Math.floor((g2 + m2) * 255 * brightness),
    Math.floor((b2 + m2) * 255 * brightness),
  ];
}

function render() {
  if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);

  if (geojson && showOSM) {
    const byWB = new Map<string, OSMFeature[]>();
    for (const f of geojson.features) {
      const wb = f.properties.water_body;
      if (wb === '青海湖') continue;
      if (!byWB.has(wb)) byWB.set(wb, []);
      byWB.get(wb)!.push(f);
    }

    for (const [wb, features] of byWB) {
      const outerSegs: number[][][] = [];
      const innerSegs: number[][][] = [];
      for (const feat of features) {
        const geom = feat.geometry;
        const role = feat.properties.role || 'outer';
        const ring = geom.type === 'Polygon' ? geom.coordinates[0] as number[][]
          : (geom.coordinates as number[][][][])[0][0];
        if (ring && ring.length >= 2) {
          (role === 'inner' ? innerSegs : outerSegs).push(ring);
        }
      }
      const assembledOuter = assembleRings(outerSegs, true);
      const assembledInner = assembleRings(innerSegs, true);
      const [stroke, fill] = wbColor(wb);
      const isSelected = wb === currentWB;
      const isRiver = features[0]?.properties?.wb_type === 'river';
      const lw = Math.max(isRiver ? 2.5 : 1.0, isSelected ? viewScale * 0.6 : viewScale * 0.25);

      if (!loggedWBs.has(wb)) {
        console.warn(`${wb}: ${outerSegs.length}o→${assembledOuter.length} rings, ${innerSegs.length}i→${assembledInner.length} rings, isRiver=${isRiver}`);
        loggedWBs.add(wb);
      }

      if (isRiver) {
        // Rivers: use assembled rings if possible, else raw segments (split at gaps)
        const riverRings = [...assembledOuter, ...assembledInner];
        const sources = riverRings.length > 0 ? riverRings : [...outerSegs, ...innerSegs];
        for (const ring of sources) {
          // Split at gaps > 0.05° to avoid long straight lines
          const chunks = splitAtGaps(ring, 0.05);
          for (const chunk of chunks) {
            if (chunk.length >= 2) {
              drawRing(chunk, false);
              ctx.strokeStyle = stroke;
              ctx.lineWidth = lw;
              ctx.stroke();
            }
          }
        }
      } else {
        // Lake fill + island holes
        for (const ring of assembledOuter) { drawRing(ring, true); ctx.fillStyle = fill; ctx.fill(); }
        for (const ring of assembledInner) { drawRing(ring, true); ctx.fillStyle = '#1a1a2e'; ctx.fill(); }
        for (const ring of [...assembledOuter, ...assembledInner]) {
          drawRing(ring, true); ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke();
        }
      }

      // Label for selected
      if (isSelected && assembledOuter.length > 0) {
        const r = assembledOuter[0];
        let clat = 0, clon = 0;
        for (const p of r) { clat += p[1]; clon += p[0]; }
        clat /= r.length; clon /= r.length;
        const [lx, ly] = geoToScreen(clat, clon);
        ctx.font = `${Math.max(14, 20 * viewScale / 3)}px monospace`;
        ctx.textAlign = 'center';
        ctx.strokeStyle = '#000'; ctx.lineWidth = 4;
        ctx.strokeText(wb, lx, ly - 12);
        ctx.fillStyle = stroke;
        ctx.fillText(wb, lx, ly - 12);
      }

      // Sampling rectangles with 8x8 JAXA elevation heatmaps
      const lakeSamples = sampleMeta?.filter(s =>
        s.k === wb && (!goodOnly || s.q === 1));
      if (showSamples && lakeSamples && lakeSamples.length > 0 && !isRiver) {
        const PX = 16;  // patch size: 16×16 JAXA pixels
        const halfDeg = PX / 2 / 3600;

        for (const s of lakeSamples) {
          const [sxNW, syNW] = geoToScreen(s.lat + halfDeg, s.lon - halfDeg);
          const [sxSE, sySE] = geoToScreen(s.lat - halfDeg, s.lon + halfDeg);
          const w = sxSE - sxNW;
          const h = sySE - syNW;

          // Cull off-screen patches
          if (sxSE < -w || sxNW > canvas.width + w || sySE < -h || syNW > canvas.height + h) continue;

          // 16×16 elevation heatmap with absolute color (32m cycle)
          const elev = elevCache.get(s.i);
          if (elev && w > 2 && h > 2) {
            const stride = Math.round(Math.sqrt(elev.length));
            const iw = Math.ceil(w), ih = Math.ceil(h);
            const imgData = ctx.createImageData(iw, ih);
            for (let py = 0; py < ih; py++) {
              const row = Math.floor(py / ih * stride);
              for (let px = 0; px < iw; px++) {
                const col = Math.floor(px / iw * stride);
                const [r, g, b] = elevColor(elev[row * stride + col]);
                const idx = (py * iw + px) * 4;
                imgData.data[idx] = r; imgData.data[idx + 1] = g;
                imgData.data[idx + 2] = b; imgData.data[idx + 3] = 255;
              }
            }
            ctx.putImageData(imgData, Math.floor(sxNW), Math.floor(syNW));

            // ---- Flat water detection overlay (baseline algorithm) ----
            if (showClassify) {
              const flatness = flatnessVal;
              // Compute lakeMask (continuous 0..1) for each pixel
              const lakeMask = new Float32Array(stride * stride);
              for (let r = 1; r < stride - 1; r++) {
                for (let c = 1; c < stride - 1; c++) {
                  // Central difference gradient
                  const du = elev[(r) * stride + (c + 1)] - elev[(r) * stride + (c - 1)];
                  const dv = elev[(r + 1) * stride + c] - elev[(r - 1) * stride + c];
                  const steep = Math.sqrt(du * du + dv * dv) / 2;
                  // smoothstep: 1=flat(water), 0=steep(land)
                  const lo = flatness * 0.3, hi = flatness;
                  if (steep <= lo) lakeMask[r * stride + c] = 1.0;
                  else if (steep >= hi) lakeMask[r * stride + c] = 0.0;
                  else {
                    const t = (steep - lo) / (hi - lo);
                    lakeMask[r * stride + c] = 1.0 - t * t * (3.0 - 2.0 * t);
                  }
                }
              }
              // Overlay: blue tint = water (lakeMask→1), warm/dark = land (lakeMask→0)
              for (let r = 0; r < stride; r++) {
                for (let c = 0; c < stride; c++) {
                  const m = lakeMask[r * stride + c];
                  if (m >= 0.99) continue;  // pure water: no overlay (show elevation color)
                  const ox = Math.floor(sxNW + c / stride * w);
                  const oy = Math.floor(syNW + r / stride * h);
                  const pw = Math.max(1, Math.ceil(w / stride));
                  const ph = Math.max(1, Math.ceil(h / stride));
                  if (m <= 0.01) {
                    // Confident land: red-brown overlay
                    ctx.fillStyle = 'rgba(180,80,40,0.6)';
                  } else if (m < 0.5) {
                    // Probable land: warm overlay
                    const a = 0.3 + (0.5 - m) * 0.5;
                    ctx.fillStyle = `rgba(200,120,60,${a})`;
                  } else {
                    // Probable water: slight blue boost
                    const a = (m - 0.5) * 0.2;
                    ctx.fillStyle = `rgba(30,80,180,${a})`;
                  }
                  ctx.fillRect(ox, oy, pw, ph);
                }
              }
            }
          }  // end if (elev && w > 2 && h > 2)

          // Border color by sample type
          const stype = s.t;
          const isGood = s.q === 1;
          ctx.lineWidth = Math.max(0.5, (isGood ? 1.5 : 0.3) * viewScale * 0.1);
          if (stype === 'water') ctx.strokeStyle = 'rgba(52,152,219,0.8)';
          else if (stype === 'land') ctx.strokeStyle = 'rgba(231,76,60,0.8)';
          else ctx.strokeStyle = isGood ? '#2ecc71' : 'rgba(128,128,128,0.3)';
          ctx.strokeRect(sxNW, syNW, w, h);

          const [cx, cy] = geoToScreen(s.lat, s.lon);
          const fontSize = Math.max(8, 10 * viewScale / 3);
          ctx.font = `${fontSize}px monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          const acc = s.ac ? s.ac[String(flatnessVal)] : undefined;
          const typePrefix = stype === 'water' ? 'W' : (stype === 'land' ? 'L' : '');
          const label = typePrefix
            ? `${s.i}${typePrefix}${acc != null ? ' A' + (acc*100).toFixed(0) : ''}`
            : (isGood && s.we != null
              ? `${s.i}:${Math.round(s.we!)}m${acc != null ? ' A' + (acc*100).toFixed(0) : ''}`
              : String(s.i));
          const tw = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(0,0,0,0.7)';
          ctx.fillRect(cx - tw / 2 - 2, cy - fontSize / 2 - 1, tw + 4, fontSize + 2);
          ctx.fillStyle = isGood ? '#2ecc71' : '#999';
          ctx.fillText(label, cx, cy);
        }
      }
    }
  }

  // Status bar
  const cosLat = Math.cos(refLat * Math.PI / 180);
  const hoverLat = refLat - ((mouseY - H / 2) - viewY) / (111320 * viewScale);
  const hoverLon = refLon + ((mouseX - W / 2) - viewX) / (111320 * cosLat * viewScale);
  document.getElementById('info')!.textContent =
    `ref ${refLat.toFixed(4)}, ${refLon.toFixed(4)}  cursor ${hoverLat.toFixed(4)}, ${hoverLon.toFixed(4)}  ` +
    `scale ${viewScale.toFixed(2)}x  ${waterBodies.length} bodies ` +
    `f=${flatnessVal} [O:OSM S:samp Q:qual C:class R:recenter F:fit]`;
}

function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight - TOPBAR_H; render(); }

function centerOn(wb: string) {
  const patches = meta.filter(m => m.water_body === wb);
  if (patches.length > 0) {
    const mid = patches[Math.floor(patches.length / 2)];
    refLat = mid.lat; refLon = mid.lon;
  }
  viewX = 0; viewY = 0;
  viewScale = Math.min(6, canvas.width / (5000 / 30));
  render();
}

function fitAllBodies() {
  if (meta.length === 0) return;
  let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
  for (const pm of meta) {
    if (pm.lat < minLat) minLat = pm.lat;
    if (pm.lat > maxLat) maxLat = pm.lat;
    if (pm.lon < minLon) minLon = pm.lon;
    if (pm.lon > maxLon) maxLon = pm.lon;
  }
  refLat = (minLat + maxLat) / 2;
  refLon = (minLon + maxLon) / 2;
  const cl = Math.cos(refLat * Math.PI / 180);
  const spanM = Math.max((maxLat - minLat) * 111320, (maxLon - minLon) * 111320 * cl) * 1.2;
  viewScale = Math.min(10, canvas.width / (spanM / 30));
  viewX = 0; viewY = 0;
  render();
}

export async function loadPatches() {
  canvas = document.getElementById('canvas') as HTMLCanvasElement;
  ctx = canvas.getContext('2d')!;
  resize();

  const [metaResp, geoResp] = await Promise.all([
    fetch('/export/patch_meta.json'),
    fetch('/export/boundaries.geojson'),
  ]);
  meta = await metaResp.json();
  if (meta.length > 0) { refLat = meta[0].lat; refLon = meta[0].lon; }
  try { geojson = await geoResp.json(); } catch (e) { console.warn('OSM not available'); }

  waterBodies = [...new Set(meta.map(m => m.water_body))].filter(w => w !== '青海湖');
  const sel = document.getElementById('sel-water') as HTMLSelectElement;
  waterBodies.forEach(w => { const o = document.createElement('option'); o.value = w; o.textContent = w; sel.appendChild(o); });
  currentWB = waterBodies.find(w => w === '西湖') || waterBodies[0];
  sel.value = currentWB;
  sel.addEventListener('change', () => { currentWB = sel.value; centerOn(currentWB); });

  document.getElementById('btn-zoom-in')!.onclick = () => { viewScale *= 1.5; render(); };
  document.getElementById('btn-zoom-out')!.onclick = () => { viewScale /= 1.5; render(); };
  document.getElementById('btn-reset')!.onclick = () => centerOn(currentWB);
  const flatSelect = document.getElementById('sel-flatness') as HTMLSelectElement;
  FLATNESS_LIST.forEach(f => {
    const o = document.createElement('option');
    o.value = String(f); o.textContent = f.toFixed(f < 1 ? 2 : 1);
    if (f === flatnessVal) o.selected = true;
    flatSelect.appendChild(o);
  });
  flatSelect.addEventListener('change', function() {
    flatnessVal = parseFloat(this.value);
    render();
  });

  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const oldScale = viewScale;
    viewScale = Math.max(0.005, Math.min(50, viewScale * (e.deltaY < 0 ? 1.08 : 1 / 1.08)));
    viewX *= viewScale / oldScale; viewY *= viewScale / oldScale;
    render();
  });
  canvas.addEventListener('mousedown', e => {
    dragging = true; dragStartX = e.clientX; dragStartY = e.clientY;
    dragViewX = viewX; dragViewY = viewY;
  });
  canvas.addEventListener('mousemove', e => {
    mouseX = e.clientX; mouseY = e.clientY - TOPBAR_H;
    if (dragging) { viewX = dragViewX + (e.clientX - dragStartX); viewY = dragViewY + (e.clientY - dragStartY); }
    render();
  });
  canvas.addEventListener('mouseup', () => { dragging = false; });
  canvas.addEventListener('mouseleave', () => { dragging = false; });
  window.addEventListener('resize', resize);
  window.addEventListener('keydown', e => {
    if (e.key === 'o' || e.key === 'O') { showOSM = !showOSM; render(); }
    if (e.key === 's' || e.key === 'S') { showSamples = !showSamples; render(); }
    if (e.key === 'q' || e.key === 'Q') { goodOnly = !goodOnly; render(); }
    if (e.key === 'c' || e.key === 'C') { showClassify = !showClassify; render(); }
    if (e.key === '[' || e.key === '{') {
      const idx = FLATNESS_LIST.indexOf(flatnessVal);
      flatnessVal = idx > 0 ? FLATNESS_LIST[idx - 1] : FLATNESS_LIST[FLATNESS_LIST.length - 1];
      (document.getElementById('sel-flatness') as HTMLSelectElement).selectedIndex = FLATNESS_LIST.indexOf(flatnessVal);
      render();
    }
    if (e.key === ']' || e.key === '}') {
      const idx = FLATNESS_LIST.indexOf(flatnessVal);
      flatnessVal = idx < FLATNESS_LIST.length - 1 ? FLATNESS_LIST[idx + 1] : FLATNESS_LIST[0];
      (document.getElementById('sel-flatness') as HTMLSelectElement).selectedIndex = FLATNESS_LIST.indexOf(flatnessVal);
      render();
    }
    if (e.key === 'r' || e.key === 'R') { centerOn(currentWB); }
    if (e.key === 'f' || e.key === 'F') { fitAllBodies(); }
  });

  loadSamples().then(() => {
    if (sampleMeta) {
      const byLake: Record<string, number> = {};
      for (const s of sampleMeta) { byLake[s.k] = (byLake[s.k] || 0) + 1; }
      console.warn('Sample summary:', JSON.stringify(byLake));
    }
  });
  centerOn('西湖');
}
