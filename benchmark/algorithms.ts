/** Lake detection algorithms — TypeScript port of scripts/lake_algorithms.py.
 *
 * Pure functions matching the GLSL shader behavior. Used for real-time
 * profile detail chart lakeMask computation in the browser.
 */

// ============================================================================
// GLSL smoothstep
// ============================================================================
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

// ============================================================================
// Catmull-Rom cubic weight
// ============================================================================
function cubicCR(x: number): number {
  const ax = Math.abs(x);
  if (ax < 1.0) return (1.5 * ax - 2.5) * ax * ax + 1.0;
  if (ax < 2.0) return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0;
  return 0.0;
}

// ============================================================================
// 1D filtered sampling (Catmull-Rom 4-tap)
// ============================================================================
function sampleFiltered1D(arr: Float64Array, uvContinuous: number): number {
  const n = arr.length;
  const center = uvContinuous;
  const frac = center - Math.floor(center);

  let result = 0.0;
  let totalWeight = 0.0;
  for (let i = -1; i <= 2; i++) {
    const w = cubicCR(i - frac);
    const idx = Math.max(0, Math.min(n - 1, Math.floor(uvContinuous + i)));
    result += arr[idx] * w;
    totalWeight += w;
  }
  if (totalWeight > 0.001) return result / totalWeight;
  return arr[Math.max(0, Math.min(n - 1, Math.floor(uvContinuous)))];
}

// ============================================================================
// Gradient operators (forward diff = G2, the champion)
// ============================================================================
function gradForwardDiff(arr: Float64Array, i: number, step: number): number {
  const hC = sampleFiltered1D(arr, i);
  const hR = sampleFiltered1D(arr, i + step);
  return (hR - hC) / step;
}

// ============================================================================
// Compute lake mask — champion algorithm (G2 forward diff + M4 squared)
// ============================================================================
export interface LakeParams {
  flatness?: number;
  lake_range?: number;
  texture_size?: number;
  grad_op?: string;
  multi_scale?: string;
  decision_fn?: string;
  decision_k?: number;
  pre_smooth?: string;
}

export function computeLakeMask(arr: Float64Array, params: LakeParams = {}): Float64Array {
  const n = arr.length;
  const flatness = params.flatness ?? 500;
  const lakeRange = params.lake_range ?? 1;
  const textureSize = params.texture_size ?? 256;
  const scale = n / textureSize;
  const scaleFactor = textureSize;
  const step = lakeRange * scale;

  const mask = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    // G2 forward difference
    const g = gradForwardDiff(arr, i, step);
    const steepness = Math.abs(g) * scaleFactor;

    // M4 squared gradient
    const steepnessSq = steepness * steepness;

    // Decision: smoothstep with lo=0.3*flatness
    const lo = flatness * 0.3;
    const hi = flatness;
    mask[i] = 1.0 - smoothstep(lo, hi, steepnessSq);
  }

  return mask;
}

/**
 * Compute lake mask with arbitrary algorithm config.
 * Supports G1-G6 gradient operators, M1-M4 multi-scale, D1-D3 decision.
 */
export function computeLakeMaskConfig(
  arr: Float64Array,
  flatness: number,
  lakeRange: number,
  textureSize: number,
  gradOp: string,
  multiScale: string,
  decisionFn: string,
  decisionK: number,
): Float64Array {
  const n = arr.length;
  const scale = n / textureSize;
  const scaleFactor = textureSize;
  const mask = new Float64Array(n);

  const ranges: number[] = multiScale === "M2"
    ? [lakeRange] : (multiScale === "M3" || multiScale === "M4")
    ? [1.0, lakeRange] : [1.0];

  // Compute per-scale steepnesses
  const scaleSteepnesses: Float64Array[] = [];
  for (const r of ranges) {
    const step = r * scale;
    const steepnessArr = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      let g: number;
      if (gradOp === "G2") {
        g = (sampleFiltered1D(arr, i + step) - sampleFiltered1D(arr, i)) / step;
      } else {
        // G1, G3, G5, G6: central-diff family
        const hR = sampleFiltered1D(arr, i + step);
        const hL = sampleFiltered1D(arr, i - step);
        g = (hR - hL) / (2 * step);
      }
      steepnessArr[i] = Math.abs(g) * scaleFactor;
    }
    scaleSteepnesses.push(steepnessArr);
  }

  // Combine scales
  let combined: Float64Array;
  if (multiScale === "M3") {
    combined = new Float64Array(n);
    for (let i = 0; i < n; i++) combined[i] = Math.min(scaleSteepnesses[0][i], scaleSteepnesses[1][i]);
  } else if (multiScale === "M4") {
    combined = new Float64Array(n);
    for (let i = 0; i < n; i++) combined[i] = scaleSteepnesses[0][i] * scaleSteepnesses[1][i];
  } else {
    combined = scaleSteepnesses[0];
  }

  // Decision
  for (let i = 0; i < n; i++) {
    const s = combined[i];
    if (decisionFn === "D2") {
      mask[i] = s < flatness ? 1.0 : 0.0;
    } else {
      const lo = flatness * decisionK;
      const hi = flatness;
      mask[i] = 1.0 - smoothstep(lo, hi, s);
    }
  }

  return mask;
}
