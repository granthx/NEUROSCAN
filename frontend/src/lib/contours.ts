/**
 * Builds stacked contour line-segments from the phantom for the 3D reconstruction.
 * Each axial level is run through marching squares on a coarse grid, giving the
 * classic "slices stacked into a volume" wireframe look.
 */
import { brainE, lesionAt, LESION_GT } from "./phantom";

export type ContourSet = {
  positions: Float32Array; // xyz pairs (line segments) — three.js space: y = superior
  slice: Float32Array; // 0..1 per vertex: which level it came from (for assemble animation)
};

const EDGE_TABLE: number[][] = [
  [], [3, 0], [0, 1], [3, 1], [1, 2], [3, 0, 1, 2], [0, 2], [3, 2],
  [2, 3], [2, 0], [0, 1, 2, 3], [2, 1], [1, 3], [1, 0], [0, 3], [],
];

/**
 * @param field  returns a scalar; the iso-contour is drawn at 0.5 (inside > 0.5)
 */
function marching(field: (x: number, y: number) => number, n: number, level: number, z: number, out: number[], sl: number[], s: number) {
  const g = new Float32Array((n + 1) * (n + 1));
  for (let j = 0; j <= n; j++)
    for (let i = 0; i <= n; i++) g[j * (n + 1) + i] = field((i / n) * 2 - 1, (j / n) * 2 - 1);
  const lerp = (a: number, b: number) => (level - a) / (b - a || 1e-6);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const a = g[j * (n + 1) + i], b = g[j * (n + 1) + i + 1];
      const c = g[(j + 1) * (n + 1) + i + 1], d = g[(j + 1) * (n + 1) + i];
      const idx = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0);
      const edges = EDGE_TABLE[idx];
      if (!edges.length) continue;
      const x0 = (i / n) * 2 - 1, y0 = (j / n) * 2 - 1, h = 2 / n;
      const pt = (e: number): [number, number] => {
        if (e === 0) return [x0 + lerp(a, b) * h, y0];
        if (e === 1) return [x0 + h, y0 + lerp(b, c) * h];
        if (e === 2) return [x0 + (1 - lerp(d, c)) * h, y0 + h];
        return [x0, y0 + (1 - lerp(a, d)) * h];
      };
      for (let k = 0; k < edges.length; k += 2) {
        const p = pt(edges[k]), q = pt(edges[k + 1]);
        // phantom (x, y, z) → three (x, z, -y): y-up = superior
        out.push(p[0], z, -p[1], q[0], z, -q[1]);
        sl.push(s, s);
      }
    }
}

export function buildBrainContours(levels = 34, n = 72): ContourSet {
  const out: number[] = [], sl: number[] = [];
  for (let l = 0; l < levels; l++) {
    const s = l / (levels - 1);
    const z = -0.62 + s * 1.3;
    marching((x, y) => 1.5 - brainE(x, y, z), n, 0.5, z, out, sl, s);
  }
  return { positions: new Float32Array(out), slice: new Float32Array(sl) };
}

/** which: "wt" (any label) or "et" (enhancing) */
export function buildTumorContours(which: "wt" | "et", levels = 40, n = 90): ContourSet {
  const out: number[] = [], sl: number[] = [];
  const [cx, cy, cz] = LESION_GT.c;
  for (let l = 0; l < levels; l++) {
    const s = l / (levels - 1);
    const z = cz - 0.3 + s * 0.6;
    // sample only the lesion neighbourhood, remapped to the full [-1,1] grid for resolution
    const R = 0.34;
    marching(
      (u, v) => {
        const x = cx + u * R, y = cy + v * R;
        const lab = lesionAt(x, y, z, LESION_GT);
        return which === "wt" ? (lab ? 1 : 0) : lab === 3 ? 1 : 0;
      },
      n,
      0.5,
      z,
      out,
      sl,
      s,
    );
    // un-remap x/y for the vertices of this level
  }
  // Rescale the remapped coords back into world space
  for (let i = 0; i < out.length; i += 3) {
    out[i] = cx + out[i] * 0.34;
    out[i + 2] = -cy + out[i + 2] * 0.34;
  }
  return { positions: new Float32Array(out), slice: new Float32Array(sl) };
}

