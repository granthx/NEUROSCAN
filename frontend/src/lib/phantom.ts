/**
 * Procedural MRI "phantom".
 * ---------------------------------------------------------------------------
 * PLACEHOLDER DATA. This generates a skull-stripped, BraTS-like brain volume
 * analytically (no files needed) with a glioma-like lesion made of three
 * sub-regions: necrotic core (1), peritumoral edema (2) and enhancing tumor (3).
 * Two versions of the lesion exist: ground truth ("gt") and a slightly
 * perturbed "pred" that stands in for a model prediction, so every readout
 * (Dice, IoU, HD95, volume) is genuinely computed from masks.
 *
 * SWAPPING IN REAL DATA: replace `getSlice()` with a loader that returns the
 * same `SliceData` shape from real volumes, e.g. pre-exported PNG/NPY slices
 * of a BraTS case and the model's predicted mask (see README → "Real data").
 *
 * Coordinates: x = left→right, y = posterior→anterior, z = inferior→superior,
 * all normalised to [-1, 1] over a 240 mm field of view.
 * ------------------------------------------------------------------------- */

export type Plane = "axial" | "coronal" | "sagittal";
export type Modality = "t1" | "t1ce" | "t2" | "flair";

export const PLANE_DIMS: Record<Plane, number> = { axial: 155, coronal: 240, sagittal: 240 };
export const MODALITIES: Modality[] = ["t1", "t1ce", "t2", "flair"];
export const MODALITY_LABEL: Record<Modality, string> = {
  t1: "T1",
  t1ce: "T1ce",
  t2: "T2",
  flair: "FLAIR",
};

export const RES = 160; // pixels per slice edge
export const FOV_MM = 240;
export const MM_PER_PX = FOV_MM / RES;

/* ----------------------------- noise ----------------------------------- */
function hash(ix: number, iy: number, iz: number): number {
  let h = Math.imul(ix, 374761393) ^ Math.imul(iy, 668265263) ^ Math.imul(iz, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967295;
}
const smooth = (t: number) => t * t * (3 - 2 * t);
function vnoise(x: number, y: number, z: number): number {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = smooth(x - ix), fy = smooth(y - iy), fz = smooth(z - iz);
  const l = (a: number, b: number, t: number) => a + (b - a) * t;
  const c000 = hash(ix, iy, iz), c100 = hash(ix + 1, iy, iz);
  const c010 = hash(ix, iy + 1, iz), c110 = hash(ix + 1, iy + 1, iz);
  const c001 = hash(ix, iy, iz + 1), c101 = hash(ix + 1, iy, iz + 1);
  const c011 = hash(ix, iy + 1, iz + 1), c111 = hash(ix + 1, iy + 1, iz + 1);
  return l(
    l(l(c000, c100, fx), l(c010, c110, fx), fy),
    l(l(c001, c101, fx), l(c011, c111, fx), fy),
    fz,
  );
}
function fbm(x: number, y: number, z: number): number {
  return 0.55 * vnoise(x, y, z) + 0.3 * vnoise(x * 2.03, y * 2.03, z * 2.03) + 0.15 * vnoise(x * 4.1, y * 4.1, z * 4.1);
}

/* ----------------------------- anatomy --------------------------------- */
const BRAIN_R = [0.7, 0.86, 0.68];
const BRAIN_Z = 0.04;

/** Ellipsoid "radius" of the brain at a point; <= 1 is inside. */
export function brainE(x: number, y: number, z: number): number {
  const zz = z - BRAIN_Z;
  const taper = zz < -0.15 ? 1 + (zz + 0.15) * 0.7 : 1; // narrower towards the skull base
  const e = (x / (BRAIN_R[0] * taper)) ** 2 + (y / BRAIN_R[1]) ** 2 + (zz / BRAIN_R[2]) ** 2;
  return e + 0.07 * (fbm(x * 5 + 7, y * 5, z * 5) - 0.5);
}

/** 0 air · 1 CSF · 2 grey matter · 3 white matter */
function tissueAt(x: number, y: number, z: number, e: number): number {
  if (e > 1) return 0;
  const zz = z - BRAIN_Z;
  if (Math.abs(x) < 0.018 && zz > -0.1 && e > 0.2) return 1; // interhemispheric fissure
  for (const s of [-1, 1]) {
    const bend = 0.05 * Math.sin(y * 4);
    const v = ((x - s * 0.11) / 0.06) ** 2 + ((y - 0.02) / 0.25) ** 2 + ((zz - 0.08 - bend) / 0.1) ** 2;
    if (v < 1) return 1; // lateral ventricles
  }
  if (e > 0.7) {
    const g = fbm(x * 9 + 3, y * 9, z * 9);
    return g > 0.63 ? 1 : 2; // sulci / cortex
  }
  const nuc = ((Math.abs(x) - 0.19) / 0.08) ** 2 + (y / 0.12) ** 2 + ((zz + 0.02) / 0.1) ** 2;
  if (nuc < 1) return 2; // deep grey nuclei
  return 3;
}

/* ----------------------------- lesion ---------------------------------- */
type Lesion = { c: [number, number, number]; rN: number; rE: number; rD: number; seed: number };
export const LESION_GT: Lesion = { c: [0.33, -0.12, 0.18], rN: 0.075, rE: 0.125, rD: 0.245, seed: 0 };
export const LESION_PRED: Lesion = { c: [0.342, -0.131, 0.188], rN: 0.07, rE: 0.131, rD: 0.236, seed: 0.18 };

/** 0 none · 1 necrotic (NCR) · 2 edema (ED) · 3 enhancing (ET) — BraTS convention after 4→3 remap */
export function lesionAt(x: number, y: number, z: number, L: Lesion): number {
  const dx = x - L.c[0], dy = y - L.c[1], dz = z - L.c[2];
  const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (d > L.rD * 1.6) return 0;
  const n1 = fbm(x * 4 + L.seed, y * 4, z * 4) - 0.5;
  const dc = d * (1 + 0.5 * n1);
  if (dc < L.rN) return 1;
  if (dc < L.rE) return 3;
  const n2 = fbm(x * 3 + 11 + L.seed, y * 3, z * 3) - 0.5;
  if (d * (1 + 0.8 * n2) < L.rD) return 2;
  return 0;
}

/* ----------------------------- intensities ----------------------------- */
// columns: CSF, GM, WM, NCR, ED, ET
const CONTRAST: Record<Modality, [number, number, number, number, number, number]> = {
  t1: [0.12, 0.5, 0.74, 0.22, 0.42, 0.5],
  t1ce: [0.12, 0.52, 0.72, 0.18, 0.44, 0.98],
  t2: [0.95, 0.58, 0.34, 0.9, 0.82, 0.62],
  flair: [0.1, 0.55, 0.42, 0.4, 0.94, 0.72],
};

/* ----------------------------- slicing --------------------------------- */
export type SliceData = {
  plane: Plane;
  index: number;
  tissue: Uint8Array; // RES*RES
  grain: Float32Array; // per-pixel noise
  gt: Uint8Array; // labels 0..3
  pred: Uint8Array;
};

/** Map a pixel (u,v) on a plane at slice index to volume coords. v=0 is the top row. */
export function pixelToWorld(plane: Plane, index: number, u: number, v: number): [number, number, number] {
  const a = (u + 0.5) / RES * 2 - 1;
  const b = 1 - (v + 0.5) / RES * 2;
  const s = (index / (PLANE_DIMS[plane] - 1)) * 2 - 1;
  if (plane === "axial") return [a, b, s * 0.82]; // 155 slices span the brain's S-I extent
  if (plane === "coronal") return [a, s, b];
  return [s, a, b]; // sagittal: anterior to the right, superior up
}

const cache = new Map<string, SliceData>();

export function getSlice(plane: Plane, index: number): SliceData {
  const key = `${plane}:${index}`;
  const hit = cache.get(key);
  if (hit) return hit;

  const n = RES * RES;
  const tissue = new Uint8Array(n);
  const grain = new Float32Array(n);
  const gt = new Uint8Array(n);
  const pred = new Uint8Array(n);

  for (let v = 0; v < RES; v++) {
    for (let u = 0; u < RES; u++) {
      const i = v * RES + u;
      const [x, y, z] = pixelToWorld(plane, index, u, v);
      const e = brainE(x, y, z);
      const t = tissueAt(x, y, z, e);
      tissue[i] = t;
      if (t === 0) continue;
      grain[i] = vnoise(x * 60, y * 60, z * 60) - 0.5;
      gt[i] = lesionAt(x, y, z, LESION_GT);
      pred[i] = lesionAt(x, y, z, LESION_PRED);
    }
  }
  const data = { plane, index, tissue, grain, gt, pred };
  cache.set(key, data);
  if (cache.size > 120) cache.delete(cache.keys().next().value as string);
  return data;
}

/** Grey value 0..1 for every pixel for the chosen modality. */
export function intensities(s: SliceData, modality: Modality): Float32Array {
  const c = CONTRAST[modality];
  const out = new Float32Array(s.tissue.length);
  for (let i = 0; i < out.length; i++) {
    const t = s.tissue[i];
    if (t === 0) continue;
    const lab = s.gt[i];
    let val = lab === 1 ? c[3] : lab === 2 ? c[4] : lab === 3 ? c[5] : c[t - 1];
    val += s.grain[i] * 0.14;
    out[i] = Math.min(1, Math.max(0, val));
  }
  return out;
}

/* ----------------------------- metrics --------------------------------- */
export type SliceMetrics = { dice: number | null; iou: number | null; hd95: number | null; areaMm2: number; et: boolean };

/** Whole-tumour (labels 1|2|3) agreement between pred and gt on one slice. */
export function sliceMetrics(s: SliceData): SliceMetrics {
  let tp = 0, fp = 0, fn = 0, et = false, area = 0;
  for (let i = 0; i < s.gt.length; i++) {
    const g = s.gt[i] > 0, p = s.pred[i] > 0;
    if (g && p) tp++;
    else if (p) fp++;
    else if (g) fn++;
    if (p) area++;
    if (s.pred[i] === 3) et = true;
  }
  const areaMm2 = area * MM_PER_PX * MM_PER_PX;
  if (tp + fp + fn === 0) return { dice: null, iou: null, hd95: null, areaMm2, et };
  const dice = (2 * tp) / (2 * tp + fp + fn);
  const iou = tp / (tp + fp + fn);
  return { dice, iou, hd95: hd95(s.gt, s.pred), areaMm2, et };
}

function boundary(mask: Uint8Array): number[] {
  const pts: number[] = [];
  for (let v = 0; v < RES; v++)
    for (let u = 0; u < RES; u++) {
      const i = v * RES + u;
      if (!mask[i]) continue;
      if (u === 0 || v === 0 || u === RES - 1 || v === RES - 1 || !mask[i - 1] || !mask[i + 1] || !mask[i - RES] || !mask[i + RES])
        pts.push(u, v);
    }
  return pts;
}

function hd95(gt: Uint8Array, pred: Uint8Array): number | null {
  const bg = boundary(gt), bp = boundary(pred);
  if (!bg.length || !bp.length) return null;
  const dists: number[] = [];
  const nearest = (a: number[], b: number[]) => {
    for (let i = 0; i < a.length; i += 2) {
      let best = Infinity;
      for (let j = 0; j < b.length; j += 2) {
        const dx = a[i] - b[j], dy = a[i + 1] - b[j + 1];
        const d = dx * dx + dy * dy;
        if (d < best) best = d;
      }
      dists.push(Math.sqrt(best) * MM_PER_PX);
    }
  };
  nearest(bg, bp);
  nearest(bp, bg);
  dists.sort((p, q) => p - q);
  return dists[Math.min(dists.length - 1, Math.floor(dists.length * 0.95))];
}

/* ----------------------------- volumes --------------------------------- */
export type Volumes = { wt: number; tc: number; et: number; ed: number; ncr: number };
let volCache: { gt: Volumes; pred: Volumes; dice3d: number } | null = null;

/** Whole-volume sub-region volumes in cm³ (grid sampled at n³). */
export function computeVolumes(n = 56) {
  if (volCache) return volCache;
  const voxMm = FOV_MM / n;
  const voxCm3 = (voxMm * voxMm * voxMm) / 1000;
  const zero = (): Volumes => ({ wt: 0, tc: 0, et: 0, ed: 0, ncr: 0 });
  const g = zero(), p = zero();
  let tp = 0, sg = 0, sp = 0;
  for (let k = 0; k < n; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const x = ((i + 0.5) / n) * 2 - 1, y = ((j + 0.5) / n) * 2 - 1, z = ((k + 0.5) / n) * 2 - 1;
        const dx = x - LESION_GT.c[0], dy = y - LESION_GT.c[1], dz = z - LESION_GT.c[2];
        if (dx * dx + dy * dy + dz * dz > 0.2) continue; // skip far voxels
        if (brainE(x, y, z) > 1) continue;
        const a = lesionAt(x, y, z, LESION_GT), b = lesionAt(x, y, z, LESION_PRED);
        const add = (v: Volumes, l: number) => {
          if (!l) return;
          v.wt += voxCm3;
          if (l === 1) { v.ncr += voxCm3; v.tc += voxCm3; }
          if (l === 3) { v.et += voxCm3; v.tc += voxCm3; }
          if (l === 2) v.ed += voxCm3;
        };
        add(g, a);
        add(p, b);
        if (a) sg++;
        if (b) sp++;
        if (a && b) tp++;
      }
  volCache = { gt: g, pred: p, dice3d: (2 * tp) / Math.max(1, sg + sp) };
  return volCache;
}
