"use client";
/**
 * HERO 3D SCENE — RETRO TOMOGRAPHIC SCANNER TERMINAL
 * 
 * - Camera perspective stays LOCKED ON THE PERSON at all times.
 * - The scanner ring sweeps through the body from feet to head.
 * - As you scroll, the camera glides straight ("seetha") directly onto the 3D brain.
 * - The brain zooms in dead-center in the clear viewport, with no text overlap
 *   and no blown-out white glare.
 * - True neuroanatomy: dual cerebral hemispheres, distinct lobes, gyral crests,
 *   striate cerebellum, brainstem, and 3-layer glowing glioma.
 * - Stationary by default; left-click drag anywhere to rotate.
 */
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { buildBrainContours, buildTumorContours } from "@/lib/contours";
import { frameState } from "@/lib/store";
import { CRTPass, srgb } from "./CRTPass";

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const smooth = (t: number) => t * t * (3 - 2 * t);
const TWO_PI = Math.PI * 2;

// Center of the cranial vault in model space
export const HEAD_CENTER = new THREE.Vector3(1.36, 0.28, 0);
// Right-parietal glioma lesion center
export const TUMOR_CENTER = new THREE.Vector3(1.39, 0.35, 0.065);

type AnatStation = {
  x: number;
  yBack: number;
  yFront: number;
  zHalf: number;
  zOffset?: number;
  pinch?: number;
};

/**
 * Builds an anatomically lofted mesh along X with superellipse cross-sections.
 */
function buildAnatomicalMesh(
  stations: AnatStation[],
  radialSegs = 36,
  mirrorZ = false,
): THREE.BufferGeometry {
  const rings = stations.length;
  const positions: number[] = [];
  const indices: number[] = [];

  for (let i = 0; i < rings; i++) {
    const { x, yBack, yFront, zHalf, zOffset = 0, pinch = 0 } = stations[i];
    const yc = (yBack + yFront) * 0.5;
    const ryTop = (yFront - yBack) * 0.5;
    const ryBot = (yFront - yBack) * 0.5;

    for (let j = 0; j <= radialSegs; j++) {
      const a = (j / radialSegs) * TWO_PI;
      const ca = Math.cos(a);
      const sa = Math.sin(a);

      const midlineDip = pinch * Math.exp(-ca * ca * 14) * (sa > 0 ? sa : -0.45 * sa);
      const ry = (sa >= 0 ? ryTop : ryBot) * (1 - midlineDip);

      const signZ = Math.sign(ca);
      const signY = Math.sign(sa);
      const zLocal = signZ * Math.pow(Math.abs(ca), 0.88) * zHalf;
      const yLocal = signY * Math.pow(Math.abs(sa), 0.92) * ry;

      const zCenter = zOffset * (mirrorZ ? -1 : 1);
      positions.push(x, yc + yLocal, zCenter + zLocal);
    }
  }

  const stride = radialSegs + 1;
  for (let i = 0; i < rings - 1; i++) {
    for (let j = 0; j < radialSegs; j++) {
      const a = i * stride + j;
      const b = (i + 1) * stride + j;
      const c = (i + 1) * stride + (j + 1);
      const d = i * stride + (j + 1);
      indices.push(a, b, d, b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Head, Neck, and Torso with distinct facial structure:
 * brow, nose, lips, chin, jawline, ears, throat, pectorals, ribcage, waist, hips.
 */
function buildSculptedHeadAndTorso(): THREE.BufferGeometry {
  const stations: AnatStation[] = [
    { x: -0.04, yBack: 0.11, yFront: 0.25, zHalf: 0.14, pinch: 0.22 },
    { x: 0.08, yBack: 0.085, yFront: 0.29, zHalf: 0.22, pinch: 0.18 },
    { x: 0.18, yBack: 0.09, yFront: 0.30, zHalf: 0.23, pinch: 0.12 },
    { x: 0.30, yBack: 0.11, yFront: 0.285, zHalf: 0.19, pinch: 0.1 },
    { x: 0.42, yBack: 0.12, yFront: 0.295, zHalf: 0.185, pinch: 0.12 },
    { x: 0.56, yBack: 0.105, yFront: 0.325, zHalf: 0.21, pinch: 0.15 },
    { x: 0.70, yBack: 0.095, yFront: 0.355, zHalf: 0.235, pinch: 0.20 },
    { x: 0.84, yBack: 0.09, yFront: 0.37, zHalf: 0.255, pinch: 0.22 },
    { x: 0.96, yBack: 0.10, yFront: 0.345, zHalf: 0.275, pinch: 0.12 },
    { x: 1.05, yBack: 0.12, yFront: 0.30, zHalf: 0.17, pinch: 0.04 },
    { x: 1.11, yBack: 0.145, yFront: 0.285, zHalf: 0.082 },
    { x: 1.17, yBack: 0.15, yFront: 0.295, zHalf: 0.076 },
    { x: 1.21, yBack: 0.14, yFront: 0.37, zHalf: 0.098 },
    { x: 1.25, yBack: 0.12, yFront: 0.41, zHalf: 0.115 },
    { x: 1.28, yBack: 0.11, yFront: 0.42, zHalf: 0.125 },
    { x: 1.32, yBack: 0.095, yFront: 0.455, zHalf: 0.142 },
    { x: 1.36, yBack: 0.09, yFront: 0.42, zHalf: 0.154 },
    { x: 1.40, yBack: 0.095, yFront: 0.435, zHalf: 0.156 },
    { x: 1.45, yBack: 0.11, yFront: 0.415, zHalf: 0.148 },
    { x: 1.50, yBack: 0.14, yFront: 0.38, zHalf: 0.125 },
    { x: 1.53, yBack: 0.19, yFront: 0.33, zHalf: 0.08 },
    { x: 1.55, yBack: 0.25, yFront: 0.27, zHalf: 0.02 },
  ];
  return buildAnatomicalMesh(stations, 40, false);
}

/**
 * Detailed leg with thigh, kneecap, calf, ankle, and foot with 5 toes.
 */
function buildSculptedLeg(mirror: boolean): THREE.BufferGeometry {
  const stations: AnatStation[] = [
    { x: -1.53, yBack: 0.27, yFront: 0.375, zHalf: 0.046, zOffset: 0.115 },
    { x: -1.47, yBack: 0.16, yFront: 0.36, zHalf: 0.052, zOffset: 0.115 },
    { x: -1.39, yBack: 0.075, yFront: 0.26, zHalf: 0.046, zOffset: 0.115 },
    { x: -1.31, yBack: 0.10, yFront: 0.22, zHalf: 0.044, zOffset: 0.115 },
    { x: -1.16, yBack: 0.095, yFront: 0.23, zHalf: 0.056, zOffset: 0.118 },
    { x: -0.94, yBack: 0.075, yFront: 0.255, zHalf: 0.082, zOffset: 0.122 },
    { x: -0.74, yBack: 0.095, yFront: 0.26, zHalf: 0.076, zOffset: 0.124 },
    { x: -0.60, yBack: 0.11, yFront: 0.28, zHalf: 0.074, zOffset: 0.125 },
    { x: -0.44, yBack: 0.095, yFront: 0.285, zHalf: 0.090, zOffset: 0.126 },
    { x: -0.24, yBack: 0.08, yFront: 0.30, zHalf: 0.110, zOffset: 0.125 },
    { x: -0.06, yBack: 0.08, yFront: 0.305, zHalf: 0.118, zOffset: 0.118 },
    { x: 0.08, yBack: 0.09, yFront: 0.29, zHalf: 0.105, zOffset: 0.105 },
  ];
  return buildAnatomicalMesh(stations, 24, mirror);
}

/**
 * Arm from shoulder to wrist.
 */
function buildSculptedArm(mirror: boolean): THREE.BufferGeometry {
  const stations: AnatStation[] = [
    { x: 0.22, yBack: 0.13, yFront: 0.19, zHalf: 0.034, zOffset: 0.32 },
    { x: 0.36, yBack: 0.125, yFront: 0.205, zHalf: 0.046, zOffset: 0.33 },
    { x: 0.50, yBack: 0.12, yFront: 0.22, zHalf: 0.054, zOffset: 0.335 },
    { x: 0.62, yBack: 0.11, yFront: 0.22, zHalf: 0.052, zOffset: 0.335 },
    { x: 0.78, yBack: 0.115, yFront: 0.245, zHalf: 0.062, zOffset: 0.325 },
    { x: 0.94, yBack: 0.12, yFront: 0.285, zHalf: 0.075, zOffset: 0.30 },
    { x: 1.02, yBack: 0.14, yFront: 0.29, zHalf: 0.06, zOffset: 0.25 },
  ];
  return buildAnatomicalMesh(stations, 20, mirror);
}

/**
 * High-realism articulated human hand resting beside the hips.
 */
function buildArticulatedHand(mirror: boolean): { mesh: THREE.BufferGeometry; wire: THREE.BufferGeometry } {
  const pts: number[] = [];
  const s = mirror ? -1 : 1;
  const zBase = 0.325 * s;
  const yBase = 0.165;
  const xWrist = 0.22;

  const pWristMed = [xWrist, yBase - 0.015, zBase - 0.024 * s];
  const pWristLat = [xWrist, yBase + 0.015, zBase + 0.026 * s];
  const pThenar   = [0.15, yBase + 0.018, zBase + 0.046 * s];
  const pMcp2     = [0.12, yBase + 0.014, zBase + 0.028 * s];
  const pMcp3     = [0.115, yBase + 0.011, zBase + 0.009 * s];
  const pMcp4     = [0.118, yBase + 0.008, zBase - 0.009 * s];
  const pMcp5     = [0.122, yBase + 0.005, zBase - 0.026 * s];

  const palmVerts: number[] = [];
  const addTri = (a: number[], b: number[], c: number[]) => {
    palmVerts.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  };
  addTri(pWristMed, pWristLat, pThenar);
  addTri(pWristMed, pThenar, pMcp2);
  addTri(pWristMed, pMcp2, pMcp3);
  addTri(pWristMed, pMcp3, pMcp4);
  addTri(pWristMed, pMcp4, pMcp5);

  const palmOutline = [pWristMed, pWristLat, pThenar, pMcp2, pMcp3, pMcp4, pMcp5, pWristMed];
  for (let i = 0; i < palmOutline.length - 1; i++) {
    pts.push(
      palmOutline[i][0], palmOutline[i][1], palmOutline[i][2],
      palmOutline[i + 1][0], palmOutline[i + 1][1], palmOutline[i + 1][2]
    );
  }
  pts.push(pWristMed[0], pWristMed[1], pWristMed[2], pMcp3[0], pMcp3[1], pMcp3[2]);
  pts.push(pThenar[0], pThenar[1], pThenar[2], pMcp5[0], pMcp5[1], pMcp5[2]);

  const makePhalanx = (p0: number[], p1: number[], r0: number, r1: number, segs = 6) => {
    const ring0: number[][] = [];
    const ring1: number[][] = [];
    for (let k = 0; k < segs; k++) {
      const a = (k / segs) * TWO_PI;
      const uy = Math.sin(a);
      const uz = Math.cos(a);
      ring0.push([p0[0], p0[1] + uy * r0, p0[2] + uz * r0]);
      ring1.push([p1[0], p1[1] + uy * r1, p1[2] + uz * r1]);
    }
    for (let k = 0; k < segs; k++) {
      const nxt = (k + 1) % segs;
      addTri(ring0[k], ring1[k], ring1[nxt]);
      addTri(ring0[k], ring1[nxt], ring0[nxt]);
      pts.push(ring0[k][0], ring0[k][1], ring0[k][2], ring1[k][0], ring1[k][1], ring1[k][2]);
      pts.push(ring0[k][0], ring0[k][1], ring0[k][2], ring0[nxt][0], ring0[nxt][1], ring0[nxt][2]);
    }
  };

  const buildFinger = (root: number[], dirX: number, dirY: number, dirZ: number, segLengths: number[], baseRadius: number) => {
    let cur = [...root];
    for (let seg = 0; seg < segLengths.length; seg++) {
      const len = segLengths[seg];
      const r0 = baseRadius * (1 - seg * 0.16);
      const r1 = baseRadius * (1 - (seg + 1) * 0.16);
      const curl = seg * 0.08;
      const nxt = [cur[0] - len * dirX, cur[1] - len * (dirY + curl), cur[2] + len * dirZ * s];
      makePhalanx(cur, nxt, r0, r1);
      cur = nxt;
    }
  };

  buildFinger(pThenar, 0.68, 0.20, 0.72, [0.038, 0.032], 0.013);
  buildFinger(pMcp2, 0.95, 0.12, 0.18, [0.038, 0.028, 0.024], 0.011);
  buildFinger(pMcp3, 0.98, 0.10, 0.03, [0.044, 0.032, 0.026], 0.0115);
  buildFinger(pMcp4, 0.96, 0.12, -0.12, [0.040, 0.028, 0.023], 0.0105);
  buildFinger(pMcp5, 0.91, 0.15, -0.26, [0.030, 0.022, 0.019], 0.0095);

  const meshGeo = new THREE.BufferGeometry();
  meshGeo.setAttribute("position", new THREE.Float32BufferAttribute(palmVerts, 3));
  meshGeo.computeVertexNormals();

  const wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));

  return { mesh: meshGeo, wire: wireGeo };
}

/**
 * Internal medical skeleton.
 */
function buildDetailedSkeleton(): THREE.BufferGeometry {
  const pts: number[] = [];
  const ring = (cx: number, cy: number, cz: number, rx: number, ry: number, rz: number, n = 16) => {
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * TWO_PI;
      const a1 = ((i + 1) / n) * TWO_PI;
      pts.push(
        cx + Math.cos(a0) * rx, cy + Math.sin(a0) * ry, cz + Math.cos(a0) * rz,
        cx + Math.cos(a1) * rx, cy + Math.sin(a1) * ry, cz + Math.cos(a1) * rz
      );
    }
  };

  for (const s of [-1, 1]) {
    ring(1.36, 0.37, s * 0.052, 0.022, 0.024, 0.0, 14);
  }
  ring(1.31, 0.38, 0, 0.02, 0.018, 0.015, 10);

  for (let i = 0; i < 14; i++) {
    const t0 = (i / 14) * Math.PI - Math.PI / 2;
    const t1 = ((i + 1) / 14) * Math.PI - Math.PI / 2;
    pts.push(
      1.26 - Math.cos(t0) * 0.04, 0.24 + Math.cos(t0) * 0.13, Math.sin(t0) * 0.105,
      1.26 - Math.cos(t1) * 0.04, 0.24 + Math.cos(t1) * 0.13, Math.sin(t1) * 0.105
    );
  }

  for (let v = 0; v < 24; v++) {
    const f = v / 23;
    const x = 0.14 + f * 1.04;
    const y = 0.14 + Math.sin(f * Math.PI * 2) * 0.018;
    const r = 0.026 * (1 - f * 0.35);
    for (let j = 0; j < 8; j++) {
      const a0 = (j / 8) * TWO_PI;
      const a1 = ((j + 1) / 8) * TWO_PI;
      pts.push(
        x, y + Math.sin(a0) * r, Math.cos(a0) * r,
        x, y + Math.sin(a1) * r, Math.cos(a1) * r
      );
    }
  }

  pts.push(0.62, 0.335, 0, 0.96, 0.33, 0);
  for (let r = 0; r < 11; r++) {
    const f = r / 10;
    const x = 0.52 + f * 0.43;
    const w = (0.195 - Math.abs(f - 0.45) * 0.08) * (r < 2 ? 0.85 : 1);
    const yTop = 0.30 + Math.sin(f * Math.PI) * 0.04;
    for (const side of [-1, 1]) {
      const segs = 14;
      const arc = r < 2 ? 0.72 : 0.94;
      for (let s = 0; s < segs; s++) {
        const a0 = (s / segs) * Math.PI * arc;
        const a1 = ((s + 1) / segs) * Math.PI * arc;
        pts.push(
          x - Math.sin(a0) * 0.035, 0.14 + (1 - Math.cos(a0)) * (yTop - 0.14) * 0.5, side * Math.sin(a0) * w,
          x - Math.sin(a1) * 0.035, 0.14 + (1 - Math.cos(a1)) * (yTop - 0.14) * 0.5, side * Math.sin(a1) * w
        );
      }
    }
  }

  for (const s of [-1, 1]) {
    pts.push(0.97, 0.32, s * 0.02, 0.96, 0.25, s * 0.28);
    for (let i = 0; i < 10; i++) {
      const a0 = (i / 10) * Math.PI;
      const a1 = ((i + 1) / 10) * Math.PI;
      pts.push(
        0.14 + Math.sin(a0) * 0.09, 0.16 + Math.sin(a0) * 0.09, s * (0.04 + Math.cos(a0) * 0.14 + 0.08),
        0.14 + Math.sin(a1) * 0.09, 0.16 + Math.sin(a1) * 0.09, s * (0.04 + Math.cos(a1) * 0.14 + 0.08)
      );
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

/**
 * Realistic Cerebral Hemisphere with gyri and sulci.
 */
function buildRealisticHemisphere(side: 1 | -1): THREE.BufferGeometry {
  const uSegs = 48;
  const vSegs = 36;
  const positions: number[] = [];
  const indices: number[] = [];

  for (let j = 0; j <= vSegs; j++) {
    const theta = (j / vSegs) * Math.PI;
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);

    for (let i = 0; i <= uSegs; i++) {
      const phi = (i / uSegs) * Math.PI * 2;
      const sinP = Math.sin(phi);
      const cosP = Math.cos(phi);

      let y = cosP * 0.115;
      let x = cosT * 0.084;
      let z = sinT * sinP * 0.090;

      if (side * z < 0) {
        z = side * 0.0035;
      } else {
        z = side * (0.005 + Math.abs(z));
      }

      if (y > 0.04) x += (y - 0.04) * 0.16;
      if (y < -0.04) {
        z *= 1.0 - (-y - 0.04) * 3.8;
        x *= 0.92;
      }
      if (y > -0.03 && y < 0.075 && x < -0.008 && Math.abs(z) > 0.035) {
        y += 0.026;
        x -= 0.018;
        z *= 1.15;
      }

      const sylvianDist = Math.abs(x + 0.005) + Math.abs(y - 0.02);
      if (sylvianDist < 0.038 && Math.abs(z) > 0.03) {
        const dip = (1 - sylvianDist / 0.038) * 0.014;
        z -= Math.sign(z) * dip;
      }

      const centralDist = Math.abs(y - (x * 0.65 - 0.01));
      if (centralDist < 0.025 && x > 0) {
        const dip = (1 - centralDist / 0.025) * 0.009;
        x -= dip;
      }

      const gyri =
        0.008 * Math.sin(x * 58 + y * 40) * Math.cos(z * 50) +
        0.005 * Math.sin(y * 74 - z * 60) +
        0.003 * Math.cos(x * 84 + z * 68);

      const r = Math.sqrt(x * x + y * y + z * z) || 1e-4;
      x += (x / r) * gyri;
      y += (y / r) * gyri;
      z += (z / r) * gyri;

      positions.push(x, y, z);
    }
  }

  const stride = uSegs + 1;
  for (let j = 0; j < vSegs; j++) {
    for (let i = 0; i < uSegs; i++) {
      const a = j * stride + i;
      const b = (j + 1) * stride + i;
      const c = (j + 1) * stride + (i + 1);
      const d = j * stride + (i + 1);
      indices.push(a, b, d, b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Cerebellum with horizontal folial ridges.
 */
function buildRealisticCerebellum(): THREE.BufferGeometry {
  const pts: number[] = [];
  const rings = 16;
  for (let i = 0; i < rings; i++) {
    const f = i / (rings - 1);
    const xRing = -0.038 - f * 0.055;
    const foliaR = Math.sin(f * Math.PI);
    const rY = 0.046 * foliaR;
    const rZ = 0.060 * foliaR;
    const cy = -0.068 - f * 0.015;

    for (const side of [-1, 1]) {
      const cz = side * 0.038;
      const segs = 18;
      for (let s = 0; s < segs; s++) {
        const a0 = (s / segs) * TWO_PI;
        const a1 = ((s + 1) / segs) * TWO_PI;
        const rip0 = 1 + 0.08 * Math.sin(s * 3.5);
        const rip1 = 1 + 0.08 * Math.sin((s + 1) * 3.5);
        pts.push(
          xRing, cy + Math.sin(a0) * rY * rip0, cz + Math.cos(a0) * rZ * rip0,
          xRing, cy + Math.sin(a1) * rY * rip1, cz + Math.cos(a1) * rZ * rip1
        );
      }
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

/**
 * Brainstem: Pons bulb + Medulla.
 */
function buildBrainstem(): THREE.BufferGeometry {
  const pts: number[] = [];
  const rings = 12;
  for (let i = 0; i < rings; i++) {
    const f = i / (rings - 1);
    const x = -0.04 - f * 0.09;
    const isPons = Math.exp(-Math.pow((f - 0.35) * 4.5, 2));
    const rY = 0.020 + isPons * 0.014;
    const rZ = 0.022 + isPons * 0.008;
    const yCenter = -0.035 + isPons * 0.012;

    const segs = 14;
    for (let s = 0; s < segs; s++) {
      const a0 = (s / segs) * TWO_PI;
      const a1 = ((s + 1) / segs) * TWO_PI;
      pts.push(
        x, yCenter + Math.sin(a0) * rY, Math.cos(a0) * rZ,
        x, yCenter + Math.sin(a1) * rY, Math.cos(a1) * rZ
      );
    }
  }
  for (let j = 0; j < 6; j++) {
    const a = (j / 6) * TWO_PI;
    pts.push(
      -0.04, -0.035 + Math.sin(a) * 0.020, Math.cos(a) * 0.022,
      -0.13, -0.035 + Math.sin(a) * 0.018, Math.cos(a) * 0.018
    );
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

function makeBodyShader(color: string, baseOpacity: number) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    wireframe: true,
    uniforms: {
      uColor: { value: srgb(color) },
      uScanX: { value: -1.7 },
      uOpacity: { value: baseOpacity },
      uHeadFade: { value: 0 },
    },
    vertexShader: /* glsl */ `
      uniform float uScanX;
      uniform float uHeadFade;
      varying float vGlow;
      varying float vPassed;
      varying float vCraniumMask;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vGlow = smoothstep(0.20, 0.0, abs(wp.x - uScanX));
        vPassed = smoothstep(wp.x - 0.08, wp.x + 0.08, uScanX);
        // Fade the cranium skull completely on zoom so the 3D brain inside is in full, unobstructed view
        float isCranium = smoothstep(1.10, 1.25, position.x);
        vCraniumMask = 1.0 - isCranium * uHeadFade * 0.98;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      varying float vGlow;
      varying float vPassed;
      varying float vCraniumMask;
      void main() {
        vec3 col = uColor * (0.50 + 0.50 * vPassed) + vec3(0.92, 1.0, 0.94) * vGlow * 1.15;
        float a = (uOpacity * (0.55 + 0.45 * vPassed) + vGlow * 0.65) * vCraniumMask;
        gl_FragColor = vec4(col, a);
      }
    `,
  });
}

function AnatomicalPatient() {
  const { torsoHeadGeo, legL, legR, armL, armR, handL, handR, skelGeo } = useMemo(() => {
    return {
      torsoHeadGeo: buildSculptedHeadAndTorso(),
      legL: buildSculptedLeg(false),
      legR: buildSculptedLeg(true),
      armL: buildSculptedArm(false),
      armR: buildSculptedArm(true),
      handL: buildArticulatedHand(false),
      handR: buildArticulatedHand(true),
      skelGeo: buildDetailedSkeleton(),
    };
  }, []);

  const bodyMat = useMemo(() => makeBodyShader("#7cff9b", 0.32), []);
  const handMeshMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: srgb("#4fae68"),
        wireframe: true,
        transparent: true,
        opacity: 0.60,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );
  const handWireMat = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: srgb("#7cff9b"),
        transparent: true,
        opacity: 0.88,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );
  const skelMat = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: srgb("#4fae68"),
        transparent: true,
        opacity: 0.42,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  useFrame(() => {
    const p = frameState.heroProgress;
    const sweepPhase = smooth(clamp01(p / 0.62));
    const scanX = -1.55 + sweepPhase * (HEAD_CENTER.x - -1.55);
    const zoomPhase = smooth(clamp01((p - 0.62) / 0.38));
    bodyMat.uniforms.uScanX.value = scanX;
    bodyMat.uniforms.uHeadFade.value = zoomPhase;
    skelMat.opacity = 0.42 * (1 - zoomPhase * 0.85);
  });

  return (
    <group>
      <mesh geometry={torsoHeadGeo} material={bodyMat} />
      <mesh geometry={legL} material={bodyMat} />
      <mesh geometry={legR} material={bodyMat} />
      <mesh geometry={armL} material={bodyMat} />
      <mesh geometry={armR} material={bodyMat} />

      {/* Articulated 3D Hands */}
      <group>
        <mesh geometry={handL.mesh} material={handMeshMat} />
        <lineSegments geometry={handL.wire} material={handWireMat} />
        <mesh geometry={handR.mesh} material={handMeshMat} />
        <lineSegments geometry={handR.wire} material={handWireMat} />
      </group>

      <lineSegments geometry={skelGeo} material={skelMat} />
    </group>
  );
}

/**
 * HIGH-DEFINITION NEUROANATOMICAL BRAIN & GLIOMA
 * Clearly visible inside the cranium, blooming into clean phosphor brilliance.
 */
function BrainAndTumor() {
  const brainGroup = useRef<THREE.Group>(null);
  const reticleRef = useRef<THREE.Group>(null);

  const { hemiL, hemiR, cerebellumGeo, brainstemGeo, brainSlices, wtSlices, etSlices } = useMemo(() => {
    const bc = buildBrainContours(28, 56);
    const bGeo = new THREE.BufferGeometry();
    bGeo.setAttribute("position", new THREE.BufferAttribute(bc.positions, 3));

    const wc = buildTumorContours("wt", 26, 60);
    const wGeo = new THREE.BufferGeometry();
    wGeo.setAttribute("position", new THREE.BufferAttribute(wc.positions, 3));

    const ec = buildTumorContours("et", 24, 60);
    const eGeo = new THREE.BufferGeometry();
    eGeo.setAttribute("position", new THREE.BufferAttribute(ec.positions, 3));

    return {
      hemiL: buildRealisticHemisphere(1),
      hemiR: buildRealisticHemisphere(-1),
      cerebellumGeo: buildRealisticCerebellum(),
      brainstemGeo: buildBrainstem(),
      brainSlices: bGeo,
      wtSlices: wGeo,
      etSlices: eGeo,
    };
  }, []);

  const cortexWireMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: srgb("#85ffaa"),
        wireframe: true,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
      }),
    [],
  );

  const cortexSolidMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: srgb("#174723"),
        transparent: true,
        opacity: 0.60,
        depthWrite: false,
      }),
    [],
  );

  const cerebellumMat = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: srgb("#7cff9b"),
        transparent: true,
        opacity: 0.82,
        depthWrite: false,
      }),
    [],
  );

  const brainstemMat = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: srgb("#5fd37d"),
        transparent: true,
        opacity: 0.78,
        depthWrite: false,
      }),
    [],
  );

  const sliceLineMat = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: srgb("#7cff9b"),
        transparent: true,
        opacity: 0.75,
        depthWrite: false,
      }),
    [],
  );

  const edemaMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: srgb("#ffb000"),
        wireframe: true,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const enhancingMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: srgb("#ff5a1f"),
        wireframe: true,
        transparent: true,
        opacity: 1.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const necroticMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: srgb("#ff3b3b"),
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  useFrame((state) => {
    const p = frameState.heroProgress;
    const t = state.clock.elapsedTime;
    const zoom = smooth(clamp01((p - 0.62) / 0.38));

    cortexWireMat.opacity = 0.65 + zoom * 0.35;
    cortexSolidMat.opacity = 0.45 + zoom * 0.30;
    cerebellumMat.opacity = 0.65 + zoom * 0.35;
    brainstemMat.opacity = 0.60 + zoom * 0.35;
    sliceLineMat.opacity = 0.50 + zoom * 0.45;

    edemaMat.opacity = (0.75 + zoom * 0.25) * (0.85 + 0.15 * Math.sin(t * 4));
    enhancingMat.opacity = 0.88 + zoom * 0.12;
    necroticMat.opacity = (0.8 + zoom * 0.2) * (0.8 + 0.2 * Math.cos(t * 6));

    if (reticleRef.current) {
      reticleRef.current.visible = zoom > 0.08;
      reticleRef.current.scale.setScalar(0.85 + zoom * 0.25);
      reticleRef.current.rotation.z = t * 0.75;
      reticleRef.current.rotation.x = Math.sin(t * 0.9) * 0.35;
    }
  });

  return (
    <group ref={brainGroup} position={HEAD_CENTER}>
      <group>
        <mesh geometry={hemiL} material={cortexSolidMat} />
        <mesh geometry={hemiL} material={cortexWireMat} />
        <mesh geometry={hemiR} material={cortexSolidMat} />
        <mesh geometry={hemiR} material={cortexWireMat} />
      </group>

      <lineSegments geometry={cerebellumGeo} material={cerebellumMat} />
      <lineSegments geometry={brainstemGeo} material={brainstemMat} />

      <group scale={[0.17, 0.17, 0.17]} rotation={[0, 0, -Math.PI / 2]}>
        <lineSegments geometry={brainSlices} material={sliceLineMat} />
        <lineSegments geometry={wtSlices}>
          <lineBasicMaterial
            color={srgb("#ffb000")}
            transparent
            opacity={0.95}
            depthWrite={false}
          />
        </lineSegments>
        <lineSegments geometry={etSlices}>
          <lineBasicMaterial
            color={srgb("#ff5a1f")}
            transparent
            opacity={1}
            depthWrite={false}
          />
        </lineSegments>
      </group>

      <group position={[0.035, 0.055, 0.058]}>
        <mesh material={edemaMat} scale={[1.15, 0.95, 1.05]}>
          <icosahedronGeometry args={[0.052, 2]} />
        </mesh>
        <mesh material={enhancingMat} scale={[1.05, 1.1, 0.95]}>
          <icosahedronGeometry args={[0.034, 2]} />
        </mesh>
        <mesh material={necroticMat}>
          <icosahedronGeometry args={[0.017, 1]} />
        </mesh>

        <group ref={reticleRef}>
          <mesh>
            <ringGeometry args={[0.064, 0.068, 48]} />
            <meshBasicMaterial
              color={srgb("#ffb000")}
              side={THREE.DoubleSide}
              transparent
              opacity={0.9}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
            />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.074, 0.078, 48]} />
            <meshBasicMaterial
              color={srgb("#ff5a1f")}
              side={THREE.DoubleSide}
              transparent
              opacity={0.75}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
            />
          </mesh>
        </group>
      </group>
    </group>
  );
}

/**
 * SCANNER GANTRY (Open Hollow Bore Ring)
 * Stage 1 (p: 0 -> 0.62): Sweeps along the patient body from feet (-1.55) to brain (HEAD_CENTER.x = 1.36).
 * Stage 2 (p: 0.62 -> 1.0): Remains locked at the brain while the camera zooms in.
 */
function ScannerGantry() {
  const group = useRef<THREE.Group>(null);

  const ringMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: srgb("#7cff9b"), wireframe: true, transparent: true, opacity: 0.85 }),
    [],
  );
  const housingMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: srgb("#4fae68"), wireframe: true, transparent: true, opacity: 0.24 }),
    [],
  );
  const beamRingMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: srgb("#7cff9b"),
        wireframe: true,
        transparent: true,
        opacity: 0.70,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );
  const amberMat = useMemo(() => new THREE.MeshBasicMaterial({ color: srgb("#ffb000") }), []);

  useFrame(() => {
    const p = frameState.heroProgress;
    const sweepPhase = smooth(clamp01(p / 0.62));
    const zoomPhase = smooth(clamp01((p - 0.62) / 0.38));
    // Sweeps from feet x = -1.55 along the patient directly to HEAD_CENTER.x = 1.36
    const x = -1.55 + sweepPhase * (HEAD_CENTER.x - -1.55);
    if (group.current) group.current.position.x = x;
    ringMat.opacity = 0.85 - zoomPhase * 0.55;
    housingMat.opacity = 0.24 * (1 - zoomPhase * 0.80);
    beamRingMat.opacity = 0.70 * (1 - zoomPhase * 0.85);
  });

  return (
    <group ref={group} position={[-1.55, 0.22, 0]}>
      {/* Superconducting Ring Magnet */}
      <mesh rotation={[0, Math.PI / 2, 0]} material={ringMat}>
        <torusGeometry args={[0.80, 0.026, 6, 96]} />
      </mesh>
      {/* Outer Gantry Housing */}
      <mesh rotation={[0, Math.PI / 2, 0]} material={housingMat}>
        <torusGeometry args={[1.00, 0.16, 8, 48]} />
      </mesh>
      {/* Inner Bore Laser Scan Ring (Hollow open bore, perfectly clear inside!) */}
      <mesh rotation={[0, Math.PI / 2, 0]} material={beamRingMat}>
        <ringGeometry args={[0.76, 0.79, 64]} />
      </mesh>
      {/* Tomograph Bore LEDs */}
      {[0.35, 0.5, 0.65].map((a, i) => (
        <mesh
          key={i}
          position={[0.08, Math.sin(a * Math.PI) * 1.14, Math.cos(a * Math.PI) * 1.14]}
          material={amberMat}
        >
          <sphereGeometry args={[0.022, 8, 6]} />
        </mesh>
      ))}
    </group>
  );
}

function Bed() {
  const { edges, ticks } = useMemo(() => {
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(3.6, 0.055, 0.88));
    const pts: number[] = [];
    for (let i = 0; i <= 36; i++) {
      const x = -1.8 + i * 0.1;
      const len = i % 10 === 0 ? 0.1 : i % 5 === 0 ? 0.065 : 0.032;
      pts.push(x, 0.032, 0.44, x, 0.032, 0.44 - len);
    }
    const ticks = new THREE.BufferGeometry();
    ticks.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return { edges, ticks };
  }, []);

  return (
    <group position={[0, 0.02, 0]}>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color={srgb("#4fae68")} transparent opacity={0.60} />
      </lineSegments>
      <lineSegments geometry={ticks}>
        <lineBasicMaterial color={srgb("#ffb000")} transparent opacity={0.70} />
      </lineSegments>
      {/* Bed Base Pedestal */}
      <mesh position={[-0.25, -0.38, 0]}>
        <boxGeometry args={[1.65, 0.72, 0.56, 4, 3, 2]} />
        <meshBasicMaterial color={srgb("#173320")} wireframe />
      </mesh>
    </group>
  );
}

/**
 * 3D SCANNER STAGE
 * - Patient and Bed stay at their exact world coordinates (no drift!).
 * - contentRef dynamically offsets (-px, -py, -pz) to cancel pivotRef (px, py, pz)
 *   so world coordinates of the head and body NEVER shift during scroll!
 */
function ScannerStage() {
  const pivotRef = useRef<THREE.Group>(null);
  const contentRef = useRef<THREE.Group>(null);
  const curYaw = useRef(0);
  const curPitch = useRef(0);

  useFrame(() => {
    curYaw.current += (frameState.dragYaw - curYaw.current) * 0.18;
    curPitch.current += (frameState.dragPitch - curPitch.current) * 0.18;

    const p = frameState.heroProgress;
    const zoomPhase = smooth(clamp01((p - 0.62) / 0.38));

    // Dynamic pivot: in wide view, pivot at body center (0.0, 0.22, 0);
    // when zoomed into the brain, pivot strictly at the cranial vault HEAD_CENTER (1.36, 0.28, 0).
    const px = HEAD_CENTER.x * zoomPhase;
    const py = 0.22 + (HEAD_CENTER.y - 0.22) * zoomPhase;
    const pz = HEAD_CENTER.z * zoomPhase;

    if (pivotRef.current && contentRef.current) {
      pivotRef.current.position.set(px, py, pz);
      // Cancel out the pivot translation so the patient never shifts in world space!
      contentRef.current.position.set(-px, -py, -pz);
      pivotRef.current.rotation.y = curYaw.current;
      pivotRef.current.rotation.x = curPitch.current;
      pivotRef.current.rotation.z = 0;
    }
  });

  return (
    <group ref={pivotRef}>
      <group ref={contentRef}>
        <Bed />
        <AnatomicalPatient />
        <BrainAndTumor />
        <ScannerGantry />
      </group>
    </group>
  );
}

/**
 * TWO-STAGE CAMERA RIG:
 * 
 * Stage 1 (Scroll 0% -> 62%):
 * - Camera stays in the wide full-body view while the scanner ring travels from
 *   the feet across the legs, hips, chest, and neck until it reaches the brain.
 * 
 * Stage 2 (Scroll 62% -> 100%):
 * - Once the scanner ring has arrived at the brain (HEAD_CENTER = 1.36, 0.28, 0),
 *   the camera zooms straight into the brain so the 3D cerebral hemispheres and
 *   glioma fill the center of the screen!
 */
function CameraRig() {
  const look = useMemo(() => new THREE.Vector3(), []);
  const camTarget = useMemo(() => new THREE.Vector3(), []);

  useFrame(({ camera }) => {
    const p = frameState.heroProgress;
    const sweepPhase = smooth(clamp01(p / 0.62));
    const zoomPhase = smooth(clamp01((p - 0.62) / 0.38));

    // Stage 1: Wide full-body view with gentle tracking of the gantry ring
    const wideX = -0.05 + sweepPhase * 0.25;
    const wideY = 0.85;
    const wideZ = 3.95;

    // Stage 2: Close cranial zoom directly onto HEAD_CENTER (1.36, 0.28, 0)
    const closeX = HEAD_CENTER.x; // 1.36
    const closeY = HEAD_CENTER.y + 0.14; // 0.42
    const closeZ = 0.62; // 0.62m distance fills the screen with the 3D brain

    camTarget.set(
      wideX + (closeX - wideX) * zoomPhase + frameState.pointerX * (0.08 * (1 - zoomPhase * 0.7)),
      wideY + (closeY - wideY) * zoomPhase + -frameState.pointerY * (0.06 * (1 - zoomPhase * 0.7)),
      wideZ + (closeZ - wideZ) * zoomPhase,
    );
    camera.position.lerp(camTarget, 0.16);

    // Look target transitions from body center straight to the brain center
    const lookWideX = -0.05 + sweepPhase * 0.25;
    const lookWideY = 0.22;
    const lookWideZ = 0;

    const lookCloseX = HEAD_CENTER.x; // 1.36
    const lookCloseY = HEAD_CENTER.y + 0.02; // 0.30
    const lookCloseZ = HEAD_CENTER.z; // 0.0

    look.set(
      lookWideX + (lookCloseX - lookWideX) * zoomPhase,
      lookWideY + (lookCloseY - lookWideY) * zoomPhase,
      lookWideZ + (lookCloseZ - lookWideZ) * zoomPhase,
    );
    camera.lookAt(look);
  });
  return null;
}

export default function HeroScene() {
  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [-0.05, 0.85, 3.95], fov: 38, near: 0.03, far: 60 }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      aria-hidden
    >
      <color attach="background" args={["#0b0d0a"]} />
      <ScannerStage />
      <CameraRig />
      <CRTPass />
    </Canvas>
  );
}
