"use client";
import { create } from "zustand";
import type { Plane, Modality } from "./phantom";

export type CrtMode = "full" | "light" | "off";

type Layers = { ed: boolean; et: boolean; ncr: boolean };

type ViewerState = {
  /* --- global console --- */
  booted: boolean;
  crtMode: CrtMode;
  sound: boolean;
  shuttingDown: boolean;
  setBooted: (b: boolean) => void;
  setCrtMode: (m: CrtMode) => void;
  toggleSound: () => void;
  setShuttingDown: (b: boolean) => void;

  /* --- slice viewer --- */
  plane: Plane;
  slices: Record<Plane, number>;
  modality: Modality;
  layers: Layers;
  source: "gt" | "pred";
  overlayOpacity: number;
  setPlane: (p: Plane) => void;
  setSlice: (p: Plane, i: number) => void;
  setModality: (m: Modality) => void;
  toggleLayer: (k: keyof Layers) => void;
  setSource: (s: "gt" | "pred") => void;
  setOverlayOpacity: (v: number) => void;
};

export const useViewer = create<ViewerState>((set) => ({
  booted: false,
  crtMode: "full",
  sound: false,
  shuttingDown: false,
  setBooted: (booted) => set({ booted }),
  setCrtMode: (crtMode) => set({ crtMode }),
  toggleSound: () => set((s) => ({ sound: !s.sound })),
  setShuttingDown: (shuttingDown) => set({ shuttingDown }),

  plane: "axial",
  // Default slices pass through the phantom lesion.
  slices: { axial: 94, coronal: 105, sagittal: 159 },
  modality: "flair",
  layers: { ed: true, et: true, ncr: true },
  source: "pred",
  overlayOpacity: 0.55,
  setPlane: (plane) => set({ plane }),
  setSlice: (p, i) => set((s) => ({ slices: { ...s.slices, [p]: i } })),
  setModality: (modality) => set({ modality }),
  toggleLayer: (k) => set((s) => ({ layers: { ...s.layers, [k]: !s.layers[k] } })),
  setSource: (source) => set({ source }),
  setOverlayOpacity: (overlayOpacity) => set({ overlayOpacity }),
}));

/**
 * Non-reactive, per-frame values (read inside useFrame without re-rendering React).
 */
export const frameState = {
  heroProgress: 0, // 0..1 scroll progress through the gantry hero
  pointerX: 0,
  pointerY: 0,
  isDraggingHero: false,
  dragYaw: 0,      // manual left-click 3D yaw rotation (radians)
  dragPitch: 0,    // manual left-click 3D pitch rotation (radians)
};
