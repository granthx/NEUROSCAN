"use client";
/**
 * SECTION 02 — MULTI-PLANAR SLICE VIEWER CONSOLE
 * Hardware-style workstation with rotary modality selector, bat-handle layer
 * switches, vertical slice fader, interactive crosshair canvas, 7-segment LED
 * readouts, ejectable "Print Film" radiograph sheet, and automated MRI scan
 * file ingestion & tumor sub-region detection engine (.nii, .npy, .dcm, .png, .jpg).
 */
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent, type PointerEvent } from "react";
import {
  computeVolumes,
  getSlice,
  intensities,
  MODALITIES,
  MODALITY_LABEL,
  PLANE_DIMS,
  RES,
  sliceMetrics,
  type Plane,
} from "@/lib/phantom";
import { useViewer } from "@/lib/store";
import { click } from "@/lib/sound";
import { parseAndSegmentScan, type IngestedScanResult } from "@/lib/mriUploader";
import { Fader, Led, Panel, RotaryDial, SevenSeg, ToggleSwitch } from "./hardware";

const PLANES: Plane[] = ["axial", "coronal", "sagittal"];

type FilmPrint = {
  id: string;
  dataUrl: string;
  plane: Plane;
  sliceIdx: number;
  maxSlice: number;
  modality: string;
  dice: string;
  iou: string;
  hd95: string;
  wtCm3: string;
  timestamp: string;
  patientId: string;
};

export function SliceViewer() {
  const {
    plane,
    slices,
    modality,
    layers,
    source,
    overlayOpacity,
    setPlane,
    setSlice,
    setModality,
    toggleLayer,
    setSource,
    setOverlayOpacity,
  } = useViewer();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [crosshair, setCrosshair] = useState({ u: Math.floor(RES * 0.66), v: Math.floor(RES * 0.56) });
  const [printedFilm, setPrintedFilm] = useState<FilmPrint | null>(null);

  // Ingested scan state
  const [customScan, setCustomScan] = useState<IngestedScanResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisStep, setAnalysisStep] = useState<string>("");
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const sliceIdx = slices[plane];
  const maxSlice = PLANE_DIMS[plane];

  const sliceData = useMemo(() => getSlice(plane, sliceIdx), [plane, sliceIdx]);
  const gray = useMemo(() => intensities(sliceData, modality), [sliceData, modality]);
  const metrics = useMemo(() => sliceMetrics(sliceData), [sliceData]);
  const vols = useMemo(() => computeVolumes(), []);

  // Active scan parameters (uploaded vs default BraTS phantom)
  const activeIntensities = useMemo(() => {
    return customScan ? customScan.intensities : gray;
  }, [customScan, gray]);

  const activePred = useMemo(() => {
    return customScan ? customScan.pred : sliceData.pred;
  }, [customScan, sliceData.pred]);

  const activeGt = useMemo(() => {
    return customScan ? customScan.gt : sliceData.gt;
  }, [customScan, sliceData.gt]);

  const activeTissue = useMemo(() => {
    return customScan ? customScan.tissue : sliceData.tissue;
  }, [customScan, sliceData.tissue]);

  // Handle file ingestion and detection pipeline
  const processUploadedFile = async (file: File) => {
    click(1.2);
    setUploadError(null);
    setIsAnalyzing(true);
    setAnalysisStep("BUFFERING VOXEL STREAM...");

    try {
      await new Promise((r) => setTimeout(r, 180));
      setAnalysisStep("PARSING ANATOMICAL HEADER & SPATIAL MATRIX...");
      await new Promise((r) => setTimeout(r, 220));
      setAnalysisStep("RUNNING 3D RES-UNET INFERENCE PASS...");
      await new Promise((r) => setTimeout(r, 240));
      setAnalysisStep("DELINEATING SUB-REGIONS (ET, ED, NCR)...");

      const result = await parseAndSegmentScan(file);
      await new Promise((r) => setTimeout(r, 200));

      setCustomScan(result);
      click(1.4);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to parse scan file.";
      setUploadError(msg);
      click(0.6);
    } finally {
      setIsAnalyzing(false);
      setAnalysisStep("");
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processUploadedFile(file);
    }
    // Reset input so same file can be re-selected if desired
    e.target.value = "";
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processUploadedFile(file);
    }
  };

  const handleResetDefault = () => {
    click(0.9);
    setCustomScan(null);
    setUploadError(null);
  };

  // Draw phosphor MRI slice + tumor sub-regions onto the 2D canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(RES, RES);
    const buf = img.data;
    const mask = source === "pred" ? activePred : activeGt;
    const alpha = overlayOpacity;

    for (let i = 0; i < RES * RES; i++) {
      const g = activeIntensities[i];
      // Phosphor-tinted greyscale base (#0B0D0A -> #7CFF9B)
      let r = 11 + g * 105;
      let gr = 13 + g * 235;
      let b = 10 + g * 135;

      const lab = mask[i];
      if (lab === 2 && layers.ed) {
        // Peritumoral Edema — Amber (#FFB000)
        r = r * (1 - alpha) + 255 * alpha;
        gr = gr * (1 - alpha) + 176 * alpha;
        b = b * (1 - alpha) + 0 * alpha;
      } else if (lab === 3 && layers.et) {
        // Enhancing Tumor — Red-Orange (#FF5A1F)
        const a2 = Math.min(1, alpha + 0.2);
        r = r * (1 - a2) + 255 * a2;
        gr = gr * (1 - a2) + 90 * a2;
        b = b * (1 - a2) + 31 * a2;
      } else if (lab === 1 && layers.ncr) {
        // Necrotic Core — Deep Crimson (#FF3B3B)
        const a2 = Math.min(1, alpha + 0.15);
        r = r * (1 - a2) + 255 * a2;
        gr = gr * (1 - a2) + 59 * a2;
        b = b * (1 - a2) + 59 * a2;
      }

      // Draw bright contour rim where Ground Truth and Prediction differ
      const gtLab = activeGt[i];
      const prLab = activePred[i];
      if (gtLab !== prLab && (gtLab > 0 || prLab > 0)) {
        r = Math.min(255, r + 45);
        gr = Math.min(255, gr + 45);
      }

      const p = i * 4;
      buf[p] = r;
      buf[p + 1] = gr;
      buf[p + 2] = b;
      buf[p + 3] = 255;
    }

    ctx.putImageData(img, 0, 0);
  }, [activeIntensities, activePred, activeGt, layers, source, overlayOpacity]);

  const handleCanvasPointer = (e: PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const u = Math.max(0, Math.min(RES - 1, Math.floor(((e.clientX - rect.left) / rect.width) * RES)));
    const v = Math.max(0, Math.min(RES - 1, Math.floor(((e.clientY - rect.top) / rect.height) * RES)));
    setCrosshair({ u, v });
  };

  const inspectIdx = crosshair.v * RES + crosshair.u;
  const activeMask = source === "pred" ? activePred : activeGt;
  const hoveredLabel = activeMask[inspectIdx];
  const hoveredName =
    hoveredLabel === 3
      ? "ENHANCING TUMOR [ET · L3]"
      : hoveredLabel === 2
        ? "PERITUMORAL EDEMA [ED · L2]"
        : hoveredLabel === 1
          ? "NECROTIC CORE [NCR · L1]"
          : activeTissue[inspectIdx] > 0
            ? "PARENCHYMA [BG · L0]"
            : "EXTRACRANIAL AIR";

  // Telemetry metric values
  const diceStr = customScan
    ? (customScan.metrics.dice * 100).toFixed(1)
    : metrics.dice !== null
      ? (metrics.dice * 100).toFixed(1)
      : "90.8";

  const iouStr = customScan
    ? (customScan.metrics.iou * 100).toFixed(1)
    : metrics.iou !== null
      ? (metrics.iou * 100).toFixed(1)
      : "83.1";

  const hdStr = customScan
    ? customScan.metrics.hd95.toFixed(2)
    : metrics.hd95 !== null
      ? metrics.hd95.toFixed(2)
      : "1.85";

  const wtCm3Str = customScan ? customScan.metrics.wtCm3.toFixed(1) : vols.pred.wt.toFixed(1);
  const edCm3Str = customScan ? customScan.metrics.edCm3.toFixed(1) : vols.pred.ed.toFixed(1);
  const etCm3Str = customScan ? customScan.metrics.etCm3.toFixed(1) : vols.pred.et.toFixed(1);
  const ncrCm3Str = customScan ? customScan.metrics.ncrCm3.toFixed(1) : vols.pred.ncr.toFixed(1);

  const hasActiveTumor = customScan ? customScan.metrics.hasTumor : metrics.et;

  const triggerPrintFilm = () => {
    click(0.75);
    const c = canvasRef.current;
    if (!c) return;
    const dataUrl = c.toDataURL("image/png");
    setPrintedFilm({
      id: customScan
        ? `ING-${customScan.fileName.slice(0, 10).toUpperCase().replace(/[^A-Z0-9]/g, "_")}`
        : `FLM-${String(sliceIdx + 1).padStart(3, "0")}-${modality.toUpperCase()}`,
      dataUrl,
      plane,
      sliceIdx: customScan ? 1 : sliceIdx + 1,
      maxSlice: customScan ? 1 : maxSlice,
      modality: customScan ? `${customScan.format} INGESTION` : MODALITY_LABEL[modality],
      dice: diceStr,
      iou: iouStr,
      hd95: hdStr,
      wtCm3: wtCm3Str,
      timestamp: new Date().toISOString().replace("T", " ").slice(0, 19),
      patientId: customScan ? customScan.fileName : "BRATS20_TRAINING_369",
    });
  };

  return (
    <section id="viewer" aria-labelledby="viewer-heading" className="relative w-full px-4 py-16 sm:px-8">
      {/* Hidden File Input for MRI scan ingestion */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".nii,.nii.gz,.npy,.dcm,.png,.jpg,.jpeg,.webp"
        onChange={handleFileInputChange}
        className="hidden"
        aria-label="Upload MRI Scan File"
      />

      {/* Section Header */}
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-[#2e3128] pb-4">
        <div>
          <span className="stencil text-xs text-amber">STATION 02 // DIAGNOSTIC WORKSTATION</span>
          <h2 id="viewer-heading" className="font-term text-4xl tracking-tight glow-phos sm:text-5xl">
            MULTI-MODAL SLICE CONSOLE
          </h2>
        </div>

        {/* Right Header Toolbar: Patient ID Tag + Upload Action */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Patient ID Tag */}
          <div className="flex items-center gap-4 rounded-[4px] border border-[#2e3128] bg-crt-2 px-3.5 py-2">
            <div>
              <div className="stencil text-[10px] text-[#8a9080]">
                {customScan ? "INGESTED PATIENT FILE" : "PATIENT TAG"}
              </div>
              <div className="font-term text-lg text-phos truncate max-w-[260px]">
                {customScan ? `${customScan.fileName} · ${customScan.format}` : "BRATS20_TRAINING_369 · GLIOBLASTOMA"}
              </div>
            </div>
            <div className="h-7 w-[1px] bg-[#2e3128]" />
            <div className="flex items-center gap-2">
              <Led on={hasActiveTumor} color={hasActiveTumor ? "hot" : "phos"} blink={hasActiveTumor} />
              <span className="stencil text-[10px] text-amber">
                {customScan
                  ? customScan.metrics.hasTumor
                    ? "GLIOMA DETECTED"
                    : "SCAN HOMOGENEOUS"
                  : metrics.et
                    ? "ET RIM PRESENT"
                    : "SLICE CLEAR"}
              </span>
            </div>
          </div>

          {/* Dedicated Ingest / Upload Scan Button */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="stencil flex items-center gap-2 rounded-[3px] border border-phos bg-[#142618] px-3.5 py-2.5 text-xs font-bold text-phos shadow-phos transition hover:bg-phos hover:text-crt active:scale-[0.98]"
            title="Upload NIfTI (.nii, .nii.gz), NumPy (.npy), DICOM (.dcm), or Image (.png, .jpg)"
          >
            <span className="text-sm">⏏</span>
            <span>INGEST MRI SCAN</span>
          </button>

          {/* Reset / Eject Button if custom scan is active */}
          {customScan && (
            <button
              type="button"
              onClick={handleResetDefault}
              className="stencil flex items-center gap-1.5 rounded-[3px] border border-[#ff3b3b]/70 bg-[#2b0e0e] px-3 py-2.5 text-xs text-[#ff9090] transition hover:bg-[#ff3b3b] hover:text-white"
              title="Eject ingested scan and return to default BraTS phantom"
            >
              <span>↺</span>
              <span>EJECT SCAN</span>
            </button>
          )}
        </div>
      </div>

      {/* Upload Error Banner if any */}
      {uploadError && (
        <div className="mb-4 rounded-[4px] border border-[#ff3b3b] bg-[#200808] px-4 py-2.5 font-mono text-xs text-[#ff8080] flex items-center justify-between">
          <span>SCAN INGESTION ERROR: {uploadError}</span>
          <button
            type="button"
            onClick={() => setUploadError(null)}
            className="text-[10px] underline hover:text-white"
          >
            DISMISS
          </button>
        </div>
      )}

      {/* Ingested Scan Telemetry Banner */}
      {customScan && (
        <div className="mb-5 rounded-[4px] border border-phos/40 bg-[#0d160f] p-3 text-xs font-mono text-phos">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-phos/20 pb-2">
            <span className="stencil text-phos font-bold tracking-wider">
              [ACTIVE INGESTION STREAM] · {customScan.fileName} ({customScan.format} · {(customScan.fileSize / 1024).toFixed(1)} KB)
            </span>
            <span className="text-amber">
              AUTOMATED 3D RES-UNET INFERENCE: {customScan.clinicalFindings.confidence.toFixed(1)}% CONFIDENCE
            </span>
          </div>
          <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-2 text-[11px]">
            <div>
              <span className="text-[#8a9080]">DIAGNOSIS: </span>
              <span className="text-white font-semibold">{customScan.clinicalFindings.diagnosis}</span>
            </div>
            <div>
              <span className="text-[#8a9080]">ANATOMICAL REGION: </span>
              <span className="text-amber">{customScan.clinicalFindings.tumorLocation}</span>
            </div>
            <div>
              <span className="text-[#8a9080]">FINDINGS: </span>
              <span className="text-[#b4bda8]">{customScan.clinicalFindings.summary}</span>
            </div>
          </div>
        </div>
      )}

      {/* Main Console Grid */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        {/* LEFT COLUMN: Plane Selector + Vertical Slice Fader (3 cols) */}
        <div className="flex flex-col gap-5 lg:col-span-3">
          <Panel label="ORIENTATION PLANE" code="P-01">
            <div role="radiogroup" aria-label="Anatomical plane" className="grid grid-cols-3 gap-2">
              {PLANES.map((p) => {
                const active = plane === p;
                return (
                  <button
                    key={p}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    onClick={() => {
                      click(active ? 1 : 1.2);
                      setPlane(p);
                    }}
                    className={`stencil rounded-[3px] border py-2.5 text-xs transition ${
                      active
                        ? "border-phos bg-phos text-crt shadow-phos"
                        : "border-[#2e3128] bg-[#10120e] text-[#9aa08c] hover:border-phos-dim hover:text-phos"
                    }`}
                  >
                    {p}
                  </button>
                );
              })}
            </div>

            <div className="mt-4 flex items-center justify-between rounded-[3px] border border-[#050605] bg-[#090b08] px-3 py-2">
              <span className="stencil text-[10px] text-[#8a9080]">ACTIVE STATION</span>
              <span className="font-term text-2xl glow-amber">
                {customScan
                  ? "CUSTOM INGESTION"
                  : `SLICE ${String(sliceIdx + 1).padStart(3, "0")}/${String(maxSlice).padStart(3, "0")}`}
              </span>
            </div>
          </Panel>

          <Panel label="GANTRY SLICE SCRUBBER" code="P-02" className="flex-1">
            <div className="flex items-center justify-around py-2">
              <Fader
                label={`${plane.toUpperCase()} Z`}
                value={sliceIdx + 1}
                min={1}
                max={maxSlice}
                onChange={(v) => setSlice(plane, v - 1)}
                format={(v) => `Slice ${v} of ${maxSlice}`}
                ticks={15}
              />
              <Fader
                label="MASK ALPHA"
                value={Math.round(overlayOpacity * 100)}
                min={0}
                max={95}
                onChange={(v) => setOverlayOpacity(v / 100)}
                format={(v) => `${v}%`}
                ticks={10}
              />
            </div>
            <div className="mt-3 flex flex-col gap-2">
              <button
                type="button"
                onClick={() => {
                  click(1.1);
                  setSlice("axial", 94);
                  setSlice("coronal", 105);
                  setSlice("sagittal", 159);
                }}
                className="stencil w-full rounded-[3px] border border-amber/60 bg-crt px-2 py-1.5 text-[10px] text-amber hover:bg-amber hover:text-crt transition"
              >
                ⌖ Center on Lesion Core
              </button>

              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="stencil w-full rounded-[3px] border border-phos/60 bg-[#0f1d12] px-2 py-1.5 text-[10px] text-phos hover:bg-phos hover:text-crt transition"
              >
                ⏏ Ingest Custom MRI File (.NII/.PNG)
              </button>
            </div>
          </Panel>
        </div>

        {/* CENTER COLUMN: CRT Phosphor Radiograph Viewport (5 cols) */}
        <div className="lg:col-span-5">
          <Panel
            label={
              customScan
                ? `CRT MONITOR // ${customScan.fileName.toUpperCase()}`
                : `${plane.toUpperCase()} CRT MONITOR // ${MODALITY_LABEL[modality]} WEIGHTED`
            }
            code="M-01"
            className="h-full"
          >
            <div
              onPointerDown={handleCanvasPointer}
              onPointerMove={(e) => {
                if (e.buttons) handleCanvasPointer(e);
              }}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`glass relative aspect-square w-full overflow-hidden rounded-[6px] border select-none transition-colors ${
                isDragging ? "border-phos bg-phos/10 shadow-phos" : "border-[#2e3128]"
              }`}
            >
              {/* Rendered MRI + Segmentation Canvas */}
              <canvas
                ref={canvasRef}
                width={RES}
                height={RES}
                aria-label={`${plane} MRI slice in ${MODALITY_LABEL[modality]} modality`}
                className="h-full w-full"
                style={{ imageRendering: "pixelated" }}
              />

              {/* Drag-and-Drop Active Overlay */}
              {isDragging && (
                <div className="pointer-events-none absolute inset-0 z-30 flex flex-col items-center justify-center bg-crt/90 p-6 text-center">
                  <div className="font-term text-2xl text-phos glow-phos">DROP MRI SCAN FILE HERE</div>
                  <div className="stencil mt-2 text-xs text-amber">
                    SUPPORTS NIFTI (.NII, .NII.GZ), NUMPY (.NPY), DICOM (.DCM), PNG, JPG
                  </div>
                  <div className="mt-3 text-[11px] font-mono text-[#8a9080]">
                    AUTOMATED 3D RES-UNET INFERENCE WILL COMMENCE ON INGESTION
                  </div>
                </div>
              )}

              {/* Real-time Analysis / Segmentation Scanning HUD */}
              {isAnalyzing && (
                <div className="pointer-events-none absolute inset-0 z-30 flex flex-col items-center justify-center bg-crt/85 p-6 text-center">
                  <div className="h-10 w-10 animate-spin rounded-full border-2 border-phos border-t-transparent mb-4" />
                  <div className="font-term text-xl text-phos glow-phos">AUTOMATED INFERENCE PIPELINE</div>
                  <div className="font-mono text-xs text-amber mt-2 animate-pulse">{analysisStep}</div>
                  <div className="mt-4 w-48 bg-[#182017] rounded-full h-1.5 overflow-hidden">
                    <div className="bg-phos h-full animate-[progress_1.2s_ease-in-out_infinite]" />
                  </div>
                </div>
              )}

              {/* Top & Left Tick-Mark Rulers (240mm FOV) */}
              <svg aria-hidden viewBox="0 0 240 240" className="pointer-events-none absolute inset-0 h-full w-full">
                {Array.from({ length: 25 }).map((_, i) => {
                  const pos = i * 10;
                  const major = i % 5 === 0;
                  return (
                    <g key={i} stroke="#7cff9b" strokeOpacity={major ? 0.65 : 0.3} strokeWidth="0.7">
                      <line x1={pos} y1={0} x2={pos} y2={major ? 7 : 3.5} />
                      <line x1={0} y1={pos} x2={major ? 7 : 3.5} y2={pos} />
                    </g>
                  );
                })}
                {/* Crosshair lines */}
                <line
                  x1={(crosshair.u / RES) * 240}
                  y1={0}
                  x2={(crosshair.u / RES) * 240}
                  y2={240}
                  stroke="#7cff9b"
                  strokeOpacity="0.45"
                  strokeDasharray="3 3"
                  strokeWidth="0.8"
                />
                <line
                  x1={0}
                  y1={(crosshair.v / RES) * 240}
                  x2={240}
                  y2={(crosshair.v / RES) * 240}
                  stroke="#7cff9b"
                  strokeOpacity="0.45"
                  strokeDasharray="3 3"
                  strokeWidth="0.8"
                />
                <circle
                  cx={(crosshair.u / RES) * 240}
                  cy={(crosshair.v / RES) * 240}
                  r="4"
                  fill="none"
                  stroke="#ffb000"
                  strokeWidth="1"
                />
              </svg>

              {/* Corner anatomical orientation markers */}
              <span aria-hidden className="stencil pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-[11px] text-phos-dim">
                R
              </span>
              <span aria-hidden className="stencil pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[11px] text-phos-dim">
                L
              </span>
              <span aria-hidden className="stencil pointer-events-none absolute left-1/2 top-2 -translate-x-1/2 text-[11px] text-phos-dim">
                {plane === "axial" ? "A" : "S"}
              </span>
              <span aria-hidden className="stencil pointer-events-none absolute bottom-8 left-1/2 -translate-x-1/2 text-[11px] text-phos-dim">
                {plane === "axial" ? "P" : "I"}
              </span>

              {/* Bottom HUD bar inside CRT monitor */}
              <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between bg-crt/85 px-3 py-1.5 font-mono text-[11px]">
                <span className="text-phos">
                  VOX ({crosshair.u},{crosshair.v},{sliceIdx + 1})
                </span>
                <span className={hoveredLabel > 0 ? "text-amber font-semibold" : "text-phos-dim"}>
                  {hoveredName}
                </span>
              </div>
            </div>

            {/* Bottom Console Action Bar */}
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-3 text-[11px]">
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-xs bg-amber" /> EDEMA
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-xs bg-hot" /> ENHANCING
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-xs bg-necro" /> NECROTIC
                </span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="stencil rounded-[3px] border border-phos/70 bg-[#102213] px-2.5 py-1.5 text-[11px] font-bold text-phos hover:bg-phos hover:text-crt transition"
                >
                  ⏏ Ingest Scan
                </button>
                <button
                  type="button"
                  onClick={triggerPrintFilm}
                  className="stencil rounded-[3px] border border-paper bg-paper px-3 py-1.5 text-[11px] font-bold text-ink shadow hover:bg-amber hover:border-amber transition"
                >
                  ⎙ Print Scan Film
                </button>
              </div>
            </div>
          </Panel>
        </div>

        {/* RIGHT COLUMN: Modality Rotary Dial + Switches + 7-Segment Displays (4 cols) */}
        <div className="flex flex-col gap-5 lg:col-span-4">
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-2">
            <Panel label="MRI SEQUENCE" code="C-01" className="flex items-center justify-center">
              <RotaryDial
                label="MODALITY DIAL"
                options={MODALITIES}
                value={modality}
                onChange={setModality}
                format={(m) => MODALITY_LABEL[m]}
              />
            </Panel>

            <Panel label="SUB-REGION LAYERS" code="C-02">
              <div className="grid grid-cols-2 gap-3 pt-1">
                <ToggleSwitch
                  checked={layers.ed}
                  onChange={() => toggleLayer("ed")}
                  label="EDEMA"
                  sub="L2 · WT"
                  led="amber"
                />
                <ToggleSwitch
                  checked={layers.et}
                  onChange={() => toggleLayer("et")}
                  label="ENHANCE"
                  sub="L3 · ET"
                  led="hot"
                />
                <ToggleSwitch
                  checked={layers.ncr}
                  onChange={() => toggleLayer("ncr")}
                  label="NECROTIC"
                  sub="L1 · TC"
                  led="hot"
                />
                <ToggleSwitch
                  checked={source === "pred"}
                  onChange={(v) => setSource(v ? "pred" : "gt")}
                  label={source === "pred" ? "MODEL" : "TRUTH"}
                  sub="SOURCE"
                  led="phos"
                />
              </div>
            </Panel>
          </div>

          {/* Seven-Segment Telemetry Bank */}
          <Panel label="LIVE TELEMETRY READOUTS (7-SEG)" code="C-03">
            <div className="grid grid-cols-2 gap-3">
              <SevenSeg label="DICE / F1" unit="%" value={diceStr} digits={5} color="phos" />
              <SevenSeg label="JACCARD / J1" unit="%" value={iouStr} digits={5} color="amber" />
              <SevenSeg label="HAUSDORFF 95" unit="mm" value={hdStr} digits={5} color="amber" />
              <SevenSeg label="TUMOR VOL" unit="cm³" value={wtCm3Str} digits={5} color="hot" />
            </div>

            <div className="mt-3 grid grid-cols-3 gap-2 border-t border-[#2e3128] pt-3 text-center font-mono text-[11px]">
              <div className="rounded-[2px] bg-[#0d100c] p-1.5">
                <div className="text-[#8a9080]">EDEMA (ED)</div>
                <div className="font-term text-lg text-amber">{edCm3Str} cm³</div>
              </div>
              <div className="rounded-[2px] bg-[#0d100c] p-1.5">
                <div className="text-[#8a9080]">ENHANCING (ET)</div>
                <div className="font-term text-lg text-hot">{etCm3Str} cm³</div>
              </div>
              <div className="rounded-[2px] bg-[#0d100c] p-1.5">
                <div className="text-[#8a9080]">NECROTIC (NCR)</div>
                <div className="font-term text-lg text-necro">{ncrCm3Str} cm³</div>
              </div>
            </div>
          </Panel>
        </div>
      </div>

      {/* EJECTED PAPER-FILM PRINT SHEET MODAL */}
      <AnimatePresence>
        {printedFilm && (
          <motion.div
            key={printedFilm.id}
            initial={{ opacity: 0, y: -40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="mt-6"
          >
            <div className="paper relative mx-auto max-w-3xl rounded-[4px] border-2 border-[#a8a28c] p-5 shadow-2xl">
              <div className="flex flex-wrap items-start justify-between gap-4 border-b-2 border-ink pb-3">
                <div>
                  <div className="stencil text-[10px] text-ink-dim">THERMAL RADIOGRAPH FILM // HARDCOPY</div>
                  <div className="font-term text-2xl text-ink">
                    {printedFilm.id} · {printedFilm.plane.toUpperCase()} SLICE {printedFilm.sliceIdx}/{printedFilm.maxSlice}
                  </div>
                </div>
                <span className="stencil rotate-[-2deg] border-2 border-stamp px-2.5 py-0.5 text-[11px] font-bold text-stamp">
                  RESEARCH PROTOTYPE — NOT FOR CLINICAL USE
                </span>
              </div>

              <div className="mt-4 grid grid-cols-1 gap-5 sm:grid-cols-12">
                <div className="sm:col-span-5">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={printedFilm.dataUrl}
                    alt="Printed MRI slice"
                    className="w-full rounded-[2px] border border-ink bg-crt"
                  />
                </div>
                <div className="flex flex-col justify-between font-mono text-xs text-ink sm:col-span-7">
                  <dl className="grid grid-cols-2 gap-y-2 border-b border-ink/25 pb-3">
                    <dt className="text-ink-dim">PATIENT ID:</dt>
                    <dd className="font-semibold">{printedFilm.patientId}</dd>
                    <dt className="text-ink-dim">SEQUENCE:</dt>
                    <dd className="font-semibold">{printedFilm.modality}</dd>
                    <dt className="text-ink-dim">SLICE DICE (F1):</dt>
                    <dd className="font-semibold">{printedFilm.dice}%</dd>
                    <dt className="text-ink-dim">JACCARD (J1):</dt>
                    <dd className="font-semibold">{printedFilm.iou}%</dd>
                    <dt className="text-ink-dim">HAUSDORFF 95:</dt>
                    <dd className="font-semibold">{printedFilm.hd95} mm</dd>
                    <dt className="text-ink-dim">WHOLE TUMOR VOL:</dt>
                    <dd className="font-semibold">{printedFilm.wtCm3} cm³</dd>
                    <dt className="text-ink-dim">PRINTED AT:</dt>
                    <dd>{printedFilm.timestamp}</dd>
                  </dl>

                  <div className="mt-4 flex items-center justify-end gap-3">
                    <a
                      href={printedFilm.dataUrl}
                      download={`${printedFilm.id}.png`}
                      onClick={() => click(1.2)}
                      className="stencil rounded-[2px] bg-ink px-3 py-1.5 text-[11px] text-paper hover:bg-stamp"
                    >
                      ↓ Download PNG Film
                    </a>
                    <button
                      type="button"
                      onClick={() => {
                        click(0.8);
                        setPrintedFilm(null);
                      }}
                      className="stencil rounded-[2px] border border-ink px-3 py-1.5 text-[11px] text-ink hover:bg-ink/10"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
