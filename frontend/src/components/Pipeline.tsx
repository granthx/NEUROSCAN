"use client";
/**
 * SECTION 04 — PRINTED HOSPITAL EQUIPMENT SCHEMATIC (CREAM PAPER)
 * Technical manual fold-out diagram showing the 5-stage Attention U-Net
 * pipeline with animated signal traces and interactive node inspection.
 */
import { useState } from "react";
import { click } from "@/lib/sound";

type StageSpec = {
  id: string;
  code: string;
  title: string;
  subtitle: string;
  tensor: string;
  formula: string;
  notes: string[];
};

const STAGES: StageSpec[] = [
  {
    id: "input",
    code: "STAGE A",
    title: "4-CHANNEL NIfTI INGEST",
    subtitle: "FLAIR · T1 · T1ce · T2",
    tensor: "(4, 240, 240, 155) float32",
    formula: "X = { I_FLAIR, I_T1, I_T1ce, I_T2 }",
    notes: [
      "FLAIR highlights peritumoral vasogenic edema in bright hyperintensity.",
      "T1ce (gadolinium) delineates the active blood-brain-barrier breakdown rim.",
      "Co-registered 1 mm³ isotropic skull-stripped volumes.",
    ],
  },
  {
    id: "prep",
    code: "STAGE B",
    title: "Z-SCORE & 192² CROP",
    subtitle: "PARENCHYMA NORMALIZATION",
    tensor: "(4, 155, 192, 192) float16",
    formula: "x' = (x - μ_brain) / (σ_brain + 1e-8)",
    notes: [
      "Mean and standard deviation computed strictly inside non-zero brain mask.",
      "Center-cropped from 240×240 to 192×192 to strip empty air margin.",
      "BraTS raw label 4 remapped to contiguous class 3 (0=BG, 1=NCR, 2=ED, 3=ET).",
    ],
  },
  {
    id: "unet",
    code: "STAGE C",
    title: "31.5M ATTENTION U-NET",
    subtitle: "64 → 128 → 256 → 512 → 1024",
    tensor: "Logits (B, 4, 192, 192)",
    formula: "α = σ( ψ( ReLU(W_g g + W_x x) ) )",
    notes: [
      "4 encoder-decoder levels with DoubleConv (Conv3×3 + BatchNorm + ReLU).",
      "Additive Attention Gates filter skip connections to suppress healthy tissue noise.",
      "Trained with Weighted Dice (1.2) + Focal Loss (1.5, γ=2.0) under FP16 AMP.",
    ],
  },
  {
    id: "calib",
    code: "STAGE D",
    title: "TTA ×3 & CALIBRATION",
    subtitle: "SUB-COMPARTMENT THRESHOLDS",
    tensor: "Probs (4, 192, 192) → uint8",
    formula: "p_ET > 0.30 | p_NCR > 0.35 | p_ED > 0.40",
    notes: [
      "Test-Time Augmentation averages original, horizontal-flip, and vertical-flip probabilities.",
      "Calibrated decision boundaries recover subtle enhancing rims missed by 0.5 argmax.",
      "3D connected-component filter removes isolated false-positive clusters (<50 voxels).",
    ],
  },
  {
    id: "vol",
    code: "STAGE E",
    title: "3D VOLUMETRIC MASK",
    subtitle: "WT · TC · ET DELINEATION",
    tensor: "(240, 240, 155) uint8 NIfTI",
    formula: "WT = {1,2,3} · TC = {1,3} · ET = {3}",
    notes: [
      "Outputs NIfTI segmentation volume + quantitative cm³ sub-region integration.",
      "Computes 3D Dice (F1), Jaccard (J1), and 95th-percentile Hausdorff distance.",
    ],
  },
];

export function Pipeline() {
  const [activeId, setActiveId] = useState<string>("unet");
  const active = STAGES.find((s) => s.id === activeId) ?? STAGES[2];

  return (
    <section id="pipeline" aria-labelledby="pipeline-heading" className="w-full px-4 py-16 sm:px-8">
      <div className="paper relative rounded-[6px] border-2 border-[#a8a28c] p-6 shadow-2xl sm:p-10">
        {/* Blueprint Corner Stamp & Document Header */}
        <div className="flex flex-wrap items-start justify-between gap-4 border-b-2 border-ink pb-5">
          <div>
            <div className="stencil text-xs text-ink-dim">
              DOCUMENT DWG-89-402 // TECHNICAL SYSTEM SCHEMATIC
            </div>
            <h2 id="pipeline-heading" className="font-term text-4xl tracking-tight text-ink sm:text-5xl">
              SIGNAL PROCESSING &amp; NEURAL ARCHITECTURE
            </h2>
          </div>

          <div className="border-2 border-ink px-3 py-1.5 font-mono text-[11px] text-ink">
            <div>REV: 3.1 (ATTN-UNET-F64)</div>
            <div>PARAMS: 31,564,016 · AMP FP16</div>
          </div>
        </div>

        {/* Interactive Animated SVG Schematic Flow */}
        <div className="mt-8 overflow-x-auto pb-2">
          <div className="min-w-[880px]">
            <svg
              viewBox="0 0 1000 185"
              className="w-full"
              role="img"
              aria-label="Five-stage neural segmentation schematic from MRI input to 3D volume"
            >
              <defs>
                <pattern id="bp-grid" width="20" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1c1b16" strokeOpacity="0.08" strokeWidth="1" />
                </pattern>
              </defs>
              <rect x="0" y="0" width="1000" height="185" fill="url(#bp-grid)" />

              {/* Signal trace wires between blocks */}
              {[0, 1, 2, 3].map((i) => {
                const x1 = 182 + i * 200;
                const x2 = 212 + i * 200;
                return (
                  <g key={i}>
                    <line x1={x1} y1={92} x2={x2} y2={92} stroke="#1c1b16" strokeWidth="2.5" />
                    <circle r="4" fill="#b3261e">
                      <animateMotion dur="1.8s" repeatCount="indefinite" path={`M${x1},92 L${x2},92`} />
                    </circle>
                    <polygon points={`${x2 - 6},87 ${x2},92 ${x2 - 6},97`} fill="#1c1b16" />
                  </g>
                );
              })}

              {/* Schematic Nodes */}
              {STAGES.map((st, i) => {
                const x = 14 + i * 200;
                const selected = st.id === activeId;
                return (
                  <g
                    key={st.id}
                    onClick={() => {
                      click(1.05);
                      setActiveId(st.id);
                    }}
                    className="cursor-pointer"
                  >
                    <rect
                      x={x}
                      y={22}
                      width={168}
                      height={140}
                      rx={3}
                      fill={selected ? "#1c1b16" : "#e8e4d0"}
                      stroke="#1c1b16"
                      strokeWidth={selected ? "3" : "2"}
                    />
                    <text
                      x={x + 12}
                      y={44}
                      fill={selected ? "#ffb000" : "#4a473b"}
                      fontSize="10"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {st.code}
                    </text>
                    <text
                      x={x + 12}
                      y={68}
                      fill={selected ? "#e8e4d0" : "#1c1b16"}
                      fontSize="13"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {st.title}
                    </text>
                    <text
                      x={x + 12}
                      y={88}
                      fill={selected ? "#7cff9b" : "#4a473b"}
                      fontSize="10"
                      fontFamily="monospace"
                    >
                      {st.subtitle}
                    </text>
                    <line
                      x1={x + 12}
                      y1={102}
                      x2={x + 156}
                      y2={102}
                      stroke={selected ? "#4a473b" : "#1c1b16"}
                      strokeDasharray="3 3"
                    />
                    <text
                      x={x + 12}
                      y={124}
                      fill={selected ? "#e8e4d0" : "#1c1b16"}
                      fontSize="10"
                      fontFamily="monospace"
                    >
                      {st.tensor}
                    </text>
                    <text
                      x={x + 12}
                      y={147}
                      fill={selected ? "#ffb000" : "#b3261e"}
                      fontSize="9"
                      fontFamily="monospace"
                    >
                      [CLICK TO INSPECT]
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Selected Stage Inspector Card */}
        <div className="mt-4 grid grid-cols-1 gap-6 rounded-[4px] border-2 border-ink bg-paper-2 p-5 lg:grid-cols-12">
          <div className="border-b border-ink/30 pb-4 lg:col-span-5 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-5">
            <div className="stencil text-xs text-stamp">{active.code} // SPECIFICATION</div>
            <h3 className="mt-1 font-term text-3xl text-ink">{active.title}</h3>
            <div className="mt-3 rounded-[2px] bg-ink p-3 font-mono text-xs text-phos">
              <div>TENSOR: {active.tensor}</div>
              <div className="mt-1 text-amber">EQ: {active.formula}</div>
            </div>
          </div>

          <div className="flex flex-col justify-center lg:col-span-7">
            <div className="stencil mb-2 text-xs text-ink-dim">ENGINEERING NOTES</div>
            <ul className="space-y-2 font-mono text-xs text-ink sm:text-sm">
              {active.notes.map((n, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="font-bold text-stamp">[{idx + 1}]</span>
                  <span>{n}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

