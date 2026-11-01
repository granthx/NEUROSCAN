"use client";
/**
 * SECTION 05 — OSCILLOSCOPE TRACE & PRINTED BENCHMARK READOUT
 * Renders the real epoch-by-epoch training histories from the three trained
 * checkpoints as glowing phosphor oscilloscope traces, paired with a
 * typewriter-animated benchmark table.
 */
import { useEffect, useState } from "react";
import historyData from "@/data/training-history.json";
import { MODEL_RUNS, type ModelRun } from "@/lib/config";
import { useInView, useReducedMotion } from "@/lib/hooks";
import { click } from "@/lib/sound";
import { Led, Panel, SevenSeg } from "./hardware";

type EpochPoint = {
  epoch: number;
  train_loss: number;
  val_loss: number;
  WT: number;
  TC: number;
  ET: number;
  mean_dice: number;
};

type TraceKey = "mean_dice" | "ET" | "TC" | "WT" | "train_loss";

const TRACE_MODES: { key: TraceKey; label: string; unit: string; min: number; max: number }[] = [
  { key: "mean_dice", label: "MEAN DICE", unit: "%", min: 0.82, max: 0.91 },
  { key: "ET", label: "ENHANCING (ET)", unit: "%", min: 0.84, max: 0.92 },
  { key: "TC", label: "TUMOR CORE (TC)", unit: "%", min: 0.84, max: 0.92 },
  { key: "WT", label: "WHOLE TUMOR (WT)", unit: "%", min: 0.81, max: 0.89 },
  { key: "train_loss", label: "TRAIN LOSS", unit: "L", min: 0.15, max: 0.65 },
];

const BENCHMARK_ROWS = [
  {
    rank: "STAGE 1",
    system: "Attention U-Net Baseline (5 ep · f16 · 2.0M)",
    wt: "89.57%",
    tc: "82.23%",
    et: "81.23%",
    mean: "84.35%",
    bal: "93.45%",
    acc: "99.82%",
    highlight: false,
  },
  {
    rank: "STAGE 2",
    system: "Attention U-Net Scaled (20 ep · f32 · 7.9M + TTA)",
    wt: "90.33%",
    tc: "84.06%",
    et: "81.41%",
    mean: "85.27%",
    bal: "93.92%",
    acc: "99.83%",
    highlight: false,
  },
  {
    rank: "STAGE 3 ★",
    system: "Attention U-Net Heavy (25 ep · f64 · 31.5M + Calib. TTA)",
    wt: "90.80%",
    tc: "85.23%",
    et: "82.76%",
    mean: "86.26%",
    bal: "95.01%",
    acc: "99.83%",
    highlight: true,
  },
  {
    rank: "REF #2",
    system: "NVIDIA SegResNet (Myronenko et al., BraTS 2020 Ref)",
    wt: "91.00%",
    tc: "85.20%",
    et: "81.80%",
    mean: "86.00%",
    bal: "~94.8%",
    acc: "~99.8%",
    highlight: false,
  },
  {
    rank: "REF #1",
    system: "DKFZ nnU-Net (Isensee et al., Nature Methods Ref)",
    wt: "91.30%",
    tc: "85.60%",
    et: "82.40%",
    mean: "86.40%",
    bal: "~95.0%",
    acc: "~99.8%",
    highlight: false,
  },
];

export function Metrics() {
  const { ref, seen } = useInView<HTMLElement>("150px");
  const reduced = useReducedMotion();
  const [traceKey, setTraceKey] = useState<TraceKey>("mean_dice");
  const [activeChannels, setActiveChannels] = useState<Record<ModelRun["id"], boolean>>({
    best_attention_unet: true,
    best_attention_95: true,
    best_attention_heavy: true,
  });
  const [hoverEpoch, setHoverEpoch] = useState<number>(25);
  const [typedRows, setTypedRows] = useState(0);

  useEffect(() => {
    if (!seen) return;
    if (reduced) {
      setTypedRows(BENCHMARK_ROWS.length);
      return;
    }
    setTypedRows(0);
    const iv = setInterval(() => {
      setTypedRows((n) => {
        if (n >= BENCHMARK_ROWS.length) {
          clearInterval(iv);
          return n;
        }
        return n + 1;
      });
    }, 260);
    return () => clearInterval(iv);
  }, [seen, reduced]);

  const mode = TRACE_MODES.find((m) => m.key === traceKey) ?? TRACE_MODES[0];
  const W = 760;
  const H = 300;
  const PAD_L = 54;
  const PAD_R = 24;
  const PAD_T = 22;
  const PAD_B = 34;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const xForEpoch = (ep: number) => PAD_L + ((ep - 1) / 24) * plotW;
  const yForVal = (v: number) => {
    const t = (v - mode.min) / (mode.max - mode.min || 1);
    const clamped = Math.max(0, Math.min(1, t));
    return PAD_T + (1 - clamped) * plotH;
  };

  const heavySeries = (historyData.best_attention_heavy as EpochPoint[]) ?? [];
  const inspectedPoint =
    heavySeries.find((p) => p.epoch === hoverEpoch) ?? heavySeries[heavySeries.length - 1];

  return (
    <section id="metrics" ref={ref} aria-labelledby="metrics-heading" className="w-full px-4 py-16 sm:px-8">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-[#2e3128] pb-4">
        <div>
          <span className="stencil text-xs text-amber">STATION 05 // TELEMETRY &amp; ABLATION LOGS</span>
          <h2 id="metrics-heading" className="font-term text-4xl tracking-tight glow-phos sm:text-5xl">
            OSCILLOSCOPE CONVERGENCE TRACES
          </h2>
        </div>

        {/* Trace Metric Selector */}
        <div role="radiogroup" aria-label="Oscilloscope trace metric" className="flex flex-wrap gap-1.5">
          {TRACE_MODES.map((m) => {
            const on = m.key === traceKey;
            return (
              <button
                key={m.key}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => {
                  click(1.1);
                  setTraceKey(m.key);
                }}
                className={`stencil rounded-[3px] border px-2.5 py-1.5 text-[10px] ${
                  on
                    ? "border-phos bg-phos text-crt shadow-phos"
                    : "border-[#2e3128] bg-crt-2 text-[#9aa08c] hover:text-phos"
                }`}
              >
                {m.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Oscilloscope + Channel Readout Grid */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        {/* Oscilloscope Screen (8 cols) */}
        <div className="lg:col-span-8">
          <Panel label={`TEK-CRT DUAL-BEAM OSCILLOSCOPE // ${mode.label} (EPOCHS 01–25)`} code="OSC-1">
            <div className="glass relative overflow-hidden rounded-[6px] border border-[#2e3128] p-3">
              <svg
                viewBox={`0 0 ${W} ${H}`}
                className="h-auto w-full select-none"
                role="img"
                aria-label={`Oscilloscope chart showing ${mode.label} across 25 training epochs`}
              >
                {/* Graticule grid */}
                {Array.from({ length: 6 }).map((_, i) => {
                  const y = PAD_T + (i / 5) * plotH;
                  const val = mode.max - (i / 5) * (mode.max - mode.min);
                  const label = mode.unit === "%" ? `${(val * 100).toFixed(1)}%` : val.toFixed(2);
                  return (
                    <g key={i}>
                      <line x1={PAD_L} y1={y} x2={W - PAD_R} y2={y} stroke="#1f3a26" strokeWidth="1" />
                      <text x={PAD_L - 8} y={y + 4} textAnchor="end" fill="#4fae68" fontSize="10" fontFamily="monospace">
                        {label}
                      </text>
                    </g>
                  );
                })}
                {[1, 5, 10, 15, 20, 25].map((ep) => {
                  const x = xForEpoch(ep);
                  return (
                    <g key={ep}>
                      <line x1={x} y1={PAD_T} x2={x} y2={H - PAD_B} stroke="#1f3a26" strokeWidth="1" />
                      <text x={x} y={H - 12} textAnchor="middle" fill="#4fae68" fontSize="10" fontFamily="monospace">
                        EP {String(ep).padStart(2, "0")}
                      </text>
                    </g>
                  );
                })}

                {/* Channel Waveforms */}
                {MODEL_RUNS.map((run) => {
                  if (!activeChannels[run.id]) return null;
                  const pts = (historyData[run.id] as EpochPoint[]) ?? [];
                  const d = pts
                    .map((p, idx) => `${idx === 0 ? "M" : "L"}${xForEpoch(p.epoch).toFixed(1)},${yForVal(p[traceKey]).toFixed(1)}`)
                    .join(" ");
                  return (
                    <g key={run.id}>
                      <path
                        d={d}
                        fill="none"
                        stroke={run.color}
                        strokeWidth={run.id === "best_attention_heavy" ? "2.6" : "1.8"}
                        style={{ filter: `drop-shadow(0 0 5px ${run.color})` }}
                      />
                      {pts.map((p) => (
                        <circle
                          key={p.epoch}
                          cx={xForEpoch(p.epoch)}
                          cy={yForVal(p[traceKey])}
                          r={p.epoch === hoverEpoch ? 4.5 : 2.3}
                          fill={run.color}
                          className="cursor-pointer"
                          onMouseEnter={() => setHoverEpoch(p.epoch)}
                        />
                      ))}
                    </g>
                  );
                })}

                {/* Vertical Cursor Line */}
                <line
                  x1={xForEpoch(hoverEpoch)}
                  y1={PAD_T}
                  x2={xForEpoch(hoverEpoch)}
                  y2={H - PAD_B}
                  stroke="#ffb000"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
              </svg>

              {/* Channel Toggle Strip */}
              <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-[#1f3a26] pt-2">
                <div className="flex flex-wrap items-center gap-2">
                  {MODEL_RUNS.map((run) => {
                    const on = activeChannels[run.id];
                    return (
                      <button
                        key={run.id}
                        type="button"
                        aria-pressed={on}
                        onClick={() => {
                          click();
                          setActiveChannels((s) => ({ ...s, [run.id]: !s[run.id] }));
                        }}
                        className={`stencil flex items-center gap-2 rounded-[2px] border px-2.5 py-1 text-[10px] ${
                          on ? "border-phos bg-crt text-phos" : "border-[#2e3128] text-[#6d7265]"
                        }`}
                      >
                        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: run.color }} />
                        {run.channel}: {run.label} ({run.params})
                      </button>
                    );
                  })}
                </div>
                <span className="font-mono text-[11px] text-amber">
                  CURSOR: EPOCH {String(hoverEpoch).padStart(2, "0")} · LOSS {inspectedPoint?.train_loss.toFixed(4)}
                </span>
              </div>
            </div>
          </Panel>
        </div>

        {/* Right Column: Champion 31.5M Model 7-Seg Summary (4 cols) */}
        <div className="flex flex-col gap-5 lg:col-span-4">
          <Panel label="STAGE 3 HEAVY (31.5M) VALIDATION LOCK" code="OSC-2" className="h-full">
            <div className="grid grid-cols-2 gap-3">
              <SevenSeg label="BALANCED ACC" unit="%" value="95.01" digits={5} color="phos" />
              <SevenSeg label="VOXEL ACC" unit="%" value="99.83" digits={5} color="phos" />
              <SevenSeg label="WHOLE TUMOR" unit="F1 %" value="90.80" digits={5} color="amber" />
              <SevenSeg label="TUMOR CORE" unit="F1 %" value="85.23" digits={5} color="amber" />
              <SevenSeg label="ENHANCING ET" unit="F1 %" value="82.76" digits={5} color="hot" />
              <SevenSeg label="JOINT F1" unit="%" value="86.26" digits={5} color="phos" />
            </div>

            <div className="mt-4 rounded-[3px] border border-[#2e3128] bg-[#0a0c09] p-3 font-mono text-xs text-[#c9cdbb]">
              <div className="flex items-center justify-between text-phos">
                <span>CHECKPOINT: best_attention_heavy.pt</span>
                <Led on color="phos" />
              </div>
              <p className="mt-1.5 text-[11px] leading-relaxed text-[#9aa08c]">
                Trained 25 epochs on NVIDIA RTX Pro 6000 (55-case held-out validation split,
                TTA ×3 + calibrated sub-region thresholds).
              </p>
            </div>
          </Panel>
        </div>
      </div>

      {/* Printed Line-by-Line Typewriter Results Table */}
      <div className="mt-6">
        <Panel label="TELETYPE BENCHMARK READOUT // INTERNAL VAL SPLIT VS PUBLISHED BRATS 2020 REFS" code="TTY-01">
          <div className="overflow-x-auto rounded-[4px] border border-[#2e3128] bg-[#090b08] p-4">
            <table className="w-full min-w-[760px] border-collapse text-left font-mono text-xs">
              <thead>
                <tr className="border-b border-phos-faint text-[11px] text-amber">
                  <th className="py-2 pr-3">STAGE / REF</th>
                  <th className="py-2 pr-3">ARCHITECTURE &amp; CONFIGURATION</th>
                  <th className="py-2 pr-3">WT F1</th>
                  <th className="py-2 pr-3">TC F1</th>
                  <th className="py-2 pr-3">ET F1</th>
                  <th className="py-2 pr-3">MEAN F1</th>
                  <th className="py-2 pr-3">BAL. ACC</th>
                  <th className="py-2">VOX. ACC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1a2219]">
                {BENCHMARK_ROWS.slice(0, typedRows).map((r) => (
                  <tr
                    key={r.rank}
                    className={
                      r.highlight
                        ? "bg-phos-faint/60 font-semibold text-phos"
                        : "text-[#c9cdbb]"
                    }
                  >
                    <td className="py-2.5 pr-3 font-term text-base text-amber">{r.rank}</td>
                    <td className="py-2.5 pr-3">{r.system}</td>
                    <td className="py-2.5 pr-3">{r.wt}</td>
                    <td className="py-2.5 pr-3">{r.tc}</td>
                    <td className="py-2.5 pr-3 text-amber">{r.et}</td>
                    <td className="py-2.5 pr-3">{r.mean}</td>
                    <td className="py-2.5 pr-3">{r.bal}</td>
                    <td className="py-2.5">{r.acc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </section>
  );
}

