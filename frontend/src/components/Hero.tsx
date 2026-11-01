"use client";
/**
 * SECTION 01 — FULL-VIEWPORT GANTRY HERO
 * Stationary by default so the full anatomical body and brain stay clearly
 * visible. Includes left-click drag rotation directly on the 3D viewport and
 * a dedicated 3D Spin Trackball panel in the console strip.
 */
import dynamic from "next/dynamic";
import { useEffect, useRef, useState, type PointerEvent } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { frameState } from "@/lib/store";
import { PROJECT_NAME, PROJECT_TAGLINE } from "@/lib/config";
import { useInView, useReducedMotion } from "@/lib/hooks";
import { Led, Panel } from "./hardware";
import { scrollToId } from "./Console";
import { click } from "@/lib/sound";

gsap.registerPlugin(ScrollTrigger);

const HeroScene = dynamic(() => import("./three/HeroScene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center font-term text-xl text-phos-dim">
      ENERGIZING GANTRY COILS...
    </div>
  ),
});

const ZONES = [
  { until: 0.2, name: "LOWER EXTREMITY", status: "NOMINAL" },
  { until: 0.38, name: "LUMBAR / PELVIC", status: "NOMINAL" },
  { until: 0.54, name: "THORACIC RIBCAGE", status: "NOMINAL" },
  { until: 0.65, name: "CERVICAL SPINE", status: "APPROACHING CRANIUM" },
  { until: 1.01, name: "CRANIAL ISOCENTER · R-PARIETAL CORTEX", status: "GLIOMA LOCKED" },
];

export function Hero() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const pinRef = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  const { ref: viewRef, seen } = useInView<HTMLDivElement>("300px");
  const [progress, setProgress] = useState(0);
  const [angles, setAngles] = useState({ yaw: 0, pitch: 0 });
  const dragStart = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);

  useEffect(() => {
    if (reduced) {
      frameState.heroProgress = 0.88;
      setProgress(0.88);
      return;
    }
    const wrap = wrapRef.current;
    const pin = pinRef.current;
    if (!wrap || !pin) return;

    const st = ScrollTrigger.create({
      trigger: wrap,
      start: "top top",
      end: "+=250%",
      pin: pin,
      scrub: 0.35,
      onUpdate: (self) => {
        frameState.heroProgress = self.progress;
        setProgress(self.progress);
      },
    });
    return () => st.kill();
  }, [reduced]);

  const onDragDown = (e: PointerEvent<HTMLElement>) => {
    if (e.button !== 0) return; // left-click only
    e.currentTarget.setPointerCapture(e.pointerId);
    frameState.isDraggingHero = true;
    dragStart.current = {
      x: e.clientX,
      y: e.clientY,
      yaw: frameState.dragYaw,
      pitch: frameState.dragPitch,
    };
  };

  const onDragMove = (e: PointerEvent<HTMLElement>) => {
    if (!dragStart.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    const nextYaw = dragStart.current.yaw + dx * 0.01;
    const nextPitch = Math.max(-0.85, Math.min(0.85, dragStart.current.pitch + dy * 0.008));
    frameState.dragYaw = nextYaw;
    frameState.dragPitch = nextPitch;
    setAngles({
      yaw: Math.round((nextYaw * 180) / Math.PI),
      pitch: Math.round((nextPitch * 180) / Math.PI),
    });
  };

  const onDragUp = () => {
    dragStart.current = null;
    frameState.isDraggingHero = false;
  };

  const resetSpin = () => {
    click(1.1);
    frameState.dragYaw = 0;
    frameState.dragPitch = 0;
    setAngles({ yaw: 0, pitch: 0 });
  };

  const mm = Math.round(Math.min(1, progress / 0.68) * 1840);
  const zone = ZONES.find((z) => progress <= z.until) ?? ZONES[ZONES.length - 1];
  const locked = progress > 0.64;

  return (
    <section id="hero" ref={wrapRef} aria-label="Full-body gantry scan" className="relative w-full">
      <div
        ref={(el) => {
          pinRef.current = el;
          viewRef.current = el;
        }}
        className="glass relative flex h-screen w-screen flex-col justify-between overflow-hidden pt-16 pb-4 px-4 sm:px-8 select-none"
      >
        {/* Interactive Left-Click-to-Spin 3D Gantry & Brain Canvas */}
        <div
          onPointerDown={onDragDown}
          onPointerMove={onDragMove}
          onPointerUp={onDragUp}
          onPointerCancel={onDragUp}
          title="Left-click and drag to spin 3D model"
          className="absolute inset-0 cursor-grab active:cursor-grabbing"
        >
          {seen && <HeroScene />}
        </div>

        {/* Top Telemetry Ruler Strip */}
        <div
          aria-hidden
          className="pointer-events-none relative z-10 flex items-center justify-between border-b border-phos-faint/80 bg-crt/55 px-3 py-1.5 font-term text-sm text-phos-dim backdrop-blur-[2px]"
        >
          <span>GANTRY BORE // 1.5 TESLA SUPERCONDUCTING TOMOGRAPH</span>
          <span className="hidden md:inline text-phos">
            3D GIMBAL: YAW {angles.yaw}° · PITCH {angles.pitch}° (LEFT-CLICK &amp; DRAG TO SPIN)
          </span>
          <span className={locked ? "glow-amber" : "text-phos"}>
            Z-POS {String(mm).padStart(4, "0")} / 1840 MM
          </span>
        </div>

        {/* Middle HUD: Left Headline + Right Cranial Target Lock Callout */}
        <div className="pointer-events-none relative z-10 my-auto flex flex-col justify-between gap-6 lg:flex-row lg:items-center">
          <div
            className={`max-w-xl transition-all duration-300 ${
              locked ? "lg:max-w-md lg:opacity-90" : "lg:max-w-xl"
            }`}
          >
            <div className="mb-2 inline-flex items-center gap-2 rounded-[2px] border border-phos-dim/50 bg-crt/85 px-2.5 py-1">
              <Led on color={locked ? "amber" : "phos"} blink={locked} />
              <span className="stencil text-[11px] text-phos">
                {zone.name} — {zone.status}
              </span>
            </div>

            <h1 className="aberrate font-term text-5xl leading-[0.9] tracking-tight glow-phos sm:text-7xl lg:text-8xl">
              {PROJECT_NAME}
              <span className="animate-blink ml-1 text-amber">█</span>
            </h1>

            <p className="mt-3 max-w-lg rounded-[3px] border border-[#2e3128]/80 bg-crt/80 p-2.5 font-mono text-xs leading-relaxed text-[#c9cdbb] backdrop-blur-xs sm:text-sm">
              {PROJECT_TAGLINE}. Four-channel MRI ingestion (<span className="text-phos">T1</span>,{" "}
              <span className="text-phos">T1ce</span>, <span className="text-phos">T2</span>,{" "}
              <span className="text-phos">FLAIR</span>) with 31.5M-parameter Attention U-Net delineation of{" "}
              <span className="text-amber">peritumoral edema</span>,{" "}
              <span className="text-hot">enhancing core</span>, and{" "}
              <span className="text-necro">necrotic tissue</span>.
            </p>

            <div className="pointer-events-auto mt-4 flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => {
                  click(1.1);
                  scrollToId("viewer");
                }}
                className="stencil rounded-[3px] border border-phos bg-phos px-4 py-2 text-xs font-bold text-crt shadow-phos hover:bg-amber hover:border-amber"
              >
                ▶ Open Slice Console
              </button>
              <button
                type="button"
                onClick={() => {
                  click(0.95);
                  frameState.heroProgress = locked ? 0.15 : 0.95;
                  setProgress(locked ? 0.15 : 0.95);
                }}
                className="stencil rounded-[3px] border border-amber bg-crt/85 px-4 py-2 text-xs text-amber hover:bg-amber hover:text-crt"
              >
                {locked ? "↺ Reset Full-Body View" : "⌖ Zoom to Brain Target"}
              </button>
            </div>
          </div>

          {/* Right Cranial Sub-Region Readout Card */}
          <div
            className={`transition-all duration-300 lg:w-80 ${
              locked ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"
            }`}
          >
            <div className="rounded-[4px] border border-amber/70 bg-crt/85 p-3.5 backdrop-blur-xs">
              <div className="flex items-center justify-between border-b border-phos-faint pb-2">
                <span className="stencil text-[11px] text-amber">CRANIAL LOCK // R-PARIETAL LOBE</span>
                <Led on color="hot" blink />
              </div>
              <dl className="mt-2.5 space-y-1.5 font-mono text-xs">
                <div className="flex items-center justify-between">
                  <dt className="flex items-center gap-2 text-[#c9cdbb]">
                    <span className="h-2.5 w-2.5 rounded-xs bg-amber" /> Edema Halo (L2)
                  </dt>
                  <dd className="font-term text-lg text-amber">WT 90.80% F1</dd>
                </div>
                <div className="flex items-center justify-between">
                  <dt className="flex items-center gap-2 text-[#c9cdbb]">
                    <span className="h-2.5 w-2.5 rounded-xs bg-hot" /> Enhancing Rim (L3)
                  </dt>
                  <dd className="font-term text-lg text-hot">ET 82.76% F1</dd>
                </div>
                <div className="flex items-center justify-between">
                  <dt className="flex items-center gap-2 text-[#c9cdbb]">
                    <span className="h-2.5 w-2.5 rounded-xs bg-necro" /> Necrotic Core (L1)
                  </dt>
                  <dd className="font-term text-lg text-necro">TC 85.23% F1</dd>
                </div>
              </dl>
              <div className="mt-2 border-t border-phos-faint pt-1.5 font-mono text-[10px] text-phos-dim">
                SKULL TRANSPARENCY: 84% · LEFT-CLICK &amp; DRAG TO ROTATE BRAIN
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Telemetry & Interactive Controls Strip */}
        <div className="relative z-10 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Panel label="GANTRY TRAVEL" code="G-01" className="!p-3 !pt-6">
            <div className="flex items-baseline justify-between font-term text-2xl glow-phos">
              <span>{String(Math.round(progress * 100)).padStart(3, "0")}%</span>
              <span className="text-sm text-phos-dim">{mm} / 1840 mm</span>
            </div>
            <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-[#090b08]">
              <div
                className="h-full bg-phos transition-[width] duration-75"
                style={{ width: `${Math.round(progress * 100)}%` }}
              />
            </div>
          </Panel>

          {/* Dedicated Left-Click 3D Spin Pad Section */}
          <Panel label="3D SPIN TRACKPAD (LEFT-CLICK DRAG)" code="G-02" className="!p-3 !pt-6">
            <div className="flex items-center justify-between gap-2">
              <div
                onPointerDown={onDragDown}
                onPointerMove={onDragMove}
                onPointerUp={onDragUp}
                onPointerCancel={onDragUp}
                role="slider"
                tabIndex={0}
                aria-label="Left-click and drag to spin the 3D scanner"
                aria-valuenow={angles.yaw}
                className="grid-phos flex flex-1 cursor-grab items-center justify-between rounded-[3px] border border-phos-dim/60 bg-[#080a07] px-2.5 py-1.5 active:cursor-grabbing"
              >
                <span className="stencil text-[10px] text-phos">✥ HOLD LMB &amp; DRAG TO SPIN</span>
                <span className="font-term text-base text-amber">
                  {angles.yaw}° / {angles.pitch}°
                </span>
              </div>
              <button
                type="button"
                onClick={resetSpin}
                className="stencil rounded-[3px] border border-[#2e3128] bg-crt px-2 py-1.5 text-[10px] text-[#c9cdbb] hover:border-phos hover:text-phos"
              >
                0°
              </button>
            </div>
          </Panel>

          <Panel label="TARGET LOCK" code="G-03" className="!p-3 !pt-6">
            <div className={`font-term text-xl ${locked ? "glow-amber" : "text-phos-dim"}`}>
              {locked ? "BRAIN & GLIOMA ISOLATED" : "SCANNING AXIAL STATIONS..."}
            </div>
            <p className="font-mono text-[10px] text-[#8a9080]">
              {locked ? "R-PARIETAL · WT + TC + ET VISIBLE" : "SCROLL TO ADVANCE GANTRY TO BRAIN ↓"}
            </p>
          </Panel>

          <Panel label="MANUAL OVERRIDE" code="G-04" className="!p-3 !pt-6">
            <label className="flex items-center justify-between font-mono text-[10px] text-[#c9cdbb]">
              <span>SCAN / BRAIN ZOOM</span>
              <span>{Math.round(progress * 100)}%</span>
            </label>
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(progress * 100)}
              aria-label="Manual gantry scan and brain zoom position"
              onChange={(e) => {
                const v = Number(e.target.value) / 100;
                frameState.heroProgress = v;
                setProgress(v);
              }}
              className="mt-1.5 w-full accent-[#7cff9b]"
            />
          </Panel>
        </div>
      </div>
    </section>
  );
}
