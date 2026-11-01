"use client";
/**
 * Global console chrome: CRT overlay layers, crosshair cursor, Lenis smooth
 * scroll wired to GSAP ScrollTrigger, and the fixed top status bar with the
 * research disclaimer + CRT / sound controls.
 */
import { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";
import { useViewer, frameState, type CrtMode } from "@/lib/store";
import { click, setHum } from "@/lib/sound";
import { DISCLAIMER, PROJECT_NAME, SECTIONS } from "@/lib/config";
import { useReducedMotion } from "@/lib/hooks";

gsap.registerPlugin(ScrollTrigger);

/* ------------------------------------------------------------ CRT layers */
export function CRTOverlay() {
  const mode = useViewer((s) => s.crtMode);
  useEffect(() => {
    document.documentElement.dataset.crt = mode;
  }, [mode]);
  return (
    <div aria-hidden>
      <div className="crt-layer crt-scanlines" />
      <div className="crt-layer crt-noise" />
      <div className="crt-layer crt-flicker" />
      <div className="crt-layer crt-vignette" />
      <div className="crt-layer crt-curve" />
    </div>
  );
}

/* ---------------------------------------------------------------- Cursor */
export function Cursor() {
  const ref = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const coordsRef = useRef<HTMLSpanElement>(null);
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    if (!window.matchMedia("(pointer: fine)").matches) return;
    setEnabled(true);
    document.documentElement.dataset.cursor = "on";
    let raf = 0;

    const move = (e: MouseEvent) => {
      frameState.pointerX = (e.clientX / window.innerWidth) * 2 - 1;
      frameState.pointerY = (e.clientY / window.innerHeight) * 2 - 1;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        if (ref.current) {
          ref.current.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`;
          ref.current.style.opacity = "1";
        }
        if (coordsRef.current) {
          coordsRef.current.textContent = `X${String(e.clientX).padStart(4, "0")} Y${String(e.clientY).padStart(4, "0")}`;
        }
      });
    };

    // Scale ONLY the inner SVG around its own exact center (22px, 22px)
    // so the reticle center NEVER drifts or jumps diagonally on click!
    const down = () => {
      if (svgRef.current) {
        svgRef.current.style.transform = "scale(0.82)";
      }
    };
    const up = () => {
      if (svgRef.current) {
        svgRef.current.style.transform = "scale(1)";
      }
    };

    const leave = () => {
      if (ref.current) ref.current.style.opacity = "0";
    };
    const enter = () => {
      if (ref.current) ref.current.style.opacity = "1";
    };

    window.addEventListener("mousemove", move);
    window.addEventListener("mousedown", down);
    window.addEventListener("mouseup", up);
    document.addEventListener("mouseleave", leave);
    document.addEventListener("mouseenter", enter);

    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mousedown", down);
      window.removeEventListener("mouseup", up);
      document.removeEventListener("mouseleave", leave);
      document.removeEventListener("mouseenter", enter);
      delete document.documentElement.dataset.cursor;
    };
  }, []);

  if (!enabled) return null;
  return (
    <div
      ref={ref}
      aria-hidden
      style={{ transform: "translate3d(-100px, -100px, 0)", opacity: 0 }}
      className="pointer-events-none fixed left-0 top-0 z-[120] will-change-transform"
    >
      {/* Exact center alignment: the (0,0) coordinate is positioned precisely at (22, 22) */}
      <div className="relative select-none" style={{ marginLeft: "-22px", marginTop: "-22px" }}>
        <svg
          ref={svgRef}
          width="44"
          height="44"
          viewBox="0 0 44 44"
          style={{ transformOrigin: "22px 22px", transition: "transform 90ms ease-out" }}
          className="drop-shadow-[0_0_4px_#7cff9b]"
        >
          <circle cx="22" cy="22" r="9" fill="none" stroke="#7cff9b" strokeWidth="1" />
          <path d="M22 0v14M22 30v14M0 22h14M30 22h14" stroke="#7cff9b" strokeWidth="1" />
          <circle cx="22" cy="22" r="1.5" fill="#ffb000" />
        </svg>
        <span
          ref={coordsRef}
          className="absolute left-8 top-8 whitespace-nowrap font-term text-sm text-phos-dim pointer-events-none"
        >
          X0000 Y0000
        </span>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------- SmoothScroll */
export function SmoothScroll() {
  const reduced = useReducedMotion();
  useEffect(() => {
    if (reduced) return;
    const lenis = new Lenis({ duration: 1.15, smoothWheel: true });
    lenis.on("scroll", ScrollTrigger.update);
    const tick = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);
    (window as unknown as { __lenis?: Lenis }).__lenis = lenis;
    return () => {
      gsap.ticker.remove(tick);
      lenis.destroy();
      delete (window as unknown as { __lenis?: Lenis }).__lenis;
    };
  }, [reduced]);
  return null;
}

export function scrollToId(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  const lenis = (window as unknown as { __lenis?: Lenis }).__lenis;
  if (lenis) lenis.scrollTo(el, { offset: -40 });
  else el.scrollIntoView({ behavior: "smooth" });
}

/* ------------------------------------------------------------ Status bar */
const MODES: CrtMode[] = ["full", "light", "off"];

export function StatusBar() {
  const { crtMode, setCrtMode, sound, toggleSound, booted, setShuttingDown } = useViewer();
  const [clock, setClock] = useState("--:--:--");
  useEffect(() => {
    const t = setInterval(() => setClock(new Date().toLocaleTimeString("en-GB")), 1000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    setHum(sound && booted);
  }, [sound, booted]);

  return (
    <header className="fixed inset-x-0 top-0 z-[70]">
      {/* Disclaimer banner — always visible */}
      <div role="note" className="flex items-center justify-center gap-2 bg-amber px-3 py-1 text-center font-label text-[11px] font-bold uppercase tracking-[0.2em] text-crt">
        <span aria-hidden>▲</span> {DISCLAIMER} <span aria-hidden>▲</span>
      </div>
      <nav aria-label="Console" className="flex items-center justify-between gap-3 border-b border-[#2e3128] bg-crt/90 px-3 py-1.5 backdrop-blur-sm sm:px-5">
        <div className="flex items-center gap-3">
          <span className="font-term text-xl leading-none glow-phos">{PROJECT_NAME}</span>
          <span className="hidden font-term text-base text-phos-dim md:inline">SYS {clock}</span>
        </div>
        <ul className="hidden items-center gap-1 lg:flex">
          {SECTIONS.map((s) => (
            <li key={s.id}>
              <button
                type="button"
                onClick={() => { click(); scrollToId(s.id); }}
                className="stencil rounded-[2px] px-2 py-1 text-[10px] text-[#9aa08c] hover:bg-phos-faint hover:text-phos"
              >
                <span className="text-phos-dim">{s.code}</span> {s.label}
              </button>
            </li>
          ))}
        </ul>
        <div className="flex items-center gap-2">
          <div role="radiogroup" aria-label="CRT effects" className="flex overflow-hidden rounded-[3px] border border-[#2e3128]">
            {MODES.map((m) => (
              <button
                key={m}
                type="button"
                role="radio"
                aria-checked={crtMode === m}
                onClick={() => { click(); setCrtMode(m); }}
                className={`stencil px-2 py-1 text-[10px] ${crtMode === m ? "bg-phos text-crt" : "text-[#9aa08c] hover:text-phos"}`}
              >
                <span className="sr-only">CRT </span>{m}
              </button>
            ))}
          </div>
          <button
            type="button"
            aria-pressed={sound}
            onClick={() => { toggleSound(); setTimeout(() => click(), 0); }}
            className={`stencil rounded-[3px] border px-2 py-1 text-[10px] ${sound ? "border-amber text-amber" : "border-[#2e3128] text-[#9aa08c]"}`}
          >
            SND {sound ? "ON" : "OFF"}
          </button>
          <button
            type="button"
            onClick={() => { click(0.6); setShuttingDown(true); }}
            className="stencil hidden rounded-[3px] border border-[#5a2a22] px-2 py-1 text-[10px] text-[#ff8b6b] sm:block"
            aria-label="System shutdown"
          >
            ⏻ PWR
          </button>
        </div>
      </nav>
    </header>
  );
}

