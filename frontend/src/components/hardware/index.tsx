"use client";
/**
 * Hardware-style console primitives: panels with screws, LEDs, toggle switches,
 * rotary dials, faders and seven-segment readouts. All are keyboard operable
 * and carry ARIA roles/labels.
 */
import { motion } from "framer-motion";
import { useCallback, useId, useRef, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { click } from "@/lib/sound";

/* ------------------------------------------------------------------ Panel */
export function Screw({ className = "" }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={`absolute h-3 w-3 rounded-full ${className}`}
      style={{
        background: "radial-gradient(circle at 35% 30%, #5a5e52, #23251f 70%)",
        boxShadow: "inset 0 0 0 1px #0a0b09, 0 1px 0 rgb(255 255 255 / 0.05)",
      }}
    >
      <span className="absolute left-1/2 top-1/2 h-[1.5px] w-2 -translate-x-1/2 -translate-y-1/2 rotate-[35deg] bg-[#0d0e0b]" />
    </span>
  );
}

export function Panel({
  children,
  label,
  code,
  className = "",
}: {
  children: ReactNode;
  label?: string;
  code?: string;
  className?: string;
}) {
  return (
    <div className={`bezel relative rounded-[6px] p-4 pt-7 sm:p-5 sm:pt-8 ${className}`}>
      <Screw className="left-2 top-2" />
      <Screw className="right-2 top-2" />
      <Screw className="bottom-2 left-2" />
      <Screw className="bottom-2 right-2" />
      {label && (
        <div className="stencil absolute left-7 top-2 flex items-center gap-2 text-[10px] text-[#9aa08c]">
          {code && <span className="rounded-[2px] bg-[#9aa08c] px-1 text-[#141611]">{code}</span>}
          {label}
        </div>
      )}
      {children}
    </div>
  );
}

/* -------------------------------------------------------------------- LED */
export function Led({ on, color = "phos", blink = false, label }: { on: boolean; color?: "phos" | "amber" | "hot"; blink?: boolean; label?: string }) {
  const c = color === "phos" ? "#7cff9b" : color === "amber" ? "#ffb000" : "#ff5a1f";
  return (
    <span className="inline-flex items-center gap-1.5" role={label ? "status" : undefined} aria-label={label ? `${label} ${on ? "on" : "off"}` : undefined}>
      <span
        aria-hidden
        className={`inline-block h-2.5 w-2.5 rounded-full ${on && blink ? "animate-led" : ""}`}
        style={{
          background: on ? `radial-gradient(circle at 35% 35%, #fff 0%, ${c} 45%, ${c}aa 100%)` : "#23261f",
          boxShadow: on ? `0 0 8px ${c}, 0 0 2px ${c}` : "inset 0 1px 2px #000",
        }}
      />
      {label && <span className="stencil text-[10px] text-[#9aa08c]">{label}</span>}
    </span>
  );
}

/* ---------------------------------------------------------- ToggleSwitch */
export function ToggleSwitch({
  checked,
  onChange,
  label,
  sub,
  led = "phos",
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  sub?: string;
  led?: "phos" | "amber" | "hot";
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={`${label}${sub ? ` (${sub})` : ""}`}
      onClick={() => {
        click(checked ? 0.8 : 1.1);
        onChange(!checked);
      }}
      className="group flex flex-col items-center gap-1.5 rounded-sm p-1"
    >
      <Led on={checked} color={led} />
      {/* bat-handle toggle */}
      <span className="relative h-11 w-6 rounded-[3px] border border-[#0a0b09] bg-gradient-to-b from-[#0f100d] to-[#1d1f19] shadow-[inset_0_2px_4px_#000]">
        <motion.span
          className="absolute left-1/2 h-6 w-3 -translate-x-1/2 rounded-[2px]"
          style={{ background: "linear-gradient(90deg,#8d9283,#d7dccb 45%,#6e7266)", boxShadow: "0 2px 3px #000" }}
          animate={{ top: checked ? 2 : 18 }}
          transition={{ type: "spring", stiffness: 700, damping: 28 }}
        />
      </span>
      <span className="stencil text-[10px] leading-none text-[#c9cdbb]">{label}</span>
      {sub && <span className="font-mono text-[9px] leading-none text-[#8a9080]">{sub}</span>}
    </button>
  );
}

/* ------------------------------------------------------------ RotaryDial */
export function RotaryDial<T extends string>({
  options,
  value,
  onChange,
  label,
  format = (v) => v,
}: {
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  format?: (v: T) => string;
}) {
  const idx = options.indexOf(value);
  const sweep = 240; // degrees covered by the detents
  const angleFor = (i: number) => -sweep / 2 + (sweep / Math.max(1, options.length - 1)) * i;
  const step = (d: number) => {
    const next = (idx + d + options.length) % options.length;
    click(0.7 + next * 0.1);
    onChange(options[next]);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); step(1); }
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); step(-1); }
    if (e.key === "Home") { e.preventDefault(); onChange(options[0]); }
    if (e.key === "End") { e.preventDefault(); onChange(options[options.length - 1]); }
  };
  return (
    <div className="flex flex-col items-center">
      <div className="relative h-36 w-36">
        {/* detent labels */}
        {options.map((o, i) => {
          const a = (angleFor(i) - 90) * (Math.PI / 180);
          const r = 62;
          return (
            <button
              key={o}
              type="button"
              tabIndex={-1}
              onClick={() => { click(0.7 + i * 0.1); onChange(o); }}
              className={`absolute -translate-x-1/2 -translate-y-1/2 font-term text-lg leading-none ${i === idx ? "glow-amber" : "text-[#8a9080]"}`}
              style={{ left: 72 + Math.cos(a) * r, top: 72 + Math.sin(a) * r }}
            >
              {format(o)}
            </button>
          );
        })}
        {/* knob */}
        <motion.div
          role="slider"
          tabIndex={0}
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={options.length - 1}
          aria-valuenow={idx}
          aria-valuetext={format(value)}
          onKeyDown={onKey}
          onClick={() => step(1)}
          whileTap={{ scale: 0.96 }}
          className="absolute left-1/2 top-1/2 h-[74px] w-[74px] -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{
            background: "conic-gradient(from 0deg,#3a3d34,#1b1d18,#3a3d34,#1b1d18,#3a3d34,#1b1d18,#3a3d34)",
            boxShadow: "0 6px 14px #000, inset 0 0 0 4px #121410, inset 0 2px 0 4px rgb(255 255 255 / 0.05)",
          }}
          animate={{ rotate: angleFor(idx) }}
          transition={{ type: "spring", stiffness: 260, damping: 18 }}
        >
          <span className="absolute left-1/2 top-2 h-5 w-1 -translate-x-1/2 rounded-full bg-amber shadow-amber" />
        </motion.div>
      </div>
      <span className="stencil -mt-1 text-[10px] text-[#c9cdbb]">{label}</span>
    </div>
  );
}

/* ----------------------------------------------------------------- Fader */
export function Fader({
  value,
  min,
  max,
  onChange,
  label,
  orientation = "vertical",
  format = (v) => String(v),
  ticks = 10,
}: {
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  label: string;
  orientation?: "vertical" | "horizontal";
  format?: (v: number) => string;
  ticks?: number;
}) {
  const track = useRef<HTMLDivElement>(null);
  const id = useId();
  const vertical = orientation === "vertical";
  const t = (value - min) / (max - min || 1);
  const last = useRef(value);

  const setFromPointer = useCallback(
    (e: PointerEvent) => {
      const el = track.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const f = vertical ? 1 - (e.clientY - r.top) / r.height : (e.clientX - r.left) / r.width;
      const v = Math.round(min + Math.min(1, Math.max(0, f)) * (max - min));
      if (v !== last.current) {
        last.current = v;
        if (v % 5 === 0) click(1.4);
        onChange(v);
      }
    },
    [vertical, min, max, onChange],
  );

  const onKey = (e: KeyboardEvent) => {
    const big = Math.max(1, Math.round((max - min) / 10));
    const map: Record<string, number> = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: big, PageDown: -big };
    if (e.key in map) {
      e.preventDefault();
      onChange(Math.min(max, Math.max(min, value + map[e.key])));
      click(1.3);
    } else if (e.key === "Home") { e.preventDefault(); onChange(min); }
    else if (e.key === "End") { e.preventDefault(); onChange(max); }
  };

  return (
    <div className={`flex ${vertical ? "flex-col" : "flex-col"} items-center gap-2`}>
      <div
        ref={track}
        onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); setFromPointer(e); }}
        onPointerMove={(e) => { if (e.buttons) setFromPointer(e); }}
        className={`relative touch-none ${vertical ? "h-56 w-14" : "h-12 w-full"}`}
      >
        {/* tick ruler */}
        {Array.from({ length: ticks + 1 }).map((_, i) => (
          <span
            key={i}
            aria-hidden
            className="absolute bg-[#5d6354]"
            style={vertical
              ? { left: i % 5 === 0 ? 2 : 6, width: i % 5 === 0 ? 10 : 6, height: 1, top: `${(i / ticks) * 100}%` }
              : { top: i % 5 === 0 ? 2 : 6, height: i % 5 === 0 ? 10 : 6, width: 1, left: `${(i / ticks) * 100}%` }}
          />
        ))}
        {/* slot */}
        <span
          aria-hidden
          className={`absolute rounded-full bg-[#050605] shadow-[inset_0_1px_3px_#000] ${vertical ? "left-1/2 top-0 h-full w-1.5 -translate-x-1/2" : "left-0 top-1/2 h-1.5 w-full -translate-y-1/2"}`}
        />
        {/* cap */}
        <motion.div
          role="slider"
          tabIndex={0}
          id={id}
          aria-label={label}
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={value}
          aria-valuetext={format(value)}
          aria-orientation={orientation}
          onKeyDown={onKey}
          whileTap={{ scale: 1.05 }}
          className={`absolute rounded-[3px] ${vertical ? "left-1/2 h-7 w-12 -translate-x-1/2 -translate-y-1/2" : "top-1/2 h-10 w-6 -translate-x-1/2 -translate-y-1/2"}`}
          style={{
            [vertical ? "top" : "left"]: `${(vertical ? 1 - t : t) * 100}%`,
            background: "linear-gradient(180deg,#d9ddcc 0%,#a3a897 40%,#6f7366 100%)",
            boxShadow: "0 4px 8px #000, inset 0 1px 0 #fff8",
          }}
        >
          <span className={`absolute bg-[#141611] ${vertical ? "left-1 right-1 top-1/2 h-[2px] -translate-y-1/2" : "bottom-1 top-1 left-1/2 w-[2px] -translate-x-1/2"}`} />
        </motion.div>
      </div>
      <label htmlFor={id} className="stencil text-[10px] text-[#c9cdbb]">{label}</label>
    </div>
  );
}

/* ------------------------------------------------------- Seven segment */
const SEG: Record<string, string> = {
  "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
  "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcfgd", "-": "g", " ": "", E: "afged", r: "eg", n: "ceg", A: "abcefg",
};
function Digit({ ch, dot, color }: { ch: string; dot: boolean; color: string }) {
  const on = SEG[ch] ?? "";
  const off = "#1e231c";
  const s = (k: string) => (on.includes(k) ? color : off);
  const glow = on ? `drop-shadow(0 0 3px ${color})` : undefined;
  return (
    <svg viewBox="0 0 24 40" className="h-full w-auto" aria-hidden style={{ filter: glow }}>
      <polygon points="5,2 19,2 16,5 8,5" fill={s("a")} />
      <polygon points="20,3 20,19 17,17 17,6" fill={s("b")} />
      <polygon points="20,21 20,37 17,34 17,23" fill={s("c")} />
      <polygon points="5,38 19,38 16,35 8,35" fill={s("d")} />
      <polygon points="4,21 4,37 7,34 7,23" fill={s("e")} />
      <polygon points="4,3 4,19 7,17 7,6" fill={s("f")} />
      <polygon points="5,20 8,18 16,18 19,20 16,22 8,22" fill={s("g")} />
      <circle cx="22.5" cy="37.5" r="1.6" fill={dot ? color : off} />
    </svg>
  );
}

export function SevenSeg({
  value,
  digits = 5,
  label,
  unit,
  color = "amber",
}: {
  value: string;
  digits?: number;
  label: string;
  unit?: string;
  color?: "amber" | "phos" | "hot";
}) {
  const c = color === "amber" ? "#ffb000" : color === "phos" ? "#7cff9b" : "#ff5a1f";
  // split "12.34" into chars with dot flags
  const chars: { ch: string; dot: boolean }[] = [];
  for (const ch of value) {
    if (ch === "." && chars.length) chars[chars.length - 1].dot = true;
    else chars.push({ ch, dot: false });
  }
  while (chars.length < digits) chars.unshift({ ch: " ", dot: false });
  return (
    <div className="flex flex-col gap-1" role="group" aria-label={`${label}: ${value.trim()} ${unit ?? ""}`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="stencil text-[10px] text-[#c9cdbb]">{label}</span>
        {unit && <span className="font-mono text-[10px] text-[#8a9080]">{unit}</span>}
      </div>
      <div className="flex h-10 items-center gap-[3px] rounded-[3px] border border-[#050605] bg-[#0a0c09] px-2 py-1.5 shadow-[inset_0_2px_6px_#000]" aria-hidden>
        {chars.slice(-digits).map((d, i) => (
          <Digit key={i} ch={d.ch} dot={d.dot} color={c} />
        ))}
      </div>
    </div>
  );
}

