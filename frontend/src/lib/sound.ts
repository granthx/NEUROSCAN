"use client";
/**
 * Tiny WebAudio synth for console clicks and the power hum.
 * Sound is OFF by default; every call is a no-op unless the user enables it.
 */
import { useViewer } from "./store";

let ctx: AudioContext | null = null;
let hum: { osc: OscillatorNode; gain: GainNode } | null = null;

function audio(): AudioContext | null {
  if (typeof window === "undefined") return null;
  if (!useViewer.getState().sound) return null;
  if (!ctx) {
    const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
  }
  if (ctx.state === "suspended") void ctx.resume();
  return ctx;
}

/** Short mechanical click. `pitch` lets switches and dials sound different. */
export function click(pitch = 1) {
  const a = audio();
  if (!a) return;
  const t = a.currentTime;
  const osc = a.createOscillator();
  const gain = a.createGain();
  osc.type = "square";
  osc.frequency.setValueAtTime(1800 * pitch, t);
  osc.frequency.exponentialRampToValueAtTime(220 * pitch, t + 0.03);
  gain.gain.setValueAtTime(0.06, t);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.045);
  osc.connect(gain).connect(a.destination);
  osc.start(t);
  osc.stop(t + 0.05);
}

/** Low mains hum of the scanner; call with false to stop. */
export function setHum(on: boolean) {
  const a = audio();
  if (!on || !a) {
    if (hum) {
      hum.gain.gain.setTargetAtTime(0, hum.osc.context.currentTime, 0.1);
      const h = hum;
      setTimeout(() => h.osc.stop(), 400);
      hum = null;
    }
    return;
  }
  if (hum) return;
  const osc = a.createOscillator();
  const gain = a.createGain();
  osc.type = "sawtooth";
  osc.frequency.value = 50;
  const filter = a.createBiquadFilter();
  filter.type = "lowpass";
  filter.frequency.value = 180;
  gain.gain.value = 0;
  gain.gain.setTargetAtTime(0.025, a.currentTime, 0.4);
  osc.connect(filter).connect(gain).connect(a.destination);
  osc.start();
  hum = { osc, gain };
}

