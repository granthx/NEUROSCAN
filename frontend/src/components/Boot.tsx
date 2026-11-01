"use client";
/**
 * CRT Electron-Gun Ignition, BIOS Boot Sequence, and CRT System Shutdown Collapse.
 *
 * - On initial page open / reload:
 *   Automatically plays the full CRT power-on animation (no button click required):
 *   1. Electron-gun phosphor dot ignites in center
 *   2. Horizontal deflection sweep across 100vw
 *   3. Vertical yoke expansion across 100vh
 *   4. Terminal BIOS boot lines type in line-by-line
 *   5. Phosphor bloom opens the website!
 *
 * - On manual Power Off ("⏻ PWR"):
 *   Collapses full screen -> horizontal line -> phosphor dot -> Halted screen with "⏻ Power on" button.
 *   Clicking "⏻ Power on" triggers the full electron-gun dot -> sweep -> boot sequence!
 */
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { useViewer } from "@/lib/store";
import { useReducedMotion } from "@/lib/hooks";
import { click } from "@/lib/sound";
import { PROJECT_NAME } from "@/lib/config";

const BOOT_LINES: { t: string; ok?: string; warn?: boolean }[] = [
  { t: `${PROJECT_NAME} CONSOLE BIOS v2.7  (C) 1989-2026` },
  { t: "MEMORY TEST ........................ 640K", ok: "OK" },
  { t: "CRT DEFLECTION YOKE ................", ok: "OK" },
  { t: "INITIALIZING GANTRY ................", ok: "OK" },
  { t: "SPINNING UP BORE MAGNET (1.5T) .....", ok: "OK" },
  { t: "LOADING BraTS 2020 DATASET ......... 369 CASES", ok: "OK" },
  { t: "MOUNTING CHANNELS  T1 · T1ce · T2 · FLAIR", ok: "OK" },
  { t: "LOADING ATTN-UNET f64 (31.5M PARAMS)", ok: "OK" },
  { t: "CALIBRATING PHOSPHOR ...............", ok: "OK" },
  { t: "NOTICE: RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS", warn: true },
  { t: "READY." },
];

type BootPhase = "dot" | "line" | "expand" | "typing" | "bloom";

export function BootSequence() {
  const { booted, setBooted } = useViewer();
  const reduced = useReducedMotion();
  const [shown, setShown] = useState(0);
  const [phase, setPhase] = useState<BootPhase>("dot");
  const done = useRef(false);

  const finish = () => {
    if (done.current) return;
    done.current = true;
    setPhase("bloom");
    setTimeout(() => {
      setBooted(true);
      const main = document.querySelector("main");
      if (main) {
        main.classList.remove("signal-lock");
        void (main as HTMLElement).offsetWidth;
        main.classList.add("signal-lock");
      }
    }, reduced ? 0 : 600);
  };

  // Automatically play Dot -> Line -> Expand -> Typing whenever booted is false (on open/reload or reboot)
  useEffect(() => {
    if (booted) return;
    done.current = false;
    setShown(0);

    if (reduced) {
      setBooted(true);
      return;
    }

    // 1. Start with Electron-Gun Dot
    setPhase("dot");
    // 2. Horizontal deflection sweep
    const t1 = setTimeout(() => {
      click(1.2);
      setPhase("line");
    }, 280);
    // 3. Vertical yoke expansion
    const t2 = setTimeout(() => {
      click(1.45);
      setPhase("expand");
    }, 620);
    // 4. Terminal BIOS boot lines
    const t3 = setTimeout(() => {
      setPhase("typing");
    }, 1000);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [booted, reduced, setBooted]);

  // Line-by-line BIOS typing effect
  useEffect(() => {
    if (booted || phase !== "typing") return;
    if (shown >= BOOT_LINES.length) {
      const t = setTimeout(finish, 450);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      setShown((n) => n + 1);
      click(1.6);
    }, shown === 0 ? 180 : 105 + Math.random() * 95);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [booted, phase, shown]);

  // Keyboard shortcut: ESC/Enter/Space skips boot
  useEffect(() => {
    if (booted) return;
    const skip = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === "Enter" || e.key === " ") finish();
    };
    window.addEventListener("keydown", skip);
    return () => window.removeEventListener("keydown", skip);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [booted]);

  const isIgniting = phase === "dot" || phase === "line" || phase === "expand";
  const isBooting = phase === "typing" || phase === "bloom";

  return (
    <AnimatePresence>
      {!booted && (
        <motion.div
          key="boot"
          className="fixed inset-0 z-[90] flex items-center justify-center bg-black"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          role="dialog"
          aria-label="System boot sequence"
        >
          {/* PHASE 1: CRT Electron-Gun Ignition (Dot -> Horizontal Sweep -> Vertical Expansion) */}
          {isIgniting && (
            <motion.div
              className="bg-[#eaffef]"
              style={{
                boxShadow: "0 0 50px 14px #7cff9b, 0 0 140px 45px #7cff9b88",
              }}
              initial={{ width: 2, height: 2, borderRadius: 2, opacity: 0 }}
              animate={
                phase === "dot"
                  ? { width: 12, height: 12, borderRadius: 12, opacity: 1 }
                  : phase === "line"
                    ? { width: "100vw", height: 4, borderRadius: 2, opacity: 1 }
                    : { width: "100vw", height: "100vh", borderRadius: 0, opacity: 1 }
              }
              transition={{
                duration: phase === "dot" ? 0.24 : phase === "line" ? 0.32 : 0.38,
                ease: [0.16, 1, 0.3, 1],
              }}
            />
          )}

          {/* PHASE 2: BIOS Boot Sequence & Phosphor Screen Bloom */}
          {isBooting && (
            <motion.div
              className="glass relative h-full w-full overflow-hidden"
              initial={{ opacity: 1, scale: 1 }}
              animate={
                phase === "typing"
                  ? { scale: 1, opacity: 1, filter: "brightness(1)" }
                  : { scale: 1.08, opacity: 0, filter: "brightness(3)" }
              }
              transition={{ duration: phase === "bloom" ? 0.55 : 0.25, ease: [0.2, 0.8, 0.2, 1] }}
            >
              <div className="mx-auto flex h-full max-w-3xl flex-col justify-center px-6 font-term text-xl leading-snug sm:text-2xl">
                <pre className="whitespace-pre-wrap font-term" aria-live="polite">
                  {BOOT_LINES.slice(0, shown).map((l, i) => (
                    <div key={i} className={l.warn ? "glow-amber" : "glow-phos"}>
                      {l.t} {l.ok && <span className="bg-phos px-1 text-crt">{l.ok}</span>}
                    </div>
                  ))}
                  {shown < BOOT_LINES.length && <span className="animate-blink glow-phos">█</span>}
                </pre>
              </div>
              <button
                type="button"
                onClick={finish}
                className="stencil absolute bottom-8 right-8 rounded-[3px] border border-phos-dim px-3 py-2 text-xs text-phos hover:bg-phos hover:text-crt"
              >
                Skip boot [ESC]
              </button>
            </motion.div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/**
 * Full-screen CRT Power-Off Collapse:
 * Collapses full screen -> horizontal line -> glowing dot -> Halted state with "⏻ Power on" button.
 * Clicking "⏻ Power on" hands off to BootSequence which plays the full Dot -> Sweep -> BIOS Boot animation!
 */
export function Shutdown() {
  const { shuttingDown, setShuttingDown, setBooted } = useViewer();
  const reduced = useReducedMotion();
  const [stage, setStage] = useState(0);

  useEffect(() => {
    if (!shuttingDown) {
      setStage(0);
      return;
    }
    if (reduced) {
      setStage(3);
      return;
    }
    setStage(1);
    const a = setTimeout(() => setStage(2), 380);
    const b = setTimeout(() => setStage(3), 900);
    return () => {
      clearTimeout(a);
      clearTimeout(b);
    };
  }, [shuttingDown, reduced]);

  const reboot = () => {
    click(0.85);
    window.scrollTo({ top: 0 });
    setBooted(false);
    setShuttingDown(false);
  };

  return (
    <AnimatePresence>
      {shuttingDown && (
        <motion.div
          key="off"
          className="fixed inset-0 z-[95] flex items-center justify-center bg-black"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="bg-[#eaffef]"
            style={{ boxShadow: "0 0 40px 10px #7cff9b, 0 0 120px 30px #7cff9b55" }}
            initial={{ width: "100vw", height: "100vh", borderRadius: 0, opacity: 0.9 }}
            animate={
              stage === 1
                ? { width: "100vw", height: 3, borderRadius: 2, opacity: 1 }
                : stage === 2
                  ? { width: 8, height: 8, borderRadius: 8, opacity: 1 }
                  : stage === 3
                    ? { width: 2, height: 2, borderRadius: 2, opacity: 0 }
                    : {}
            }
            transition={{ duration: stage === 3 ? 1.2 : 0.35, ease: "easeIn" }}
          />
          {stage === 3 && (
            <motion.div
              className="absolute flex flex-col items-center gap-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: reduced ? 0 : 1.0 }}
            >
              <p className="font-term text-xl text-phos-dim">SYSTEM HALTED. SAFE TO POWER OFF.</p>
              <button
                type="button"
                onClick={reboot}
                autoFocus
                className="stencil rounded-full border-2 border-phos-dim px-6 py-3 text-sm text-phos hover:bg-phos hover:text-crt"
              >
                ⏻ Power on
              </button>
            </motion.div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
