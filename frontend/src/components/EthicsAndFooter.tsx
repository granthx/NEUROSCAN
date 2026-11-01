"use client";
/**
 * SECTION 06 & 07 — PRINTED ETHICS REPORT CARD & CRT SYSTEM SHUTDOWN FOOTER
 */
import { DISCLAIMER, PROJECT_NAME } from "@/lib/config";
import { useViewer } from "@/lib/store";
import { click } from "@/lib/sound";
import { Led } from "./hardware";

export function EthicsAndFooter() {
  const setShuttingDown = useViewer((s) => s.setShuttingDown);

  return (
    <>
      {/* SECTION 06: Printed Clinical & Ethical Limits Report on Cream Paper */}
      <section id="ethics" aria-labelledby="ethics-heading" className="w-full px-4 py-16 sm:px-8">
        <div className="paper relative rounded-[6px] border-2 border-[#a8a28c] p-6 shadow-2xl sm:p-10">
          {/* Perforated top edge holes */}
          <div aria-hidden className="mb-5 flex justify-between border-b border-dashed border-ink/40 pb-3">
            {Array.from({ length: 14 }).map((_, i) => (
              <span key={i} className="h-3 w-3 rounded-full border border-ink/40 bg-crt/10" />
            ))}
          </div>

          <div className="flex flex-wrap items-start justify-between gap-6">
            <div>
              <div className="stencil text-xs text-ink-dim">
                FORM MED-AI-09 // REGULATORY &amp; SAFETY DISCLOSURE
              </div>
              <h2 id="ethics-heading" className="mt-1 font-term text-4xl tracking-tight text-ink sm:text-5xl">
                ETHICS, INTER-RATER LIMITS &amp; CLINICAL SCOPE
              </h2>
            </div>

            {/* Rubber Stamp */}
            <div
              role="note"
              className="stencil rotate-[-2.5deg] rounded-[3px] border-4 border-stamp px-4 py-2 text-center text-sm font-extrabold tracking-[0.22em] text-stamp shadow-sm"
            >
              <div>RESEARCH PROTOTYPE</div>
              <div className="text-xs">NOT FOR CLINICAL DIAGNOSIS</div>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-1 gap-6 font-mono text-xs leading-relaxed text-ink sm:text-sm lg:grid-cols-3">
            <div className="rounded-[3px] border border-ink/40 bg-paper-2 p-4">
              <div className="stencil mb-1.5 text-xs font-bold text-stamp">
                01 // NON-DIAGNOSTIC STATUS
              </div>
              <p>
                <strong>{DISCLAIMER}</strong> {PROJECT_NAME} is an academic and engineering
                demonstration trained on retrospective research scans (MICCAI BraTS 2020). It has
                not been cleared by the FDA, CE, or CDSCO and must never be used to guide patient
                care, biopsy targeting, or neurosurgical resection margins.
              </p>
            </div>

            <div className="rounded-[3px] border border-ink/40 bg-paper-2 p-4">
              <div className="stencil mb-1.5 text-xs font-bold text-ink">
                02 // RADIOLOGIST INTER-RATER CEILING
              </div>
              <p>
                On glioblastoma MRI, board-certified neuroradiologists exhibit an inter-observer
                Dice agreement of <strong>~82%–85%</strong> on the microscopic enhancing rim (ET).
                Evaluation numbers reported here reflect a 55-patient held-out split from the
                BraTS 2020 training archive.
              </p>
            </div>

            <div className="rounded-[3px] border border-ink/40 bg-paper-2 p-4">
              <div className="stencil mb-1.5 text-xs font-bold text-ink">
                03 // KNOWN ARCHITECTURAL LIMITS
              </div>
              <p>
                While 2D axial slices with Test-Time Augmentation and 3D connected-component
                filtering achieve fast inference and high overlap, out-of-plane anisotropic spacing
                and scanner-specific coil artifacts require site-specific domain calibration before
                clinical translation.
              </p>
            </div>
          </div>

          <div className="mt-6 flex flex-wrap items-center justify-between gap-4 border-t-2 border-ink pt-4 font-mono text-xs text-ink-dim">
            <span>CHECKPOINT VERIFIED: checkpoints/best_attention_heavy.pt (31.5M PARAMS)</span>
            <span>DATASET: RSNA-ASNR-MICCAI BRATS 2020 · CC-BY-NC</span>
          </div>
        </div>
      </section>

      {/* SECTION 07: Footer with CRT System Shutdown Sequence */}
      <footer className="mt-10 border-t border-[#2e3128] bg-crt-2 px-4 py-12 sm:px-6">
        <div className="mx-auto flex max-w-[1480px] flex-col items-center justify-between gap-6 sm:flex-row">
          <div className="flex items-center gap-4">
            <Led on color="phos" />
            <div>
              <div className="font-term text-2xl glow-phos">{PROJECT_NAME} TERMINAL CONSOLE</div>
              <p className="font-mono text-xs text-[#9aa08c]">
                {DISCLAIMER} · Built with Next.js, Three.js / R3F, GLSL &amp; Tailwind v4.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => {
              click(0.55);
              setShuttingDown(true);
            }}
            className="stencil group flex items-center gap-3 rounded-[4px] border-2 border-[#ff5a1f] bg-[#1b110d] px-5 py-3 text-xs font-bold text-[#ff8b6b] shadow-lg transition hover:bg-[#ff5a1f] hover:text-crt"
          >
            <span className="h-2.5 w-2.5 rounded-full bg-[#ff5a1f] group-hover:bg-crt" />
            ⏻ Initiate System Shutdown (CRT Collapse)
          </button>
        </div>
      </footer>
    </>
  );
}

