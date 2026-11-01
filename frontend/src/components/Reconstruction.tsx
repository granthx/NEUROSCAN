"use client";
/**
 * SECTION 03 — 3D VOLUMETRIC TUMOR RECONSTRUCTION
 * Stacks axial contour slices into a rotatable 3D wireframe brain with glowing
 * amber and red-orange tumor sub-volumes and a sweeping laser plane.
 */
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { computeVolumes } from "@/lib/phantom";
import { useInView } from "@/lib/hooks";
import { Fader, Led, Panel, SevenSeg, ToggleSwitch } from "./hardware";

const ReconScene = dynamic(() => import("./three/ReconScene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center font-term text-xl text-phos-dim">
      STACKING AXIAL CONTOURS...
    </div>
  ),
});

export function Reconstruction() {
  const { ref, seen } = useInView<HTMLDivElement>("250px");
  const [assemble, setAssemble] = useState(100);
  const [explode, setExplode] = useState(0);
  const [autoRotate, setAutoRotate] = useState(true);
  const [showBrain, setShowBrain] = useState(true);
  const [showEdema, setShowEdema] = useState(true);
  const [showEnhancing, setShowEnhancing] = useState(true);

  const vols = useMemo(() => computeVolumes(), []);

  return (
    <section id="recon" ref={ref} aria-labelledby="recon-heading" className="w-full px-4 py-16 sm:px-8">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-[#2e3128] pb-4">
        <div>
          <span className="stencil text-xs text-amber">STATION 03 // VOLUMETRIC TOMOGRAPHY</span>
          <h2 id="recon-heading" className="font-term text-4xl tracking-tight glow-phos sm:text-5xl">
            3D CONTOUR STACK RECONSTRUCTION
          </h2>
        </div>
        <p className="max-w-md font-mono text-xs text-[#9aa08c]">
          Drag inside the CRT viewport to orbit the reconstructed brain mesh. Scrub{" "}
          <span className="text-phos">STACK</span> to watch 2D axial contours assemble into the 3D lesion.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        {/* 3D Viewport (8 cols) */}
        <div className="lg:col-span-8">
          <Panel label="ISOMETRIC WIREFRAME HOLOGRAM // DRAG TO ROTATE" code="V-01">
            <div className="glass relative h-[460px] w-full overflow-hidden rounded-[6px] border border-[#2e3128] sm:h-[520px]">
              {seen && (
                <ReconScene
                  assemble={assemble / 100}
                  explode={explode / 100}
                  autoRotate={autoRotate}
                  showBrain={showBrain}
                  showEdema={showEdema}
                  showEnhancing={showEnhancing}
                />
              )}

              {/* Top-left overlay tag */}
              <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 rounded-[2px] border border-[#2e3128] bg-crt/80 px-2.5 py-1 font-mono text-[11px]">
                <Led on color="amber" blink />
                <span className="text-phos">R-PARIETAL GLIOMA</span>
                <span className="text-phos-dim">· {Math.round((assemble / 100) * 38)}/38 SLICES</span>
              </div>

              {/* Bottom legend */}
              <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-wrap items-center justify-between gap-2 bg-crt/85 px-3 py-1.5 font-mono text-[11px]">
                <div className="flex items-center gap-4">
                  <span className="text-phos">● CORTICAL MESH</span>
                  <span className="text-amber">● WHOLE TUMOR (WT)</span>
                  <span className="text-hot">● ENHANCING RIM (ET)</span>
                </div>
                <span className="text-phos-dim">ORBIT: LMB DRAG · ZOOM: WHEEL</span>
              </div>
            </div>
          </Panel>
        </div>

        {/* Right Controls (4 cols) */}
        <div className="flex flex-col gap-5 lg:col-span-4">
          <Panel label="CONTOUR STACK FADERS" code="V-02">
            <div className="flex items-center justify-around py-1">
              <Fader
                label="STACK %"
                value={assemble}
                min={5}
                max={100}
                onChange={setAssemble}
                format={(v) => `${v}% assembled`}
                ticks={10}
              />
              <Fader
                label="EXPLODE"
                value={explode}
                min={0}
                max={100}
                onChange={setExplode}
                format={(v) => `${v}% separation`}
                ticks={10}
              />
            </div>
          </Panel>

          <Panel label="MESH & TURNTABLE SWITCHES" code="V-03">
            <div className="grid grid-cols-4 gap-2 pt-1">
              <ToggleSwitch
                checked={autoRotate}
                onChange={setAutoRotate}
                label="ORBIT"
                sub="AUTO"
                led="phos"
              />
              <ToggleSwitch
                checked={showBrain}
                onChange={setShowBrain}
                label="CORTEX"
                sub="MESH"
                led="phos"
              />
              <ToggleSwitch
                checked={showEdema}
                onChange={setShowEdema}
                label="EDEMA"
                sub="AMBER"
                led="amber"
              />
              <ToggleSwitch
                checked={showEnhancing}
                onChange={setShowEnhancing}
                label="ACTIVE"
                sub="CORE"
                led="hot"
              />
            </div>
          </Panel>

          <Panel label="3D VOLUMETRIC INTEGRATOR" code="V-04">
            <div className="grid grid-cols-2 gap-3">
              <SevenSeg label="3D DICE" unit="%" value={(vols.dice3d * 100).toFixed(1)} digits={5} color="phos" />
              <SevenSeg label="TUMOR CORE" unit="cm³" value={vols.pred.tc.toFixed(1)} digits={5} color="hot" />
            </div>
          </Panel>
        </div>
      </div>
    </section>
  );
}

