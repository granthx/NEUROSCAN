import { BootSequence, Shutdown } from "@/components/Boot";
import { CRTOverlay, Cursor, SmoothScroll, StatusBar } from "@/components/Console";
import { EthicsAndFooter } from "@/components/EthicsAndFooter";
import { Hero } from "@/components/Hero";
import { Metrics } from "@/components/Metrics";
import { Pipeline } from "@/components/Pipeline";
import { Reconstruction } from "@/components/Reconstruction";
import { SliceViewer } from "@/components/SliceViewer";

export default function HomePage() {
  return (
    <>
      <BootSequence />
      <Shutdown />
      <CRTOverlay />
      <Cursor />
      <SmoothScroll />
      <StatusBar />

      <main className="relative z-10">
        <Hero />
        <SliceViewer />
        <Reconstruction />
        <Pipeline />
        <Metrics />
        <EthicsAndFooter />
      </main>
    </>
  );
}

