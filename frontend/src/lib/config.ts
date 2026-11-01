/**
 * Site-wide constants. Change PROJECT_NAME here to rename the whole site.
 *
 * All metrics below come from the real runs in ../../checkpoints
 * (BraTS 2020 training data, internal patient-level split: 258 train / 55 val / 55 test).
 * They were measured on the 55-patient VALIDATION split, which was also used for
 * checkpoint selection, so treat them as optimistic. They are NOT official
 * BraTS leaderboard numbers.
 */
export const PROJECT_NAME = "NEUROSCAN/4";
export const PROJECT_TAGLINE = "Multi-modal MRI brain tumour segmentation";
export const DISCLAIMER = "Research prototype, not for clinical diagnosis.";

export type ModelRun = {
  id: "best_attention_unet" | "best_attention_95" | "best_attention_heavy";
  label: string;
  channel: string;
  params: string;
  epochs: number;
  filters: number;
  loss: string;
  inference: string;
  wt: number;
  tc: number;
  et: number;
  balAcc: number;
  color: string;
};

export const MODEL_RUNS: ModelRun[] = [
  {
    id: "best_attention_unet",
    label: "STAGE 1 · BASELINE",
    channel: "CH1",
    params: "2.0M",
    epochs: 5,
    filters: 16,
    loss: "CE + DICE",
    inference: "ARGMAX + CC FILTER",
    wt: 89.57,
    tc: 82.23,
    et: 81.23,
    balAcc: 93.45,
    color: "#4fae68",
  },
  {
    id: "best_attention_95",
    label: "STAGE 2 · SCALED",
    channel: "CH2",
    params: "7.9M",
    epochs: 20,
    filters: 32,
    loss: "DICE + FOCAL",
    inference: "TTA ×3 + CC FILTER",
    wt: 90.33,
    tc: 84.06,
    et: 81.41,
    balAcc: 93.92,
    color: "#ffb000",
  },
  {
    id: "best_attention_heavy",
    label: "STAGE 3 · HEAVY",
    channel: "CH3",
    params: "31.5M",
    epochs: 25,
    filters: 64,
    loss: "W. DICE + FOCAL",
    inference: "TTA ×3 + CAL. THRESH + CC",
    wt: 90.8,
    tc: 85.23,
    et: 82.76,
    balAcc: 95.01,
    color: "#7cff9b",
  },
];

export const SECTIONS = [
  { id: "hero", code: "01", label: "GANTRY" },
  { id: "viewer", code: "02", label: "SLICE" },
  { id: "recon", code: "03", label: "VOLUME" },
  { id: "pipeline", code: "04", label: "SCHEMATIC" },
  { id: "metrics", code: "05", label: "TRACE" },
  { id: "ethics", code: "06", label: "REPORT" },
] as const;

