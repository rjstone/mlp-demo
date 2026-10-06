import { b1, b2, w1, w2 } from "@/lib/mlp-weights";

export const INPUTS = 20;
export const HIDDEN = 8;
export const OUTPUTS = 11;
export const LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "?"] as const;

/** A hidden unit is "driven" when |W x| from the on-pixels is at least this
 * fraction of the strongest hidden unit. Bias is not included. */
export const DRIVE_FRACTION = 0.25;

export type NetState = {
  contrib: number[];
  driven: boolean[];
  probs: number[];
  winner: number;
};

const ZERO = [128, 128, 128];
const POS = [0, 255, 0];
const NEG = [255, 0, 0];

function extent(): { lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  const scan = (v: number) => {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  };
  for (const row of w1) for (const v of row) scan(v);
  for (const v of b1) scan(v);
  for (const row of w2) for (const v of row) scan(v);
  for (const v of b2) scan(v);
  return { lo, hi };
}

export const weightScale = extent();

function mix(from: number[], to: number[], t: number): string {
  const u = Math.min(1, Math.max(0, t));
  const r = Math.round(from[0] + (to[0] - from[0]) * u);
  const g = Math.round(from[1] + (to[1] - from[1]) * u);
  const b = Math.round(from[2] + (to[2] - from[2]) * u);
  return `rgb(${r}, ${g}, ${b})`;
}

/** 0 → 50% gray, most positive → #00FF00, most negative → #FF0000. */
export function weightColor(value: number): string {
  const { lo, hi } = weightScale;
  if (value >= 0) return mix(ZERO, POS, hi > 0 ? value / hi : 0);
  return mix(ZERO, NEG, lo < 0 ? value / lo : 0);
}

export const hiddenBiasColor = b1.map(weightColor);
export const outputBiasColor = b2.map(weightColor);
export const hiddenEdgeColor = w1.map((row) => row.map(weightColor));
export const outputEdgeColor = w2.map((row) => row.map(weightColor));

export function formatProb(p: number): string {
  return p.toFixed(2);
}

export function formatSigned(n: number): string {
  const text = n.toFixed(2);
  return n > 0 ? `+${text}` : text;
}

export function infer(bits: readonly number[]): NetState {
  const contrib = new Array<number>(HIDDEN);
  const hidden = new Array<number>(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let fromInputs = 0;
    let sum = b1[j];
    const row = w1[j];
    for (let i = 0; i < INPUTS; i++) {
      const term = row[i] * bits[i];
      fromInputs += term;
      sum += term;
    }
    contrib[j] = fromInputs;
    hidden[j] = Math.tanh(sum);
  }

  let maxAbs = 0;
  for (const c of contrib) {
    const a = Math.abs(c);
    if (a > maxAbs) maxAbs = a;
  }
  const driven = contrib.map((c) => maxAbs > 1e-8 && Math.abs(c) >= DRIVE_FRACTION * maxAbs);

  const logits = new Array<number>(OUTPUTS);
  for (let k = 0; k < OUTPUTS; k++) {
    let sum = b2[k];
    const row = w2[k];
    for (let j = 0; j < HIDDEN; j++) sum += row[j] * hidden[j];
    logits[k] = sum;
  }

  let peak = logits[0];
  for (let k = 1; k < OUTPUTS; k++) if (logits[k] > peak) peak = logits[k];
  const exps = logits.map((z) => Math.exp(z - peak));
  let total = 0;
  for (const e of exps) total += e;
  const probs = exps.map((e) => e / total);

  let winner = 0;
  for (let k = 1; k < OUTPUTS; k++) if (probs[k] > probs[winner]) winner = k;

  return { contrib, driven, probs, winner };
}

export function weightCount(): { weights: number; biases: number } {
  return {
    weights: HIDDEN * INPUTS + OUTPUTS * HIDDEN,
    biases: HIDDEN + OUTPUTS,
  };
}
