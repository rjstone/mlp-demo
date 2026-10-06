/**
 * Train the 4×5 digit MLP and write src/lib/mlp-weights.ts.
 * Architecture: 20 → tanh(8) → softmax(11). Labels 0–9 and ?.
 * Run: node scripts/train-mlp.mjs
 */
import { writeFileSync } from "node:fs";

const INPUTS = 20;
const HIDDEN = 8;
const OUTPUTS = 11;
const Q = 10;

function parse(rows) {
  if (rows.length !== 5 || rows.some((r) => r.length !== 4)) {
    throw new Error(`bad glyph ${rows.join("|")}`);
  }
  const x = [];
  for (const row of rows) {
    for (const ch of row) x.push(ch === "#" ? 1 : 0);
  }
  return x;
}

function keyOf(x) {
  return x.join("");
}

function pop(x) {
  return x.reduce((s, v) => s + v, 0);
}

function hamming(a, b) {
  let d = 0;
  for (let i = 0; i < INPUTS; i++) if (a[i] !== b[i]) d++;
  return d;
}

function ascii(x) {
  let s = "";
  for (let r = 0; r < 5; r++) {
    s += x.slice(r * 4, r * 4 + 4).map((v) => (v ? "#" : ".")).join("") + "\n";
  }
  return s;
}

/** @type {{y:number,x:number[]}[]} */
const prototypes = [];

function addDigit(y, glyphs) {
  for (const g of glyphs) prototypes.push({ y, x: parse(g) });
}

addDigit(0, [
  ["####", "#..#", "#..#", "#..#", "####"],
  [".###", "#..#", "#..#", "#..#", ".###"],
  [".###", ".#.#", ".#.#", ".#.#", ".###"],
  ["###.", "#..#", "#..#", "#..#", "###."],
  ["####", "#..#", "#..#", "#..#", ".###"],
]);

addDigit(1, [
  [".#..", "##..", ".#..", ".#..", "###."],
  ["..#.", ".##.", "..#.", "..#.", ".###"],
  ["...#", "..##", "...#", "...#", ".###"],
  ["#...", "#...", "#...", "#...", "#..."],
  [".#..", ".#..", ".#..", ".#..", ".#.."],
  ["..#.", "..#.", "..#.", "..#.", "..#."],
  ["...#", "...#", "...#", "...#", "...#"],
  [".#..", ".#..", ".#..", ".#..", "###."],
  ["..#.", "..#.", "..#.", "..#.", "####"],
]);

addDigit(2, [
  ["####", "...#", ".###", "#...", "####"],
  ["####", "...#", "####", "#...", "####"],
  [".###", "...#", ".###", "#...", ".###"],
  [".###", "..#.", ".#..", "#...", "####"],
  ["####", "...#", "..##", "#...", "####"],
  ["###.", "...#", "###.", "#...", "###."],
]);

addDigit(3, [
  ["####", "...#", ".###", "...#", "####"],
  ["####", "...#", "####", "...#", "####"],
  [".###", "...#", ".###", "...#", ".###"],
  [".###", "..#.", ".###", "..#.", ".###"],
  ["####", "..#.", ".###", "..#.", "####"],
  ["###.", "...#", "###.", "...#", "###."],
]);

addDigit(4, [
  ["#..#", "#..#", "####", "...#", "...#"],
  [".#.#", ".#.#", "####", "..#.", "..#."],
  ["..##", ".#.#", "####", "...#", "...#"],
  ["#.#.", "#.#.", "####", "..#.", "..#."],
  ["#..#", "#..#", "####", "..#.", "..#."],
  [".#.#", ".#.#", ".###", "...#", "...#"],
  ["#...", "#...", "####", "...#", "...#"],
]);

addDigit(5, [
  ["####", "#...", "####", "...#", "####"],
  ["####", "#...", "###.", "...#", "####"],
  [".###", "#...", ".###", "...#", ".###"],
  ["####", "#...", "####", "..#.", "##.."],
  ["####", "#...", "###.", "...#", "###."],
  ["###.", "#...", "###.", "...#", "###."],
]);

addDigit(6, [
  [".###", "#...", "####", "#..#", ".###"],
  ["####", "#...", "####", "#..#", "####"],
  [".##.", "#...", "###.", "#..#", ".##."],
  [".###", "#...", "###.", "#..#", ".###"],
  [".##.", "#...", "####", "#..#", ".###"],
  ["###.", "#...", "###.", "#..#", "###."],
]);

addDigit(7, [
  ["####", "...#", "..#.", ".#..", ".#.."],
  ["####", "...#", "..#.", "..#.", "..#."],
  ["####", "..#.", "..#.", ".#..", ".#.."],
  [".###", "...#", "..#.", ".#..", ".#.."],
  ["####", "...#", ".#..", ".#..", "#..."],
  ["####", "...#", "..#.", ".#..", "#..."],
  ["###.", "..#.", ".#..", ".#..", "#..."],
]);

addDigit(8, [
  ["####", "#..#", "####", "#..#", "####"],
  [".###", "#..#", ".###", "#..#", ".###"],
  [".###", "#.#.", ".###", "#.#.", ".###"],
  ["####", "#..#", ".###", "#..#", "####"],
  ["###.", "#..#", "###.", "#..#", "###."],
  ["####", "#..#", "####", "#..#", ".###"],
]);

addDigit(9, [
  ["####", "#..#", "####", "...#", "####"],
  [".###", "#..#", ".###", "...#", ".###"],
  ["####", "#..#", "####", "...#", ".###"],
  [".###", "#..#", "####", "...#", "..#."],
  ["####", "#..#", ".###", "...#", "####"],
  ["###.", "#..#", "###.", "...#", "###."],
  [".###", "#..#", ".###", "...#", "..#."],
]);

const seenProto = new Map();
for (const p of prototypes) {
  const k = keyOf(p.x);
  if (seenProto.has(k) && seenProto.get(k) !== p.y) {
    throw new Error(`prototype collision ${seenProto.get(k)} vs ${p.y}\n${ascii(p.x)}`);
  }
  seenProto.set(k, p.y);
}

let minSep = 99;
const closePairs = [];
for (let i = 0; i < prototypes.length; i++) {
  for (let j = i + 1; j < prototypes.length; j++) {
    if (prototypes[i].y === prototypes[j].y) continue;
    const d = hamming(prototypes[i].x, prototypes[j].x);
    if (d < minSep) minSep = d;
    if (d <= 2) {
      closePairs.push({ d, a: prototypes[i].y, b: prototypes[j].y, x: prototypes[i].x });
    }
  }
}

/** @type {Map<string, {x:number[], y:number, w:number, kind:string}>} */
const samples = new Map();

function put(x, y, w, kind) {
  const k = keyOf(x);
  const prev = samples.get(k);
  if (!prev) {
    samples.set(k, { x, y, w, kind });
    return;
  }
  if (prev.y !== y) {
    prev.y = Q;
    prev.kind = "ambiguous";
    prev.w = Math.max(prev.w, w);
    return;
  }
  prev.w = Math.max(prev.w, w);
  if (kind === "proto") prev.kind = "proto";
}

for (const p of prototypes) put(p.x, p.y, 5, "proto");

for (const p of prototypes) {
  for (let i = 0; i < INPUTS; i++) {
    const x = p.x.slice();
    x[i] ^= 1;
    const k = keyOf(x);
    if (seenProto.has(k)) continue;
    put(x, p.y, 1.5, "neighbor");
  }
}

function putJunk(x, w) {
  const k = keyOf(x);
  if (seenProto.has(k)) return;
  const prev = samples.get(k);
  if (prev && prev.y !== Q && prev.kind !== "ambiguous") return;
  put(x, Q, w, "junk");
}

putJunk(Array(INPUTS).fill(0), 12);
putJunk(Array(INPUTS).fill(1), 6);

for (let r = 0; r < 5; r++) {
  const x = Array(INPUTS).fill(0);
  for (let c = 0; c < 4; c++) x[r * 4 + c] = 1;
  putJunk(x, 3);
}

for (let r0 = 0; r0 < 5; r0++) {
  for (let r1 = r0 + 1; r1 < 5; r1++) {
    const x = Array(INPUTS).fill(0);
    for (const r of [r0, r1]) for (let c = 0; c < 4; c++) x[r * 4 + c] = 1;
    putJunk(x, 2);
  }
}

const checkerA = Array.from({ length: INPUTS }, (_, i) => ((i % 4) + Math.floor(i / 4)) % 2);
const checkerB = checkerA.map((v) => 1 - v);
putJunk(checkerA, 3);
putJunk(checkerB, 3);

for (const diag of [0, 1]) {
  const x = Array(INPUTS).fill(0);
  for (let r = 0; r < 5; r++) {
    const c = diag === 0 ? Math.min(3, r) : Math.max(0, 3 - r);
    x[r * 4 + c] = 1;
  }
  putJunk(x, 3);
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rngData = mulberry32(20261005);
const protosOnly = prototypes.map((p) => p.x);
let randomKept = 0;
let guard = 0;
while (randomKept < 80 && guard < 8000) {
  guard++;
  const density = rngData() < 0.5 ? 0.15 + rngData() * 0.2 : 0.35 + rngData() * 0.4;
  const x = Array.from({ length: INPUTS }, () => (rngData() < density ? 1 : 0));
  let near = 99;
  for (const p of protosOnly) near = Math.min(near, hamming(x, p));
  if (near < 5) continue;
  const before = samples.size;
  putJunk(x, 1);
  if (samples.size !== before) randomKept++;
}

const data = [...samples.values()];
const classCount = Array(OUTPUTS).fill(0);
for (const s of data) classCount[s.y]++;

console.log(`prototypes ${prototypes.length}  samples ${data.length}  min inter-class hamming ${minSep}`);
console.log("class counts", classCount.map((n, i) => `${i === Q ? "?" : i}:${n}`).join(" "));
if (closePairs.length) {
  console.log(`close pairs (<=2): ${closePairs.length}`);
  for (const pair of closePairs.slice(0, 8)) {
    console.log(`  ${pair.a} vs ${pair.b} d=${pair.d}\n${ascii(pair.x)}`);
  }
}

function zeros(n) {
  return Array(n).fill(0);
}
function randn(rng) {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}

function softmax(logits) {
  let m = -Infinity;
  for (const z of logits) if (z > m) m = z;
  const exps = logits.map((z) => Math.exp(z - m));
  const sum = exps.reduce((s, v) => s + v, 0);
  return exps.map((v) => v / sum);
}

function forward(model, x) {
  const h = zeros(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let s = model.b1[j];
    const row = model.w1[j];
    for (let i = 0; i < INPUTS; i++) s += row[i] * x[i];
    h[j] = Math.tanh(s);
  }
  const z = zeros(OUTPUTS);
  for (let k = 0; k < OUTPUTS; k++) {
    let s = model.b2[k];
    const row = model.w2[k];
    for (let j = 0; j < HIDDEN; j++) s += row[j] * h[j];
    z[k] = s;
  }
  return { h, p: softmax(z) };
}

function trainOne(seed) {
  const rng = mulberry32(seed);
  const scale1 = Math.sqrt(2 / INPUTS);
  const scale2 = Math.sqrt(2 / HIDDEN);
  const model = {
    w1: Array.from({ length: HIDDEN }, () => Array.from({ length: INPUTS }, () => randn(rng) * scale1)),
    b1: zeros(HIDDEN),
    w2: Array.from({ length: OUTPUTS }, () => Array.from({ length: HIDDEN }, () => randn(rng) * scale2)),
    b2: zeros(OUTPUTS),
  };
  const adam = {};
  for (const name of ["w1", "b1", "w2", "b2"]) {
    adam[name] = {
      m: model[name].map((row) => (Array.isArray(row) ? row.map(() => 0) : 0)),
      v: model[name].map((row) => (Array.isArray(row) ? row.map(() => 0) : 0)),
    };
  }

  // Feature-shaped start so the eight units begin as stroke detectors,
  // then Adam is free to move them. Rows: top, mid, bot, left, right,
  // inner, diag-down, diag-up.
  const featurePixels = [
    [0, 1, 2, 3],
    [8, 9, 10, 11],
    [16, 17, 18, 19],
    [0, 4, 8, 12, 16],
    [3, 7, 11, 15, 19],
    [5, 6, 9, 10, 13, 14],
    [0, 5, 10, 15],
    [3, 6, 9, 12],
  ];
  for (let j = 0; j < HIDDEN; j++) {
    for (let i = 0; i < INPUTS; i++) model.w1[j][i] *= 0.25;
    for (const i of featurePixels[j]) model.w1[j][i] += 0.55;
    for (let i = 0; i < INPUTS; i++) {
      if (!featurePixels[j].includes(i)) model.w1[j][i] -= 0.12;
    }
  }

  const lr0 = 0.03;
  const l2 = 2e-4;
  const steps = 700;
  let tStep = 0;

  function adamUpdate(name, grad, lr) {
    const m = adam[name].m;
    const v = adam[name].v;
    const value = model[name];
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    tStep++;
    const bc1 = 1 - b1 ** tStep;
    const bc2 = 1 - b2 ** tStep;
    const apply = (g, mm, vv, current) => {
      const nextM = b1 * mm + (1 - b1) * g;
      const nextV = b2 * vv + (1 - b2) * g * g;
      const hatM = nextM / bc1;
      const hatV = nextV / bc2;
      return {
        nextM,
        nextV,
        next: current - lr * (hatM / (Math.sqrt(hatV) + eps)),
      };
    };
    if (!Array.isArray(value[0])) {
      for (let i = 0; i < value.length; i++) {
        const u = apply(grad[i], m[i], v[i], value[i]);
        m[i] = u.nextM;
        v[i] = u.nextV;
        value[i] = u.next;
      }
      return;
    }
    for (let r = 0; r < value.length; r++) {
      for (let c = 0; c < value[r].length; c++) {
        const u = apply(grad[r][c], m[r][c], v[r][c], value[r][c]);
        m[r][c] = u.nextM;
        v[r][c] = u.nextV;
        value[r][c] = u.next;
      }
    }
  }

  let lastLoss = 0;
  for (let step = 0; step < steps; step++) {
    const lr = lr0 * (step < 500 ? 1 : 0.35);
    const gW1 = model.w1.map((row) => row.map(() => 0));
    const gB1 = zeros(HIDDEN);
    const gW2 = model.w2.map((row) => row.map(() => 0));
    const gB2 = zeros(OUTPUTS);
    let loss = 0;
    let weightSum = 0;

    for (const sample of data) {
      const { h, p } = forward(model, sample.x);
      const y = sample.y;
      const sw = sample.w;
      weightSum += sw;
      loss += -Math.log(Math.max(p[y], 1e-12)) * sw;

      const dz = p.slice();
      dz[y] -= 1;
      for (let k = 0; k < OUTPUTS; k++) dz[k] *= sw;

      for (let k = 0; k < OUTPUTS; k++) {
        gB2[k] += dz[k];
        for (let j = 0; j < HIDDEN; j++) gW2[k][j] += dz[k] * h[j];
      }
      const dh = zeros(HIDDEN);
      for (let j = 0; j < HIDDEN; j++) {
        let s = 0;
        for (let k = 0; k < OUTPUTS; k++) s += model.w2[k][j] * dz[k];
        dh[j] = s * (1 - h[j] * h[j]);
      }
      for (let j = 0; j < HIDDEN; j++) {
        gB1[j] += dh[j];
        const row = sample.x;
        for (let i = 0; i < INPUTS; i++) gW1[j][i] += dh[j] * row[i];
      }
    }

    const inv = 1 / weightSum;
    for (let j = 0; j < HIDDEN; j++) {
      gB1[j] *= inv;
      for (let i = 0; i < INPUTS; i++) gW1[j][i] = gW1[j][i] * inv + l2 * model.w1[j][i];
    }
    for (let k = 0; k < OUTPUTS; k++) {
      gB2[k] *= inv;
      for (let j = 0; j < HIDDEN; j++) gW2[k][j] = gW2[k][j] * inv + l2 * model.w2[k][j];
    }

    const tBefore = tStep;
    adamUpdate("w1", gW1, lr);
    adamUpdate("b1", gB1, lr);
    adamUpdate("w2", gW2, lr);
    adamUpdate("b2", gB2, lr);
    tStep = tBefore + 1;
    lastLoss = loss / weightSum;
  }

  return { model, loss: lastLoss };
}

function evaluate(model) {
  let protoOk = 0;
  let protoN = 0;
  let protoP = 0;
  let nearOk = 0;
  let nearN = 0;
  let junkOk = 0;
  let junkN = 0;
  const misses = [];
  let blankP = 0;
  let blankOk = false;

  for (const sample of data) {
    const { p } = forward(model, sample.x);
    let pred = 0;
    for (let k = 1; k < OUTPUTS; k++) if (p[k] > p[pred]) pred = k;
    const ok = pred === sample.y;
    if (sample.kind === "proto") {
      protoN++;
      protoP += p[sample.y];
      if (ok) protoOk++;
      else if (misses.length < 6) misses.push({ y: sample.y, pred, p: p[sample.y], x: sample.x });
    } else if (sample.kind === "neighbor") {
      nearN++;
      if (ok) nearOk++;
    } else {
      junkN++;
      if (ok) junkOk++;
    }
    if (pop(sample.x) === 0) {
      blankOk = ok;
      blankP = p[Q];
    }
  }
  const protoAcc = protoOk / protoN;
  const nearAcc = nearN ? nearOk / nearN : 0;
  const junkAcc = junkN ? junkOk / junkN : 0;
  const meanP = protoP / protoN;
  const score = protoAcc * 5 + nearAcc * 2 + junkAcc + meanP + (blankOk ? 0.5 : 0) + blankP;
  return { protoAcc, nearAcc, junkAcc, meanP, blankOk, blankP, score, misses };
}

let best = null;
const TRIALS = 8;
for (let t = 0; t < TRIALS; t++) {
  const seed = 1000 + t * 17;
  const { model, loss } = trainOne(seed);
  const ev = evaluate(model);
  console.log(
    `seed ${seed}  loss ${loss.toFixed(4)}  proto ${(ev.protoAcc * 100).toFixed(1)}%  ` +
      `p̄ ${ev.meanP.toFixed(3)}  near ${(ev.nearAcc * 100).toFixed(1)}%  junk ${(ev.junkAcc * 100).toFixed(1)}%  ` +
      `blank ${ev.blankOk ? "ok" : "MISS"} ${ev.blankP.toFixed(3)}  score ${ev.score.toFixed(3)}`,
  );
  if (!best || ev.score > best.ev.score) best = { model, ev, loss, seed };
}

if (!best || best.ev.protoAcc < 1 || !best.ev.blankOk) {
  console.log("BEST FAILED GATE");
  for (const m of best?.ev.misses ?? []) {
    console.log(`miss true ${m.y === Q ? "?" : m.y} pred ${m.pred === Q ? "?" : m.pred} pTrue ${m.p.toFixed(3)}\n${ascii(m.x)}`);
  }
  process.exit(1);
}

const { model } = best;
let lo = Infinity;
let hi = -Infinity;
function scan(v) {
  if (v < lo) lo = v;
  if (v > hi) hi = v;
}
for (const row of model.w1) for (const v of row) scan(v);
for (const v of model.b1) scan(v);
for (const row of model.w2) for (const v of row) scan(v);
for (const v of model.b2) scan(v);

console.log(`\nselected seed ${best.seed}  weight range ${lo.toFixed(3)} .. ${hi.toFixed(3)}`);
console.log("prototype gallery:");
const shown = new Set();
for (const sample of data) {
  if (sample.kind !== "proto") continue;
  if (shown.has(sample.y)) continue;
  shown.add(sample.y);
  const { p } = forward(model, sample.x);
  const line = p.map((v, i) => `${i === Q ? "?" : i}:${v.toFixed(2)}`).join(" ");
  console.log(`class ${sample.y === Q ? "?" : sample.y}\n${ascii(sample.x)}${line}\n`);
}

function fmt(n) {
  const rounded = Math.round(n * 1e6) / 1e6;
  return JSON.stringify(rounded);
}
function fmtMat(mat) {
  return `[\n${mat.map((row) => `  [${row.map(fmt).join(", ")}]`).join(",\n")},\n]`;
}
function fmtVec(vec) {
  return `[${vec.map(fmt).join(", ")}]`;
}

const out = `// Generated by scripts/train-mlp.mjs — do not edit by hand.
// 20 → tanh 8 → softmax 11. Class 10 is "?".

export const w1: number[][] = ${fmtMat(model.w1)};

export const b1: number[] = ${fmtVec(model.b1)};

export const w2: number[][] = ${fmtMat(model.w2)};

export const b2: number[] = ${fmtVec(model.b2)};
`;

writeFileSync(new URL("../src/lib/mlp-weights.ts", import.meta.url), out);
console.log("wrote src/lib/mlp-weights.ts");
