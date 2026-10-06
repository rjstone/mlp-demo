import { diagramWeightColor } from "@/lib/mlp";

const INPUT_ROWS = [
  { value: 1, weight: -1.5 },
  { value: 0.5, weight: 0 },
  { value: 0.8, weight: 2 },
] as const;

const BIAS = 0.5;

const products = INPUT_ROWS.map((row) => row.value * row.weight);
const weightedSum = products.reduce((sum, term) => sum + term, 0);
const preActivation = weightedSum + BIAS;
const activation = Math.tanh(preActivation);

const VB_W = 520;
const VB_H = 400;

const IN_X = 92;
const IN_R = 18;
const IN_Y = [156, 246, 336];

const CX = 248;
const CY = 246;
const CR = 40;

const OUT = [
  { x: 352, y: 210 },
  { x: 368, y: 246 },
  { x: 352, y: 282 },
];

const PLOT = { x: 286, y: 64, w: 206, h: 100 };

type Pt = { x: number; y: number };

function trim(from: Pt, to: Pt, fromR: number, toR: number): [Pt, Pt] {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy);
  const ux = dx / len;
  const uy = dy / len;
  return [
    { x: from.x + ux * fromR, y: from.y + uy * fromR },
    { x: to.x - ux * toR, y: to.y - uy * toR },
  ];
}

function edgeLabel(a: Pt, b: Pt, side: -1 | 1): Pt {
  const mx = a.x + (b.x - a.x) * 0.58;
  const my = a.y + (b.y - a.y) * 0.58;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy);
  return { x: mx + (-dy / len) * 22 * side, y: my + (dx / len) * 22 * side };
}

function signed(n: number, digits: number): string {
  const text = Math.abs(n).toFixed(digits);
  return n < 0 ? `−${text}` : text;
}

function weightText(w: number): string {
  if (w === 0) return "0";
  const abs = Number.isInteger(w) ? String(Math.abs(w)) : Math.abs(w).toFixed(1);
  return w < 0 ? `−${abs}` : abs;
}

const inputEdges = INPUT_ROWS.map((row, i) => {
  const [a, b] = trim({ x: IN_X, y: IN_Y[i] }, { x: CX, y: CY }, IN_R, CR);
  const side: -1 | 1 = i === 2 ? 1 : -1;
  return { a, b, label: edgeLabel(a, b, side), weight: row.weight, color: diagramWeightColor(row.weight) };
});

const outputEdges = OUT.map((tip) => {
  const [a] = trim({ x: CX, y: CY }, tip, CR, 0);
  return { a, b: tip };
});

const actText = signed(activation, 3);

function plotX(z: number): number {
  return PLOT.x + ((z + 2.2) / 4.4) * PLOT.w;
}

function plotY(y: number): number {
  return PLOT.y + ((1.15 - y) / 2.3) * PLOT.h;
}

function tanhCurve(): string {
  let d = "";
  for (let i = 0; i <= 48; i++) {
    const z = -2.2 + (4.4 * i) / 48;
    d += `${i === 0 ? "M" : "L"}${plotX(z).toFixed(1)} ${plotY(Math.tanh(z)).toFixed(1)} `;
  }
  return d;
}

const mathRows: { swatch?: string; left: string; right: string; result?: boolean }[] = [
  {
    swatch: inputEdges[0].color,
    left: `${signed(INPUT_ROWS[0].value, 1)} × ${weightText(INPUT_ROWS[0].weight)}`,
    right: signed(products[0], 2),
  },
  {
    swatch: inputEdges[1].color,
    left: `${signed(INPUT_ROWS[1].value, 1)} × ${weightText(INPUT_ROWS[1].weight)}`,
    right: signed(products[1], 2),
  },
  {
    swatch: inputEdges[2].color,
    left: `${signed(INPUT_ROWS[2].value, 1)} × ${weightText(INPUT_ROWS[2].weight)}`,
    right: signed(products[2], 2),
  },
  { left: "sum", right: signed(weightedSum, 2) },
  { left: `+ bias ${signed(BIAS, 1)}`, right: signed(preActivation, 2) },
  { left: "tanh", right: actText, result: true },
];

export function NodeDetail() {
  const axisY = plotY(0);
  const axisX = plotX(0);
  const dot = { x: plotX(preActivation), y: plotY(activation) };
  const weightRuleX = 138;

  return (
    <figure className="w-full rounded-2xl bg-paper text-ink shadow-plate">
      <figcaption className="px-5 pt-4">
        <h3 className="text-balance text-xl font-medium tracking-tight">Hidden Unit (Neuron) Detail</h3>
      </figcaption>
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        role="img"
        aria-label={`Hidden unit detail. Three inputs ${INPUT_ROWS.map((row) => signed(row.value, 1)).join(", ")} connect through weights ${INPUT_ROWS.map((row) => weightText(row.weight)).join(", ")}. The unit bias is ${signed(BIAS, 1)}. Weighted sum ${signed(weightedSum, 2)} plus the bias is ${signed(preActivation, 2)}, and tanh of that is ${actText} on every outgoing arrow.`}
        className="block w-full select-none"
      >
        <defs>
          <marker
            id="neuron-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <path d="M0 1.2 L9 5 L0 8.8 Z" fill="var(--color-ink)" />
          </marker>
          <clipPath id="neuron-tanh-clip">
            <rect x={PLOT.x} y={PLOT.y} width={PLOT.w} height={PLOT.h} rx="10" />
          </clipPath>
        </defs>

        <rect x={PLOT.x} y={PLOT.y} width={PLOT.w} height={PLOT.h} rx="10" fill="var(--color-well)" />
        <g clipPath="url(#neuron-tanh-clip)">
          <line x1={PLOT.x} y1={axisY} x2={PLOT.x + PLOT.w} y2={axisY} stroke="var(--color-ink)" strokeOpacity={0.28} strokeWidth={1} />
          <line x1={axisX} y1={PLOT.y} x2={axisX} y2={PLOT.y + PLOT.h} stroke="var(--color-ink)" strokeOpacity={0.28} strokeWidth={1} />
          <path d={tanhCurve()} fill="none" stroke="var(--color-ink)" strokeWidth={2} />
          <line x1={dot.x} y1={axisY} x2={dot.x} y2={dot.y} stroke="#007700" strokeWidth={1} strokeDasharray="2 2" />
          <circle cx={dot.x} cy={axisY} r={2.5} fill="var(--color-ink)" />
          <circle cx={dot.x} cy={dot.y} r={3.5} fill="#007700" />
        </g>
        <text
          x={dot.x}
          y={axisY + 12}
          textAnchor="middle"
          className="font-mono"
          fontSize={10}
          fill="var(--color-ink)"
        >
          {signed(preActivation, 2)}
        </text>
        <text
          x={dot.x - 6}
          y={dot.y - 6}
          textAnchor="end"
          className="font-mono"
          fontSize={10}
          fill="#007700"
        >
          {actText}
        </text>
        <text x={PLOT.x + 10} y={PLOT.y + 18} className="font-sans" fontSize={14} fontWeight={500} fill="var(--color-ink)">
          tanh()
        </text>
        <g className="font-mono" fill="var(--color-ink-soft)" fontSize={10}>
          <text x={PLOT.x - 4} y={plotY(1)} textAnchor="end" dominantBaseline="central">
            1
          </text>
          <text x={PLOT.x - 4} y={axisY} textAnchor="end" dominantBaseline="central">
            0
          </text>
          <text x={PLOT.x - 4} y={plotY(-1)} textAnchor="end" dominantBaseline="central">
            −1
          </text>
          <text x={plotX(-2)} y={PLOT.y - 4} textAnchor="middle">
            −2
          </text>
          <text x={axisX} y={PLOT.y - 4} textAnchor="middle">
            0
          </text>
          <text x={plotX(2)} y={PLOT.y - 4} textAnchor="middle">
            2
          </text>
        </g>

        {inputEdges.map((edge) => (
          <line
            key={`in-${edge.weight}`}
            x1={edge.a.x}
            y1={edge.a.y}
            x2={edge.b.x}
            y2={edge.b.y}
            stroke={edge.color}
            strokeWidth={4}
            strokeLinecap="round"
          />
        ))}

        {outputEdges.map((edge) => (
          <line
            key={`out-${edge.b.y}`}
            x1={edge.a.x}
            y1={edge.a.y}
            x2={edge.b.x}
            y2={edge.b.y}
            stroke="var(--color-ink)"
            strokeWidth={1.75}
            markerEnd="url(#neuron-arrow)"
          />
        ))}

        <line
          x1={weightRuleX}
          y1={172}
          x2={weightRuleX}
          y2={322}
          stroke="var(--color-ink-soft)"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <line x1={weightRuleX} y1={172} x2={176} y2={148} stroke="var(--color-ink-soft)" strokeWidth={1} />
        <text x={176} y={142} textAnchor="start" className="font-sans" fontSize={14} fontWeight={500} fill="var(--color-ink)">
          Weights
        </text>

        {INPUT_ROWS.map((row, i) => (
          <g key={`input-${row.weight}`}>
            <text
              x={IN_X - IN_R - 10}
              y={IN_Y[i]}
              textAnchor="end"
              dominantBaseline="central"
              className="font-mono"
              fontSize={15}
              fontWeight={500}
              fill="#007700"
            >
              {signed(row.value, 1)}
            </text>
            <circle
              cx={IN_X}
              cy={IN_Y[i]}
              r={IN_R}
              fill={diagramWeightColor(row.value)}
              stroke="var(--color-ink)"
              strokeOpacity={0.35}
              strokeWidth={1.25}
            />
          </g>
        ))}

        <circle cx={CX} cy={CY} r={CR} fill={diagramWeightColor(BIAS)} stroke="var(--color-ink)" strokeOpacity={0.45} strokeWidth={1.5} />
        <text x={CX} y={CY - 7} textAnchor="middle" className="font-sans" fontSize={11} fill="var(--color-ink)">
          bias
        </text>
        <text x={CX} y={CY + 13} textAnchor="middle" className="font-mono" fontSize={16} fontWeight={600} fill="var(--color-ink)">
          {signed(BIAS, 1)}
        </text>

        {inputEdges.map((edge) => (
          <g key={`wlabel-${edge.weight}`}>
            <rect
              x={edge.label.x - (weightText(edge.weight).length > 1 ? 22 : 10)}
              y={edge.label.y - 10}
              width={weightText(edge.weight).length > 1 ? 44 : 20}
              height={20}
              rx={5}
              fill="var(--color-paper)"
            />
            <text
              x={edge.label.x}
              y={edge.label.y}
              textAnchor="middle"
              dominantBaseline="central"
              className="font-mono"
              fontSize={14}
              fontWeight={600}
              fill="var(--color-ink)"
            >
              {weightText(edge.weight)}
            </text>
          </g>
        ))}

        {outputEdges.map((edge) => (
          <text
            key={`ylabel-${edge.b.y}`}
            x={edge.b.x + 12}
            y={edge.b.y}
            dominantBaseline="central"
            className="font-mono"
            fontSize={14}
            fontWeight={600}
            fill="#007700"
          >
            {actText}
          </text>
        ))}
      </svg>

      <div className="space-y-3 px-5 pb-5">
        <ul className="mx-auto w-[45%] min-w-fit space-y-1 font-mono text-[13px] leading-6 tabular-nums">
          {mathRows.map((row) => (
            <li key={row.left} className="flex items-baseline justify-between gap-4">
              <span className="flex items-center gap-2">
                {row.swatch ? (
                  <span className="inline-block size-2.5 shrink-0 rounded-full" style={{ background: row.swatch }} aria-hidden="true" />
                ) : (
                  <span className="inline-block size-2.5 shrink-0" aria-hidden="true" />
                )}
                <span className={row.result ? "font-semibold text-[#007700]" : undefined}>{row.left}</span>
              </span>
              <span className={row.result ? "font-semibold text-[#007700]" : undefined}>{row.right}</span>
            </li>
          ))}
        </ul>
        <p className="text-sm leading-relaxed text-pretty text-ink-soft">
          Every arrow carries that same output. At the next layer, each edge multiplies it by a different weight. Those weights are not drawn here.
        </p>
      </div>
    </figure>
  );
}
