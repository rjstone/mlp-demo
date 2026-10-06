import { useLayoutEffect, useRef, useState } from "react";
import {
  DRIVE_FRACTION,
  HIDDEN,
  INPUTS,
  LABELS,
  OUTPUTS,
  formatProb,
  formatSigned,
  hiddenBiasColor,
  hiddenEdgeColor,
  outputBiasColor,
  outputEdgeColor,
  weightScale,
  type NetState,
} from "@/lib/mlp";

const VB_W = 880;
const PAD_L = 68;
const PAD_R = 16;

const R_IN = 8;
const R_HID = 11;
const R_OUT = 12;
const Y_IN = 58;
const BELOW_OUT = 84;
const MIN_PLOT = 400;

function xs(count: number): number[] {
  const usable = VB_W - PAD_L - PAD_R;
  if (count <= 1) return [PAD_L + usable / 2];
  return Array.from({ length: count }, (_, i) => PAD_L + (i * usable) / (count - 1));
}

const X_IN = xs(INPUTS);
const X_HID = xs(HIDDEN);
const X_OUT = xs(OUTPUTS);

type Line = { key: string; x1: number; y1: number; x2: number; y2: number; color: string };

type Plot = {
  h: number;
  yIn: number;
  yHid: number;
  yOut: number;
  yClass: number;
  yProb: number;
};

function plotLayout(height: number): Plot {
  const h = Math.max(MIN_PLOT, Math.round(height));
  const yOut = h - BELOW_OUT;
  return {
    h,
    yIn: Y_IN,
    yHid: (Y_IN + yOut) / 2,
    yOut,
    yClass: yOut + 32,
    yProb: yOut + 52,
  };
}

const zeroStop = ((0 - weightScale.lo) / (weightScale.hi - weightScale.lo)) * 100;
const scaleGradient = `linear-gradient(90deg, rgb(255, 0, 0) 0%, rgb(128, 128, 128) ${zeroStop}%, rgb(0, 255, 0) 100%)`;

type DiagramProps = {
  bits: readonly number[];
  net: NetState;
};

export function MlpDiagram({ bits, net }: DiagramProps) {
  const plotHost = useRef<HTMLDivElement>(null);
  const [plot, setPlot] = useState<Plot>(() => plotLayout(MIN_PLOT));

  useLayoutEffect(() => {
    const host = plotHost.current;
    if (!host) return;
    const measure = () => {
      const next = plotLayout(host.clientHeight);
      setPlot((prev) => (prev.h === next.h ? prev : next));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const { h: plotH, yIn, yHid, yOut, yClass, yProb } = plot;

  const inputEdges: Line[] = [];
  const inputHot: Line[] = [];
  for (let j = 0; j < HIDDEN; j++) {
    for (let i = 0; i < INPUTS; i++) {
      const line = {
        key: `h-${i}-${j}`,
        x1: X_IN[i],
        y1: yIn + R_IN,
        x2: X_HID[j],
        y2: yHid - R_HID,
        color: hiddenEdgeColor[j][i],
      };
      (bits[i] === 1 ? inputHot : inputEdges).push(line);
    }
  }

  const outputEdges: Line[] = [];
  const outputHot: Line[] = [];
  for (let k = 0; k < OUTPUTS; k++) {
    for (let j = 0; j < HIDDEN; j++) {
      const line = {
        key: `o-${j}-${k}`,
        x1: X_HID[j],
        y1: yHid + R_HID,
        x2: X_OUT[k],
        y2: yOut - R_OUT,
        color: outputEdgeColor[k][j],
      };
      (net.driven[j] ? outputHot : outputEdges).push(line);
    }
  }

  const summary = LABELS.map((label, i) => `${label} ${formatProb(net.probs[i])}`).join(", ");

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-col gap-3 p-4 pb-2">
        <div className="inline-grid w-fit grid-cols-[auto_11rem_auto] items-center self-start">
          <p className="col-start-1 pr-1.5 text-xs text-ink">Weights and biases</p>
          <div className="col-start-2 h-2 rounded-full" style={{ background: scaleGradient }} />
          <p className="col-start-3 flex items-center gap-2 pl-3 text-xs text-ink">
            Active path
            <span className="inline-block h-1 w-8 rounded-full bg-signal" aria-hidden="true" />
          </p>
          <div className="col-start-2 mt-1 flex justify-between font-mono text-xs tabular-nums text-ink-soft">
            <span>{formatSigned(weightScale.lo)}</span>
            <span>0</span>
            <span>{formatSigned(weightScale.hi)}</span>
          </div>
        </div>
        <p className="max-w-3xl text-xs leading-relaxed text-pretty text-ink-soft">
          Inputs are row-major: four pixels per row, top row at the left. Node fill is the bias.
          Input nodes have none — blue means that pixel is 1. A hidden unit is driven when the
          on-pixels push it by at least {DRIVE_FRACTION} of the strongest unit, bias excluded.
        </p>
      </div>
      <div className="mx-2 mb-3 min-h-0 flex-1 overflow-x-auto">
        <div ref={plotHost} className="relative h-full min-h-[400px] w-[880px]">
          <svg
            viewBox={`0 0 ${VB_W} ${plotH}`}
            role="img"
            aria-label={`Perceptron. Highest probability is ${LABELS[net.winner]} at ${formatProb(net.probs[net.winner])}. ${summary}.`}
            className="absolute inset-0 h-full w-full font-mono select-none"
          >
          <EdgeLayer lines={inputEdges} hot={false} />
          <EdgeLayer lines={outputEdges} hot={false} />
          <EdgeLayer lines={inputHot} hot />
          <EdgeLayer lines={outputHot} hot />

          {Array.from({ length: 5 }, (_, row) => {
            const left = X_IN[row * 4];
            const right = X_IN[row * 4 + 3];
            const x = left - 16;
            const width = right - left + 32;
            return (
              <g key={row}>
                <rect x={x} y={4} width={width} height={42} rx={6} fill="var(--color-well)" />
                <text
                  x={(left + right) / 2}
                  y={16}
                  textAnchor="middle"
                  fill="var(--color-ink-soft)"
                  fontSize={10}
                >
                  row {row + 1}
                </text>
              </g>
            );
          })}

          <LayerLabel y={yIn} text="input" />
          <LayerLabel y={yHid} text="hidden" />
          <LayerLabel y={yOut} text="output" />

          {X_IN.map((x, i) => {
            const on = bits[i] === 1;
            return (
              <g key={`in-${i}`}>
                <text
                  x={x}
                  y={36}
                  textAnchor="middle"
                  fill={on ? "var(--color-signal)" : "var(--color-ink-soft)"}
                  fontSize={13}
                  fontWeight={on ? 600 : 500}
                >
                  {on ? "1" : "0"}
                </text>
                <circle
                  cx={x}
                  cy={yIn}
                  r={R_IN}
                  fill={on ? "var(--color-signal)" : "var(--color-well)"}
                  stroke="var(--color-ink)"
                  strokeOpacity={on ? 0.45 : 0.28}
                  strokeWidth={1}
                />
              </g>
            );
          })}

          {X_HID.map((x, j) => (
            <circle
              key={`hid-${j}`}
              cx={x}
              cy={yHid}
              r={R_HID}
              fill={hiddenBiasColor[j]}
              stroke={net.driven[j] ? "var(--color-signal)" : "var(--color-ink)"}
              strokeOpacity={net.driven[j] ? 1 : 0.4}
              strokeWidth={net.driven[j] ? 2.5 : 1}
            />
          ))}

          {X_OUT.map((x, k) => {
            const win = k === net.winner;
            return (
              <g key={`out-${k}`}>
                <circle
                  cx={x}
                  cy={yOut}
                  r={R_OUT}
                  fill={outputBiasColor[k]}
                  stroke="var(--color-ink)"
                  strokeOpacity={win ? 0.9 : 0.35}
                  strokeWidth={win ? 2.25 : 1}
                />
                <text
                  x={x}
                  y={yClass}
                  textAnchor="middle"
                  fill={win ? "var(--color-ink)" : "var(--color-ink-soft)"}
                  fontSize={13}
                  fontWeight={win ? 600 : 500}
                >
                  {LABELS[k]}
                </text>
                <text
                  x={x}
                  y={yProb}
                  textAnchor="middle"
                  fill={win ? "var(--color-ink)" : "var(--color-ink-soft)"}
                  fontSize={11}
                  fontWeight={win ? 600 : 400}
                >
                  {formatProb(net.probs[k])}
                </text>
              </g>
            );
          })}
          </svg>
        </div>
      </div>
    </div>
  );
}

function LayerLabel({ y, text }: { y: number; text: string }) {
  return (
    <text x={12} y={y} textAnchor="start" dominantBaseline="central" fill="var(--color-ink-soft)" fontSize={11}>
      {text}
    </text>
  );
}

function EdgeLayer({ lines, hot }: { lines: Line[]; hot: boolean }) {
  return (
    <g aria-hidden="true">
      {hot
        ? lines.map((line) => (
            <line
              key={`${line.key}-hot`}
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
              stroke="var(--color-signal)"
              strokeOpacity={0.45}
              strokeWidth={3}
              strokeLinecap="round"
            />
          ))
        : null}
      {lines.map((line) => (
        <line
          key={line.key}
          x1={line.x1}
          y1={line.y1}
          x2={line.x2}
          y2={line.y2}
          stroke={line.color}
          strokeWidth={1}
          strokeLinecap="round"
        />
      ))}
    </g>
  );
}
