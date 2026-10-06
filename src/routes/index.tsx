import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Eraser, Pencil } from "lucide-react";
import { HowItWorks } from "@/components/how-it-works";
import { MlpDiagram } from "@/components/mlp-diagram";
import { PixelGrid } from "@/components/pixel-grid";
import { TrainingData } from "@/components/training-data";
import { INPUTS, LABELS, formatProb, infer } from "@/lib/mlp";

export const Route = createFileRoute("/")({ component: Home });

type Mode = "draw" | "erase";

function emptyBits(): number[] {
  return Array.from({ length: INPUTS }, () => 0);
}

function Home() {
  const [bits, setBits] = useState(emptyBits);
  const [mode, setMode] = useState<Mode>("draw");
  const net = useMemo(() => infer(bits), [bits]);
  const label = LABELS[net.winner];
  const probability = net.probs[net.winner];
  const inked = bits.some((bit) => bit === 1);

  function setCell(index: number, value: 0 | 1) {
    setBits((prev) => {
      if (prev[index] === value) return prev;
      const next = prev.slice();
      next[index] = value;
      return next;
    });
  }

  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-6 md:px-8 md:py-10">
      <header className="mb-8 max-w-2xl">
        <h1 className="text-balance text-3xl font-medium tracking-tight md:text-4xl">
          4×5 Perceptron
        </h1>
        <p className="mt-2 text-pretty text-muted">
          Draw a digit. The network beside the grid is the one that classifies it.
        </p>
      </header>

      <div className="grid items-start gap-8 lg:grid-cols-[auto_minmax(0,1fr)] lg:items-stretch">
        <section className="flex w-fit max-w-full flex-col gap-5" aria-labelledby="draw-heading">
          <h2 id="draw-heading" className="sr-only">
            Drawing
          </h2>
          <div className="w-fit rounded-2xl bg-paper p-0 shadow-plate">
            <PixelGrid bits={bits} mode={mode} onChange={setCell} />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div role="group" aria-label="Brush" className="flex rounded-full bg-surface p-1">
              <button
                type="button"
                aria-pressed={mode === "draw"}
                onClick={() => setMode("draw")}
                className={`flex h-11 items-center gap-2 rounded-full px-4 text-sm font-medium focus-visible:outline-2 focus-visible:outline-solid focus-visible:outline-offset-2 focus-visible:outline-signal ${mode === "draw" ? "bg-fg text-bg" : "text-muted hover:text-fg"}`}
              >
                <Pencil className="size-4" aria-hidden="true" />
                Draw
              </button>
              <button
                type="button"
                aria-pressed={mode === "erase"}
                onClick={() => setMode("erase")}
                className={`flex h-11 items-center gap-2 rounded-full px-4 text-sm font-medium focus-visible:outline-2 focus-visible:outline-solid focus-visible:outline-offset-2 focus-visible:outline-signal ${mode === "erase" ? "bg-fg text-bg" : "text-muted hover:text-fg"}`}
              >
                <Eraser className="size-4" aria-hidden="true" />
                Erase
              </button>
            </div>
            <button
              type="button"
              onClick={() => setBits(emptyBits())}
              className="h-11 rounded-full border border-line px-4 text-sm font-medium text-fg hover:bg-surface focus-visible:outline-2 focus-visible:outline-solid focus-visible:outline-offset-2 focus-visible:outline-signal"
            >
              Clear
            </button>
          </div>

          <div
            className="grid w-[calc(4*4rem+3*0.375rem+1rem)] grid-cols-2 gap-2"
            aria-live="polite"
          >
            <div className="flex flex-col items-center justify-center rounded-2xl bg-paper px-3 py-4 text-center text-ink shadow-plate">
              <p className="font-mono text-xs tracking-widest text-ink-soft uppercase">Recognized</p>
              <p className="mt-2 font-mono text-7xl leading-none font-medium tabular-nums">{label}</p>
              <p className="mt-3 font-mono text-2xl tabular-nums text-ink-soft">{formatProb(probability)}</p>
            </div>
            <div className="flex items-center justify-center rounded-2xl bg-paper px-3 py-4 text-center shadow-plate">
              <p className="text-sm leading-relaxed text-pretty text-ink-soft">
                {inked
                  ? "The class with the largest softmax probability."
                  : "No pixel is on, so every weight drops out. Only the biases remain."}
              </p>
            </div>
          </div>
        </section>

        <section className="flex min-h-0 min-w-0 flex-col lg:h-full" aria-labelledby="net-heading">
          <h2 id="net-heading" className="sr-only">
            Network
          </h2>
          <div className="flex h-full min-h-0 flex-1 flex-col overflow-hidden rounded-2xl bg-paper text-ink shadow-plate">
            <MlpDiagram bits={bits} net={net} />
          </div>
        </section>
      </div>

      <HowItWorks />
      <TrainingData />
    </main>
  );
}
