import { TRAINING_CELLS } from "@/lib/glyphs";

function MiniGlyph({ rows }: { rows: readonly string[] }) {
  const pixels = rows.join("");
  return (
    <div className="grid h-fit w-fit grid-cols-4 gap-0.5 self-start" aria-hidden="true">
      {pixels.split("").map((ch, i) => (
        <span key={i} className={`size-2.5 rounded-[2px] ${ch === "#" ? "bg-pixel" : "bg-well"}`} />
      ))}
    </div>
  );
}

export function TrainingData() {
  return (
    <section className="mt-12 border-t border-line pt-8" aria-labelledby="training-heading">
      <h2 id="training-heading" className="text-balance text-xl font-medium tracking-tight">
        Training data
      </h2>
      <div className="mt-4 flex flex-col gap-4 text-base leading-relaxed text-pretty">
        <p>
          The cells show the hand-built 4×5 glyphs, including shifts that still fit, and the designed
          unknowns under ?: a blank grid, a solid fill, bars, diagonals, and checkers. One-pixel edits and
          distant random patterns were trained too, and are not drawn here.
        </p>
        <p>
          Each pixel is on or off, so these 20 bits make 2^20 = 1,048,576 possible grids. The training
          set is only a few of them. A one-pixel edit was kept with its digit only when it stayed closer
          to that digit than to any other. Ties went to ?. So did specks and random patterns far from
          every glyph. A grid that was never shown can still be recognized: if it is close to a learned
          glyph, it drives the same hidden units, and the softmax can name that digit. That is
          generalization, and it is not perfect. A careful digit, including a one-pixel slip, should land
          correctly. Other unseen grids will be claimed by the wrong digit. Eight hidden units memorize a
          small neighborhood. They do not read handwriting in general.
        </p>
      </div>
      <div className="mt-6 grid grid-cols-4 items-start gap-3">
        {TRAINING_CELLS.map((cell) => (
          <figure
            key={cell.label}
            className={`rounded-2xl bg-paper px-3 py-4 text-ink shadow-plate ${cell.label === "?" ? "col-span-2" : ""}`}
          >
            <figcaption className="text-center font-mono text-lg leading-none font-medium">
              {cell.label}
            </figcaption>
            <div
              className={
                cell.label === "?"
                  ? "mt-3 grid grid-flow-col grid-rows-2 items-start justify-center gap-x-1.5 gap-y-2"
                  : "mt-3 flex min-h-[7.75rem] flex-wrap content-start justify-center gap-2"
              }
            >
              {cell.glyphs.map((rows) => (
                <MiniGlyph key={rows.join("")} rows={rows} />
              ))}
            </div>
          </figure>
        ))}
      </div>
    </section>
  );
}
