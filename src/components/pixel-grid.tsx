import { useRef, type KeyboardEvent, type PointerEvent } from "react";
import { INPUTS } from "@/lib/mlp";

const COLS = 4;
const ROWS = 5;

type Mode = "draw" | "erase";

type PixelGridProps = {
  bits: readonly number[];
  mode: Mode;
  onChange: (index: number, value: 0 | 1) => void;
};

function erasing(event: PointerEvent, mode: Mode): boolean {
  return mode === "erase" || event.button === 2 || (event.buttons & 2) !== 0 || event.ctrlKey;
}

export function PixelGrid({ bits, mode, onChange }: PixelGridProps) {
  const gridRef = useRef<HTMLDivElement>(null);

  function indexAt(clientX: number, clientY: number): number | null {
    const grid = gridRef.current;
    if (!grid) return null;
    const style = getComputedStyle(grid);
    const padL = Number.parseFloat(style.paddingLeft) || 0;
    const padT = Number.parseFloat(style.paddingTop) || 0;
    const colGap = Number.parseFloat(style.columnGap) || 0;
    const rowGap = Number.parseFloat(style.rowGap) || 0;
    const cell = grid.querySelector("[data-i]");
    if (!(cell instanceof HTMLElement)) return null;
    const rect = grid.getBoundingClientRect();
    const cellRect = cell.getBoundingClientRect();
    const localX = clientX - rect.left - padL;
    const localY = clientY - rect.top - padT;
    const col = Math.floor(localX / (cellRect.width + colGap));
    const row = Math.floor(localY / (cellRect.height + rowGap));
    if (col < 0 || col >= COLS || row < 0 || row >= ROWS) return null;
    const insideX = localX - col * (cellRect.width + colGap);
    const insideY = localY - row * (cellRect.height + rowGap);
    if (insideX < 0 || insideX > cellRect.width || insideY < 0 || insideY > cellRect.height) return null;
    const index = row * COLS + col;
    return index >= 0 && index < INPUTS ? index : null;
  }

  function paint(event: PointerEvent) {
    const index = indexAt(event.clientX, event.clientY);
    if (index === null) return;
    onChange(index, erasing(event, mode) ? 0 : 1);
  }

  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 && event.button !== 2) return;
    event.preventDefault();
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Untrusted events and some pens cannot capture; painting still works.
    }
    paint(event);
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    if (event.buttons === 0) return;
    paint(event);
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const raw = target.getAttribute("data-i");
    if (raw === null) return;
    const index = Number(raw);
    if (!Number.isInteger(index)) return;

    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      onChange(index, bits[index] ? 0 : 1);
      return;
    }

    const col = index % COLS;
    const row = Math.floor(index / COLS);
    let next = index;
    if (event.key === "ArrowRight" && col < COLS - 1) next = index + 1;
    else if (event.key === "ArrowLeft" && col > 0) next = index - 1;
    else if (event.key === "ArrowDown" && row < ROWS - 1) next = index + COLS;
    else if (event.key === "ArrowUp" && row > 0) next = index - COLS;
    else return;
    event.preventDefault();
    const cell = gridRef.current?.querySelector(`[data-i="${next}"]`);
    if (cell instanceof HTMLElement) cell.focus();
  }

  return (
    <div
      ref={gridRef}
      role="grid"
      aria-label="4 by 5 pixel grid. Draw with the left button, erase with the right button or Control-click."
      className={`flex flex-col gap-1.5 p-2 touch-none select-none ${mode === "draw" ? "cursor-crosshair" : "cursor-cell"}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onContextMenu={(event) => event.preventDefault()}
      onKeyDown={onKeyDown}
    >
      {Array.from({ length: ROWS }, (_, row) => (
        <div role="row" key={row} className="flex gap-1.5">
          {Array.from({ length: COLS }, (_, col) => {
            const index = row * COLS + col;
            const on = bits[index] === 1;
            return (
              <div
                key={index}
                role="gridcell"
                aria-selected={on}
                aria-rowindex={row + 1}
                aria-colindex={col + 1}
                aria-label={`Row ${row + 1}, column ${col + 1}, ${on ? "black" : "empty"}`}
                data-i={index}
                tabIndex={0}
                className={`size-16 rounded-lg transition-colors duration-150 motion-reduce:transition-none focus-visible:outline-2 focus-visible:outline-solid focus-visible:outline-offset-2 focus-visible:outline-signal ${on ? "bg-pixel" : "bg-well"}`}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}
