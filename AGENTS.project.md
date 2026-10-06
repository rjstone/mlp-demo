# Numeral MLP demo

Draw a digit on a 4×5 grid. A fixed multilayer perceptron classifies it as
0–9 or `?`, and the diagram shows the weights or the activations that produced
that answer.

The running site is [https://mlp-demo.vercel.app/](https://mlp-demo.vercel.app/).
Agent-facing deploy rules also live in [AGENTS.md](AGENTS.md).

## What the page is

Left side: a drawable 4×5 plate (draw, erase, clear), then a recognized-class
box and a short caption. Those two boxes together are the same width as the
plate. Right side: the network, as tall as that left column. Below: “How this
works”, with the static neuron figure floated beside the text on wide screens,
then the training-data glyphs.

The document title is “Multi-Layer Perceptron (MLP) Neural Network Numeral
Image Recognition Demo”. The words “Neural Network” stay on one line.

## Network

| | |
|---|---|
| Input | 20 binary pixels, row-major. Empty is 0, ink is 1. Not one-hot. |
| Hidden | 8 units, `tanh(bias + Σ input × weight)` |
| Output | 11 logits, then softmax. Classes `0`–`9` and `?` (index 10). |
| Parameters | 248 weights (8×20 + 11×8) and 19 biases |

Inference is `infer()` in [src/lib/mlp.ts](src/lib/mlp.ts). `w1[j][i]` is
input `i` → hidden `j`. `w2[k][j]` is hidden `j` → output `k`. The recognized
label is the class with the largest softmax probability.

`driven` is still computed (a hidden unit is driven when the absolute input
contribution is at least a quarter of the strongest one). The diagram does not
paint a path from it.

## Training

[scripts/train-mlp.mjs](scripts/train-mlp.mjs) writes
[src/lib/mlp-weights.ts](src/lib/mlp-weights.ts). Do not edit that file by
hand.

The set is hand-built 4×5 glyphs, including shifts that still fit, plus a `?`
class: a blank grid, a solid fill, bars, diagonals, and checkers. One-pixel
edits stay with their digit only when they remain closer to that digit than to
any other. Ties, specks, and random patterns far from every glyph are `?`.
There are 2^20 possible grids. The set is a small neighborhood of them, so
generalization is real and imperfect.

Training is Adam (learning rate 0.03, L2 2e-4, 700 steps) from a
feature-shaped hidden-layer start. Eight seeds are scored. A run is kept only
if every prototype is correct and the blank grid is `?`.

Eight hidden units memorize that neighborhood. They do not read handwriting.
A much larger hidden layer and a finer grid could, with enough training.

## Diagram colors

Two modes. The legend shows one of them, never both. There is no blue path.

**Weight mode** (every pixel off). Edges are weights. Hidden and output fills
are biases. Input fills are white, with a normal ink stroke. Zero on this
scale is 80% gray, `rgb(204, 204, 204)`, so a zero almost disappears. The
trained minimum is `#FF0000` and the trained maximum is `#00FF00`. Every edge
is drawn, including the gray ones.

**Activation mode** (any pixel on). The scale is split, not clamped to ±1:

- lowest value on the board: full red
- −1: half red, if the low end is below −1
- 0: white, centered on the legend
- +1: half green, if the high end is above +1
- highest value: full green

If an extreme falls inside ±1, that side runs from white to full color at the
extreme. Values past ±1 use the outer half of the bar. An input of 1 must not
look pale.

What is colored:

- Input numerals that are 1 are `#007700`. The circle fill is the activation
  color of the bit (white at 0, half-saturated green at 1 when the board max
  is above 1), not `#007700`.
- Input → hidden edges are `input × weight`. Hidden → output edges are
  `tanh(hidden) × weight`. An edge whose color is exactly white is omitted so
  it does not cover a colored edge.
- Hidden fills are the tanh outputs.
- Output fills are the pre-softmax logits. Logits often set the red and green
  ends and wash out smaller signals; that is accepted. The numerals under the
  output nodes stay softmax probabilities.
- The winning class keeps an ink stroke, not a blue one.

`diagramWeightColor` is the 80% gray weight scale. `weightColor` (50% gray at
zero) is unused by the UI. Do not point the diagram or the neuron figure at it.

## Neuron detail

[src/components/node-detail.tsx](src/components/node-detail.tsx) is a teaching
figure, not a view of the live net. The example is fixed: inputs 1, 0.5, and
0.8; weights −1.5, 0, and 2; bias 0.5. The weighted sum plus bias is 0.60, and
`tanh(0.60)` is 0.537.

Colors follow the weight scale, not the live activation scale. Zero is 80%
gray. Input numerals and the output-arrow numerals are `#007700`. Weight and
bias numerals stay black. The tanh curve’s operating point is `#007700`. The
x-axis mark at 0.60 and its label are black, at the same small size as the
other axis labels. The 0.537 label sits upper-left of that point in `#007700`.

## Share card

[public/og.jpg](public/og.jpg) is a 1200×630 JPEG of the prototype eight the
shipped weights classify at 0.97, next to that forward pass. Identity is
[src/lib/og/site.json](src/lib/og/site.json): title, description, site name,
image alt, `"card": "custom"`, and `"url": "https://mlp-demo.vercel.app/"`.

`og:*` and `twitter:*` are injected on every HTML response. Putting them in
the root route does not stick. On this Vercel host the absolute image URL
comes from `site.json` `url`, because a `*.vercel.app` request Host is not
used as an image origin.

## Deploy

Production is `main`. Routine edits go on `pre-production`. Vercel deploys
`main` to [mlp-demo.vercel.app](https://mlp-demo.vercel.app/) and deploys
`pre-production` as a Preview. A pull request into `main` is how a preview
becomes production. Do not push day-to-day work straight to `main`.

1. Edit source on `pre-production` of
   [github.com/rjstone/mlp-demo](https://github.com/rjstone/mlp-demo).
2. Do not commit `.vercel/` or a prebuilt `.vercel/output`. Vercel will skip
   the build and ship whatever bundle was baked into the commit. That is how
   production kept serving the old app after a newer push.
3. [vercel.json](vercel.json) installs with `npm install --no-audit --no-fund`.
   Do not omit dev dependencies.
4. Push `pre-production` and wait for that Preview deployment. Confirm the
   preview title and that the build log does not say it used prebuilt
   artifacts. Production does not move until a pull request from
   `pre-production` is merged into `main`.
5. This app does not enable auth and does not need `DATABASE_URL`.
