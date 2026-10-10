# ArtMentorAI

An AI painting tutor. This repository currently implements the **pattern generation
core**: any photograph in, a printable numbered paint-by-numbers pattern out.

![status](https://img.shields.io/badge/stage-core%20pipeline-blue)

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"            # editable install + pytest/ruff

# Generate a test image and run the pipeline on it
python scripts/make_sample_image.py sample.png
python -m artmentor.cli sample.png -o out/ --colors 12
```

You get four files in `out/`:

| File | What it is |
|---|---|
| `pattern.svg` | The printable pattern — white regions, outlines, numbers, color key |
| `preview.svg` | The same regions filled with their colors, so you can see the target |
| `quantized.png` | Flat raster of the reduced palette, for eyeballing palette quality |
| `palette.json` | Each number with its hex, Lab value, and share of the image |

Useful flags: `--colors` (palette size), `--min-area` (smallest paintable region),
`--max-dimension`, `--smoothing`, `--seed`. Difficulty is essentially
`--colors` and `--min-area` together.

Run the tests with `pytest` and the linter with `ruff check .`.

---

## The one architectural rule

**Everything under `artmentor/core/` is pure Python — no web framework, no
database, no HTTP.** It must stay runnable from a plain script with no server
up. The CLI exists partly as a living proof of that: if `python -m artmentor.cli`
ever stops working, the boundary has been broken.

When the API layer arrives, it will be thin and will call into `core`. If
anything in an API route ever needs `import cv2` directly, that is the bug.

```
artmentor/
  core/                    pure Python, no framework
    types.py               plain dataclasses that cross the boundary
    interfaces.py          abstract bases for every swappable capability
    color/
      space.py             sRGB <-> CIELAB
      difference.py        CIEDE2000 + the verdict bands shown to users
    imaging/
      loader.py            SECURITY BOUNDARY — all untrusted images enter here
      quantize.py          Lab k-means
      segment.py           adaptive despeckling + region merging
      labeling.py          where the number goes
      contours.py          outlines, holes, corner-preserving smoothing
    render/svg.py          SVG output
    pipeline.py            orchestration; owns PIPELINE_VERSION
  cli.py
tests/
scripts/
```

### Why capabilities sit behind interfaces

Every AI capability in this product will be replaced at least once as it matures:

| Tier | What | Used for | When |
|---|---|---|---|
| A | Deterministic CV and color science, no model | Patterns, mixing, grid, shade checks | **now** |
| B | Pretrained models, off the shelf | Subject segmentation, Loomis auto-fit | later |
| C | Hosted vision-language model | Conversational critique wording | later |
| D | Models fine-tuned on our own data | Improved critique, personalised levels | much later |

`core/interfaces.py` defines `Segmenter`, `PigmentMixer`, and `FeedbackProvider`
so a Tier B or C implementation can be swapped in behind a feature flag without
touching API, database, or frontend code.

**No custom model training is needed for the current scope, and that is the
correct answer rather than a shortcut.** Pattern generation, color mixing, and
shade comparison all have computable correct answers; training a model to
approximate arithmetic would be slower, less accurate, and impossible to debug.

---

## Notes on the implementation

A few decisions that are easy to get wrong and expensive to reverse:

**All color comparison happens in CIELAB with CIEDE2000, never in RGB.** RGB
distance does not match perception. Clustering in RGB produces palettes that look
arbitrary, and naive RGB mixing of blue and yellow gives grey rather than green.

**Label placement uses the pole of inaccessibility, not the centroid.** The
centroid of a crescent or an L-shape falls outside the shape, which puts the
printed digit in a neighbouring region. The distance transform gives both the
best interior point and the room available, which then drives font size.

**Despeckling adapts to local content.** One global smoothing kernel cannot win:
a wide neighbourhood vote is needed to straighten the wandering boundary left by
quantizing a sky gradient, but that same wide vote erases small deliberate
details like a pupil. The filter votes wide where the source image is flat and
narrow where there is real structure.

**Contour smoothing spares structural corners.** Corner-cutting cleans up
quantization zigzag and would otherwise round off every intentional corner,
turning a beak into a thumb. Noise and corners separate by scale, so a vertex is
only cut when both of its adjacent edges are short.

**Output is SVG, not raster.** It prints at any size, the files are small, and
each region arrives in the browser as an addressable DOM element — which is what
makes an interactive gallery possible later without re-architecting.

**Patterns record `pipeline_version` and their params.** Without this you cannot
answer "why does my pattern look different now?", and you cannot tell whether a
pipeline change was an improvement or a regression.

---

## Security

`core/imaging/loader.py` is the only sanctioned entry point for user images, and
it is a security boundary rather than a convenience wrapper. Untrusted images are
being fed to C-based parsers, so it enforces:

- format validated by **magic bytes**, never by file extension or client-supplied
  content type (both are attacker-controlled)
- **decompressed pixel count capped before decode** — a decompression bomb is a
  few KB of PNG that expands to tens of GB of RAM
- **EXIF stripped** — phone photos of a painting carry the GPS coordinates of the
  user's home; orientation is applied first, then all metadata is discarded
- **canonical re-encode** of every image, dropping trailing payloads and
  embedded thumbnails

Pillow and OpenCV are pinned in `requirements.txt` for this reason. Watch their
advisories and update deliberately.

Still to do before any public deployment: run pattern generation in a **job queue
with hard timeouts and memory caps** rather than in the request thread; enforce
object-level authorization at the query layer for the gallery; serve images via
short-lived signed URLs; and decide the under-13 policy, which has COPPA and
GDPR-K consequences for the whole data architecture.

---

## Licensing note

For pigment mixing, use **[Spectral.js](https://github.com/rvanwijnen/spectral.js)
(MIT)**, not Mixbox. Mixbox is the better-known Kubelka-Munk implementation and
is **CC BY-NC 4.0 — non-commercial only**; the `pymixbox` PyPI package inherits
that restriction. A commercial license can be negotiated with Secret Weapons, but
that should be a deliberate decision rather than something discovered at launch.

---

## Roadmap

- [x] Pattern generation: quantization, regions, labels, SVG
- [x] Color core: Lab, CIEDE2000, verdict bands, mix-direction hints
- [x] Hardened image loading
- [ ] Paint database (brand, line, measured Lab, opacity, tinting strength)
- [ ] `PigmentMixer` implementation — inverse mixing via Spectral.js
- [ ] Printable calibration card + photo white-balance correction
- [ ] Shade checking from a photo (`verdict_for` is already waiting for it)
- [ ] FastAPI layer + job queue
- [ ] Grid overlay, then manual Loomis overlay
- [ ] Tier A critique: fidelity metrics against a reference
- [ ] Gallery as a progress timeline
