# ArtMentorAI

An AI painting tutor that turns a photograph into a numbered pattern, tells you
which paints to mix to hit each colour, and measures how close your attempt
came — in colour, in value, and over time.

It runs on a laptop with no GPU, no API keys, and no account.

```
pip install -e ".[dev]"
uvicorn artmentor.api.app:app --reload
```

Then open <http://127.0.0.1:8000>.

---

## The decision that explains the rest of the codebase

**No model is trained, and nothing calls out to a language model.**

That is not a shortcut taken to ship faster. Every feature here has a
computable answer:

| Feature | How it is solved |
|---|---|
| Paint-by-numbers pattern | k-means in CIELAB, connected components, contour tracing |
| "What do I mix to get this colour?" | Kubelka–Munk subtractive mixing, inverted by enumeration |
| "Is this shade right?" | CIEDE2000 against a white-balanced photograph |
| "Did my painting improve?" | Fidelity metrics against the reference, stored over time |

Training a network to approximate these would be slower, less accurate, and
impossible to debug. When a recipe is wrong today, you can read the spectrum
that produced it. `core/interfaces.py` defines an abstract class for every
capability so a model can be swapped in later — segmentation is the obvious
first candidate — without touching anything around it.

**Colour is never compared in RGB.** The average of blue and yellow in RGB is
exactly `#808080`, a dead grey; on a palette it is green. Every comparison in
this codebase happens in CIELAB, and every judgement a learner acts on uses
CIEDE2000.

---

## What works today

- **Patterns.** Photograph in, numbered regions out, as SVG with every region
  addressable so the UI can highlight one on hover. Content-adaptive
  despeckling keeps a gradient sky clean without dissolving an eye.
- **Mixing.** Real subtractive pigment mixing via Kubelka–Munk, with
  reflectance recovered from sRGB by constrained optimisation. The inverse
  problem — "which paints, in what whole-number ratio, hit this target?" — is
  solved by enumerating ratios a human can actually measure out by hand.
- **Shade checking.** Photograph a mix, get told how far off it is and which
  way to push it. Print the greyscale calibration card from
  `/calibration-card.svg` and include it in frame; without it the answer still
  comes back but says it is a rough read.
- **Grid method.** A grid overlay for copying, with the cell size in
  millimetres on A4 — the number you actually use at a drawing board — and a
  warning when the image's proportions will not divide into square cells.
- **Feedback.** Four measurements against the reference, and exactly one
  instruction. See "What this refuses to do" below.
- **Studio.** Save a piece, photograph it each sitting, and see the numbers
  move. Everything is stored on your machine.
- **Mediums and skill levels.** Not advice text: constraints the pipeline
  obeys. Watercolour drops white from the palette because the paper is the
  white; graphite gets a value map instead of a colour one; a beginner gets
  fewer colours and two-pigment mixes.

**Not built yet:** the Loomis head construction. The grid method is the only
sketching mode so far.

---

## What this refuses to do

`core/critique.py` does **not** judge whether a painting is good. There is no
ground truth for that, professionals disagree, and a model trained on someone
else's taste would hand a beginner a confident verdict it has no right to. A
wrong discouraging critique is the exact opposite of what a tutor is for.

What it measures instead is **fidelity to a stated target** — where your
painting differs from what you were copying, and by how much. Every number is
computed, explainable, and checkable against your own eyes. That reframing is
also what makes progress trackable: "your values were 12% off last week and 6%
off today" is a real statement in a way that an aesthetic score never is.

A test enforces this. If anyone adds "beautiful", "talented" or a score out of
ten, the suite fails and makes them read the docstring explaining why.

---

## Architecture

One rule, and a test that enforces it:

> `artmentor/core/` is pure Python. No web framework, no database, no HTTP.

`tests/test_api.py::test_api_does_not_reimplement_core` reads `app.py` as text
and fails if `import cv2`, `from skimage` or `delta_E` ever appear in it. The
API layer translates; it does not compute.

This is what lets the colour science be tested in milliseconds with no
fixtures, and what will let the storage layer move to Postgres without touching
a single formula.

```
artmentor/
├── core/                  pure computation — no framework, no I/O beyond images
│   ├── color/
│   │   ├── space.py       sRGB ↔ CIELAB
│   │   ├── difference.py  CIEDE2000
│   │   ├── spectral.py    reflectance recovery, 380–730nm at 10nm
│   │   ├── mixing.py      Kubelka–Munk forward mixing
│   │   └── inverse.py     "which paints make this?" by ratio enumeration
│   ├── imaging/
│   │   ├── loader.py      ← the security boundary. Read this one first.
│   │   ├── quantize.py    bilateral presmooth + k-means in Lab
│   │   ├── segment.py     content-adaptive despeckling, region merging
│   │   ├── labeling.py    label placement by pole of inaccessibility
│   │   └── contours.py    corner-preserving smoothing
│   ├── render/            svg.py, grid.py, calibration.py
│   ├── paints.py          the starter catalogue (12 student acrylics)
│   ├── recipes.py         a pattern's palette → mixing instructions
│   ├── swatch.py          white balance and shade comparison
│   ├── measure.py         tooling to replace the catalogue with real data
│   ├── critique.py        fidelity metrics
│   ├── mediums.py         medium and skill level as pipeline constraints
│   ├── pipeline.py        photograph → Pattern
│   ├── types.py           the dataclasses that cross module boundaries
│   └── interfaces.py      abstract classes for every swappable capability
├── store/
│   └── gallery.py         SQLite + content-addressed media
├── api/
│   ├── app.py             FastAPI; translation only
│   └── static/index.html  the whole frontend, one file, no build step
└── cli.py                 the pipeline with no server and no database
```

### Security

`core/imaging/loader.py` is the only sanctioned way an image enters the system.
Every image is untrusted input being fed to a C parser, so it enforces four
things that are easy to skip and expensive to retrofit:

1. Format is validated by **magic bytes**, never by filename or content type —
   both are attacker-controlled.
2. Decompressed pixel count is capped **before decode**. A decompression bomb
   is a few kilobytes of PNG that expands to tens of gigabytes of RAM.
3. **EXIF is stripped.** A phone photo of a painting carries the GPS
   coordinates of the painter's home. Orientation is applied, then all metadata
   is discarded.
4. Every image is **re-encoded** through a canonical RGB path, dropping
   trailing payloads and embedded thumbnails.

`encode_image()` is the inverse, and the gallery stores *that* — never the
bytes a client uploaded. A test uploads a JPEG with a marker buried in its
metadata and asserts those bytes are absent from what the server returns.

### Storage

One SQLite file and a `media/` folder, so the whole gallery can be backed up by
copying one directory. Images are stored under the SHA-256 of their bytes,
never under a name anyone supplied — that removes path traversal as a class of
bug rather than defending against it, and deduplicates for free. Deletion is
refcounted across both tables.

Measurements are stored as JSON rather than columns, so `critique.py` can grow
a fifth measure without a migration and without silently dropping measures
written by a newer version.

Attempts are critiqued **when saved**, not when read. `critique.py` will
change, and a learner's history must keep saying what it said at the time.

---

## The API

Interactive docs at `/docs` once the server is running.

```
GET    /health
GET    /paints                              the starter catalogue
GET    /mediums  /levels  /guidance         what a medium and level change
POST   /patterns                            image → numbered pattern + recipes
GET    /mix                                 predict the colour of a mix
POST   /grid                                grid overlay for copying
POST   /critique                            reference + attempt → measurements
POST   /shade-check                         photo of a mix → verdict
GET    /calibration-card.svg                the printable grey card

POST   /gallery/pieces                      start a piece
GET    /gallery/pieces                      the wall
GET    /gallery/pieces/{id}                 one piece and its attempts
PATCH  /gallery/pieces/{id}                 rename
DELETE /gallery/pieces/{id}
POST   /gallery/pieces/{id}/attempts        photo → critiqued and saved
DELETE /gallery/attempts/{id}
GET    /gallery/pieces/{id}/progress        latest attempt vs the first
GET    /gallery/images/{digest}[/thumb]
```

The gallery lives in `~/.artmentor` by default; set `ARTMENTOR_DATA` to move it.

### Without a server

```
python -m artmentor.cli photo.jpg -o out/ --colors 12
```

This exists so the pipeline can be exercised with no server, no database and no
frontend. If it stops working, the `core` boundary has been broken.

---

## Tests

```
pytest -q        # 397 tests
ruff check .
```

| File | Tests | File | Tests |
|---|---|---|---|
| `test_api.py` | 63 | `test_critique.py` | 26 |
| `test_gallery.py` | 45 | `test_measure.py` | 22 |
| `test_mediums.py` | 44 | `test_color.py` | 22 |
| `test_mixing.py` | 39 | `test_pipeline.py` | 21 |
| `test_grid.py` | 30 | `test_inverse.py` | 20 |
| `test_swatch.py` | 19 | `test_calibration.py` | 18 |
| `test_loader.py` | 16 | `test_recipes.py` | 12 |

The test count doubles as a drift detector: if your local number is lower than
397, a file did not land where you thought it did.

---

## Known limitations

Written down because they are real, not because they are small.

**The paint catalogue is eyeballed.** The twelve starter acrylics are
approximate sRGB values converted to Lab, not measured pigments. Four of the
limitations below trace back to this single fact. `core/measure.py` and the
calibration card exist to fix it: photograph real swatches under controlled
light and `to_catalog_entry()` emits a replacement entry. It is an afternoon
with paint, not a coding task.

**No job queue.** Routes are declared `def`, so Starlette runs them in a worker
thread and the event loop keeps serving. That is enough for development and
light use and is *not* enough for production — there is no timeout, no memory
ceiling and no backpressure, so a flood of large images will exhaust the pool.
Celery or RQ before any public deployment.

**Critique assumes alignment.** The attempt is scaled to the reference's
dimensions and compared directly, which holds for a photograph taken square-on
and cropped to the artwork. Perspective correction via feature matching would
lift that restriction and is a sizeable piece of work.

**Monochrome palettes can duplicate a tone.** A graphite pattern may give two
numbers to values no eye can separate. Merging them needs the region merger to
understand value adjacency.

**Single user, no accounts.** There is no owner column and no authentication.
Adding accounts means a `user_id` on two tables and a filter on every query —
which is why every query goes through a method rather than being written inline
at the call site.

**Dry shift is estimated.** The value change on drying in `mediums.py` is from
the literature and from how these paints are generally described, not measured
here. Same category as the catalogue.

---

## Licensing note

[Mixbox](https://github.com/scrtwpns/mixbox) is the best-known pigment mixing
library and is **CC BY-NC 4.0** — non-commercial. `pymixbox` inherits that
licence. For a commercial product that is a problem which would surface exactly
at launch, so the Kubelka–Munk implementation in `core/color/` is written from
the published model instead, with a single-constant approximation corrected by
a per-paint `tinting_strength`.

Check the licence of anything you add here against the same standard.

---

## Performance notes

Rough figures from a development laptop — they will differ on yours, and the
ratios matter more than the absolutes.

- `srgb_to_reflectance` is the expensive call and is `lru_cache`d.
- Inverse mixing caches its candidate table per palette. The first colour costs
  a few seconds, every subsequent colour about 10ms. Almost all of that first
  cost is spectral recovery, not the search.
- CIEDE2000 over 41,000 pixels measured at 22ms — which is why the earlier
  assumption that it was the bottleneck was wrong, and the real cost turned out
  to be rebuilding the candidate table. Measure before optimising.

---

## Requirements

Python 3.11+. Dependencies in `pyproject.toml`: numpy, scipy,
opencv-python-headless, scikit-image, scikit-learn, colour-science, Pillow;
FastAPI and uvicorn for the server; pytest and ruff for development.