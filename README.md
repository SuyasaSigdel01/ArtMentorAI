# ArtMentorAI

An AI painting tutor. Give it a photograph and it produces a printable
paint-by-numbers pattern, works out how to mix every colour in it from the
paints you own, and checks the paint you actually mixed from a photo.

**169 tests.** No model training, no GPU, no dataset — see
[Why there is no model](#why-there-is-no-model).

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Generate a pattern from an image:

```bash
python scripts/make_sample_image.py sample.png
python -m artmentor.cli sample.png -o out/ --colors 12
```

Run the API and open http://127.0.0.1:8000/docs for an interactive page
generated from the code:

```bash
python -m uvicorn artmentor.api.app:app --reload
```

Tests and linting:

```bash
pytest -q
ruff check .
```

---

## What it does

### Photo to printable pattern

Quantizes an image to N paintable colours, builds contiguous regions, merges
away anything too small to paint, and renders numbered outlines as SVG.

`python -m artmentor.cli photo.jpg -o out/ --colors 12` writes `pattern.svg`
(the printable outlines), `preview.svg` (the same regions filled),
`quantized.png`, and `palette.json`. Difficulty is `--colors` and `--min-area`
together.

### Mixing recipes for every number

For each colour in the pattern, which paints to combine and in what ratio:

```python
from artmentor.core.recipes import recipes_for_pattern, format_mixing_guide
from artmentor.core.paints import STARTER_ACRYLICS

print(format_mixing_guide(recipes_for_pattern(pattern, STARTER_ACRYLICS)))
```

```
   1  #4e8a46  8 parts Cadmium Yellow + 4 parts Yellow Ochre + 1 part Phthalo Blue   18%  match
   4  #b3d7ee  40 parts Titanium White + 1 part Phthalo Blue (green shade)           17%  match
~  5  #3a6b38  8 parts Cadmium Yellow + 2 parts Phthalo Blue + 1 part Burnt Umber    13%  close
```

A `~` means the palette cannot actually reach that colour and this is the
closest available. Saying so matters: a beginner who follows good instructions
and gets a visibly different colour concludes they can't paint.

### Checking a mix from a photo

```python
from artmentor.core.swatch import check_shade

result = check_shade(photo, target_lab, white_region=grey_card_box)
print(result.message())   # "Close. add more yellow would nail it."
```

### HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /paints` | the paint catalog, for a "what do you own?" picker |
| `POST /patterns` | image → pattern SVG, palette, and mixing recipes |
| `POST /shade-check` | photo of a mix + target → verdict and correction |

---

## The one architectural rule

**Everything under `artmentor/core/` is pure Python — no web framework, no
database, no HTTP.** It must stay runnable from a plain script with no server.
The CLI is a living proof of that, and `tests/test_api.py` enforces it
mechanically: if `import cv2` ever appears in the API layer, that test fails.

```
artmentor/
  core/                      pure Python, no framework
    types.py                 plain dataclasses that cross the boundary
    interfaces.py            abstract bases for every swappable capability
    pipeline.py              orchestration; owns PIPELINE_VERSION
    paints.py                the paint catalog
    recipes.py               joins patterns to mixing recipes
    swatch.py                reads paint colour out of a photograph
    color/
      space.py               sRGB <-> CIELAB
      difference.py          CIEDE2000 and the verdicts shown to users
      spectral.py            reflectance recovery, Kubelka-Munk K/S
      mixing.py              forward: paints + ratios -> colour
      inverse.py             inverse: colour -> paints + ratios
    imaging/
      loader.py              SECURITY BOUNDARY — all untrusted images enter here
      quantize.py            Lab k-means
      segment.py             adaptive despeckling, region merging
      labeling.py            where the number goes
      contours.py            outlines, holes, corner-preserving smoothing
    render/svg.py            SVG output
  api/app.py                 FastAPI. Thin. Calls core, never imports cv2.
  cli.py
tests/                       169 tests
scripts/
```

Every AI capability sits behind an interface in `core/interfaces.py`
(`Segmenter`, `PigmentMixer`, `FeedbackProvider`) so a better implementation
can be swapped in behind a feature flag without touching the API, the database,
or a frontend.

---

## Why there is no model

Nothing here is trained, and that is the correct answer rather than a shortcut.

Pattern generation, colour mixing and shade comparison all have computable
correct answers. Training a model to approximate arithmetic would be slower,
less accurate, unexplainable to a user, and impossible to debug. The only
genuine ML problem in the product vision is critiquing a finished painting, and
on day one there is no training data for it.

The intended progression:

| Tier | What | Status |
|---|---|---|
| A | Deterministic CV and colour science | **built** |
| B | Pretrained models off the shelf (segmentation, pose) | later |
| C | Hosted vision-language model for conversational critique | later |
| D | Models fine-tuned on data this product generates | much later |

The asset worth building is the dataset Tier D needs. Every (target colour →
photographed mix → correction → next attempt) chain is a labelled training
example nobody else has. Logging that is worth doing long before it is useful.

---

## Known limitations

Documented here and in the code, rather than discovered later.

**The paint catalog is eyeballed, not measured.** Every colour in
`core/paints.py` is an approximation of a typical student acrylic. Good enough
to develop against; not good enough to give someone advice about their actual
tubes. Replacing it means photographing swatches under controlled light beside
a calibrated reference and recording the Lab values.

**Mixing uses single-constant Kubelka-Munk.** Full K-M tracks absorption and
scattering separately, and both cannot be recovered from an sRGB value, so
every pigment is assumed to scatter equally. That is badly wrong for titanium
white. Each paint's `tinting_strength` compensates, which is why white needs a
value around 6, but saturated reds still tint less than real paint does.

**Inferred spectra drift on selective pigments.** A spectrum reconstructed from
three numbers is broader than a real pigment's. Warm mixes behave well;
ultramarine plus yellow comes out teal where it should be olive. Pale tints are
the weak spot — cream reaches only about dE 5.

All three have the same fix: measured paint spectra.

**The API is not production-ready under load.** Endpoints run in FastAPI's
threadpool, with no timeout, no memory ceiling and no backpressure. A job queue
(Celery or RQ) is required before this faces real users.

**Nothing persists.** No users, no gallery, no progress tracking.

---

## Security

`core/imaging/loader.py` is the only sanctioned entry point for user images and
is a security boundary, not a convenience wrapper. Untrusted images are being
fed to C-based parsers, so it enforces:

- format validated by **magic bytes**, never file extension or client-supplied
  content type — both are attacker-controlled
- **decompressed pixel count capped before decode** — a decompression bomb is a
  few KB of PNG that expands to tens of GB of RAM
- **EXIF stripped** — phone photos carry the GPS coordinates of the user's home;
  orientation is applied first, then all metadata discarded
- **canonical re-encode** of every image, dropping trailing payloads

Pillow and OpenCV are pinned for this reason; watch their advisories.

Before any public deployment: the job queue above, object-level authorization
enforced at the query layer, short-lived signed URLs for images, and a decision
on under-13 users, which has COPPA and GDPR-K consequences for the whole data
model.

---

## Notes on the implementation

Decisions that are easy to get wrong and expensive to reverse.

**All colour comparison happens in CIELAB with CIEDE2000, never RGB.** RGB
distance does not match perception. Averaging blue and yellow in RGB gives grey
where paint gives green.

**Label placement uses the pole of inaccessibility, not the centroid.** The
centroid of a crescent falls outside the shape, putting the printed digit in a
neighbouring region. The distance transform gives both the best interior point
and the room available, which then drives font size.

**Despeckling adapts to local content.** A wide neighbourhood vote is needed to
straighten the boundary left by quantizing a sky gradient, but the same wide
vote erases a pupil. The filter votes wide where the source is flat and narrow
where there is real structure.

**Contour smoothing spares structural corners.** Noise and intentional corners
separate by scale, so a vertex is only cut when both adjacent edges are short.
Douglas-Peucker is the wrong tool — it preserves extreme points, keeping zigzag
tips while faceting smooth curves.

**Inverse mixing enumerates executable ratios.** Optimizing concentrations
continuously yields "0.6187 ultramarine", useless at a palette, and rounding
afterwards lands somewhere never evaluated — so the reported accuracy would be
fiction. Searching whole-number ratios means every candidate is a real recipe.

**The photo reference card must be grey, not white.** White sits near the
sensor's ceiling, so warm light clips its red channel and the correction fails
silently. Measured under simulated tungsten: a white patch took a 5.3 dE error
to 4.6; a mid-grey patch took it to 0.9.

**Patterns record `pipeline_version` and their params.** Without this you cannot
explain why a regenerated pattern looks different, or tell an improvement from
a regression.

---

## Licensing note

Pigment mixing is an independent implementation of published Kubelka-Munk
theory, with no licence encumbrance.

If its quality ceiling becomes limiting, port
[Spectral.js](https://github.com/rvanwijnen/spectral.js) (MIT). Do **not** reach
for Mixbox, the better-known implementation: it is CC BY-NC 4.0, non-commercial
only, and the `pymixbox` PyPI package inherits that restriction. A commercial
licence can be negotiated with Secret Weapons, but that should be a deliberate
decision rather than something discovered at launch.

---

## Roadmap

- [x] Pattern generation: quantization, regions, labels, SVG
- [x] Colour core: Lab, CIEDE2000, verdict bands, mix-direction hints
- [x] Hardened image loading
- [x] Forward pigment mixing (Kubelka-Munk) + starter catalog
- [x] Inverse mixing: target colour → executable recipe
- [x] Pattern → per-number mixing guide with honest accuracy flags
- [x] Photo shade checking with lighting correction
- [x] HTTP API
- [ ] Measured paint data to replace the approximate catalog
- [ ] Printable grey calibration card
- [ ] Job queue, persistence, and auth
- [ ] Grid overlay, then manual Loomis overlay
- [ ] Tier A critique: fidelity metrics against a reference
- [ ] Gallery as a progress timeline