"""Tests for the HTTP layer.

These check translation, not color science: that uploads are validated, bad
input is rejected with the right status, and the JSON carries what a frontend
needs. The maths is already covered by the core tests, and duplicating it here
would mean two places to update every time a threshold moves.

`test_api_does_not_reimplement_core` guards the project's one architectural
rule — if the API layer ever grows its own image processing, that test fails.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from artmentor.api.app import app, gallery_store
from artmentor.store import Gallery

client = TestClient(app)


@pytest.fixture
def gallery(tmp_path):
    """A gallery of its own for each test, injected in place of the real one.

    Through the dependency rather than by reaching into the module, so the
    tests exercise the same wiring the server uses.
    """
    store = Gallery(tmp_path / "studio")
    app.dependency_overrides[gallery_store] = lambda: store
    yield store
    app.dependency_overrides.clear()


def new_piece(title: str = "Harbour", **extra) -> dict:
    response = client.post(
        "/gallery/pieces",
        files={"image": ("ref.png", scene(240, 240), "image/png")},
        data={"title": title, **extra},
    )
    assert response.status_code == 201, response.text
    return response.json()


def encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def scene(width: int = 200, height: int = 240) -> bytes:
    """A small image with a few distinct regions, like a simple reference photo."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = (130, 180, 225)
    arr[height // 3 : 2 * height // 3] = (80, 140, 70)
    arr[2 * height // 3 :] = (200, 90, 60)
    rng = np.random.default_rng(0)
    arr = np.clip(arr + rng.normal(0, 2.0, arr.shape), 0, 255).astype(np.uint8)
    return encode(Image.fromarray(arr))


def swatch_photo(hex_color: str = "#4a7a3a", reference: bool = True) -> bytes:
    """A photo of mixed paint, optionally with a grey card along the top."""
    from artmentor.core.types import RGBColor

    colour = RGBColor.from_hex(hex_color)
    arr = np.full((200, 200, 3), (90, 90, 95), dtype=np.uint8)
    arr[60:160, 50:150] = (colour.r, colour.g, colour.b)
    if reference:
        arr[0:30, :] = (200, 200, 200)
    return encode(Image.fromarray(arr))


class TestHealthAndCatalog:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_paints_lists_the_catalog(self):
        response = client.get("/paints")
        assert response.status_code == 200
        paints = response.json()
        assert len(paints) == 12
        ids = {p["id"] for p in paints}
        assert {"ti-white", "ultramarine", "cad-red"} <= ids
        assert all(p["hex"].startswith("#") for p in paints)


class TestCreatePattern:
    def test_returns_a_pattern_with_recipes(self):
        response = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"colors": 5, "max_dimension": 240},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["colors"] <= 5
        assert body["regions"] >= 1
        assert body["svg"].startswith("<?xml")
        assert len(body["palette"]) == body["colors"]

        first = body["palette"][0]
        assert first["number"] == 1
        assert first["mix"], "recipes were requested"
        assert "part" in first["instructions"]
        assert isinstance(first["achievable"], bool)

    def test_recipes_can_be_skipped(self):
        response = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"colors": 4, "max_dimension": 240, "include_recipes": False},
        )
        assert response.status_code == 200
        assert response.json()["palette"][0]["mix"] is None

    def test_honours_a_restricted_palette(self):
        """A learner can only be told to mix paints they own."""
        response = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={
                "colors": 4,
                "max_dimension": 240,
                "paints": "ti-white,ultramarine,cad-red",
            },
        )
        assert response.status_code == 200
        allowed = {"ti-white", "ultramarine", "cad-red"}
        for colour in response.json()["palette"]:
            assert {step["paint_id"] for step in colour["mix"]} <= allowed

    def test_rejects_unknown_paint_ids(self):
        response = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"colors": 3, "max_dimension": 240, "paints": "unobtainium"},
        )
        assert response.status_code == 400
        assert "unobtainium" in response.json()["detail"]

    def test_rejects_a_non_image(self):
        response = client.post(
            "/patterns",
            files={"image": ("evil.png", b"not an image at all, but long enough", "image/png")},
            params={"colors": 4},
        )
        assert response.status_code == 400

    def test_rejects_an_empty_upload(self):
        response = client.post("/patterns", files={"image": ("x.png", b"", "image/png")})
        assert response.status_code == 400

    @pytest.mark.parametrize("colors", [1, 99])
    def test_validates_color_count(self, colors):
        response = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"colors": colors},
        )
        assert response.status_code == 422

    def test_filled_preview_differs_from_outlines(self):
        files = {"image": ("scene.png", scene(), "image/png")}
        base = {"colors": 4, "max_dimension": 240, "include_recipes": False}
        outline = client.post("/patterns", files=files, params=base).json()["svg"]
        filled = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={**base, "filled": True},
        ).json()["svg"]
        assert outline != filled


class TestShadeCheck:
    def test_accepts_a_correct_mix(self):
        response = client.post(
            "/shade-check",
            files={"image": ("mix.png", swatch_photo(), "image/png")},
            data={"target": "#4a7a3a", "reference": "0.0,0.0,1.0,0.12"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["matched"] is True
        assert body["confidence"] == "calibrated"
        assert body["delta_e"] < 3.0

    def test_rejects_a_wrong_mix_and_says_which_way(self):
        response = client.post(
            "/shade-check",
            files={"image": ("mix.png", swatch_photo("#3a6b8a"), "image/png")},
            data={"target": "#3a6b3a", "reference": "0.0,0.0,1.0,0.12"},
        )
        body = response.json()
        assert body["matched"] is False
        assert "yellow" in body["hint"]

    def test_without_a_reference_the_answer_is_hedged(self):
        response = client.post(
            "/shade-check",
            files={"image": ("mix.png", swatch_photo(reference=False), "image/png")},
            data={"target": "#4a7a3a"},
        )
        body = response.json()
        assert body["confidence"] == "estimated"
        assert "rough read" in body["message"]

    def test_rejects_a_bad_target_color(self):
        response = client.post(
            "/shade-check",
            files={"image": ("mix.png", swatch_photo(), "image/png")},
            data={"target": "lime green"},
        )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "box", ["0.1,0.1", "0.5,0.1,0.2,0.9", "2,0,3,1", "a,b,c,d"]
    )
    def test_rejects_malformed_regions(self, box):
        response = client.post(
            "/shade-check",
            files={"image": ("mix.png", swatch_photo(), "image/png")},
            data={"target": "#4a7a3a", "swatch": box},
        )
        assert response.status_code == 400


class TestGrid:
    def test_returns_a_grid_and_the_numbers_an_artist_needs(self):
        response = client.post(
            "/grid",
            files={"image": ("scene.png", scene(300, 200), "image/png")},
            params={"divisions": 6},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["columns"] == 6
        assert body["cells"] == body["columns"] * body["rows"]
        assert body["cell_mm_a4"] > 0
        assert body["svg"].startswith("<?xml")
        assert "data:image/jpeg;base64," in body["svg"]

    def test_reports_when_cells_cannot_be_square(self):
        """An oblong grid distorts the drawing, so the client must be able to warn."""
        wide = client.post(
            "/grid",
            files={"image": ("w.png", scene(1200, 300), "image/png")},
            params={"divisions": 6},
        ).json()
        assert wide["square_enough"] is False

    def test_labels_and_diagonals_are_switchable(self):
        files = {"image": ("scene.png", scene(300, 200), "image/png")}
        plain = client.post("/grid", files=files, params={"labels": False}).json()["svg"]
        assert "<text" not in plain

    @pytest.mark.parametrize("divisions", [1, 99])
    def test_validates_divisions(self, divisions):
        response = client.post(
            "/grid",
            files={"image": ("scene.png", scene(300, 200), "image/png")},
            params={"divisions": divisions},
        )
        assert response.status_code == 422

    def test_rejects_a_non_image(self):
        response = client.post(
            "/grid", files={"image": ("x.png", b"definitely not an image here", "image/png")}
        )
        assert response.status_code == 400


class TestCritique:
    def test_measures_a_painting_against_its_reference(self):
        response = client.post(
            "/critique",
            files={
                "reference": ("ref.png", scene(240, 240), "image/png"),
                "attempt": ("att.png", scene(240, 240), "image/png"),
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["message"]
        assert body["next_step"]
        keys = {m["key"] for m in body["measurements"]}
        assert keys == {
            "colour_accuracy",
            "value_structure",
            "lightness_bias",
            "chroma_bias",
        }
        assert all(m["explanation"] for m in body["measurements"])

    def test_notices_a_painting_that_is_too_dark(self):
        import numpy as np
        from PIL import Image

        arr = np.asarray(Image.open(io.BytesIO(scene(240, 240))), dtype=np.float64)
        dark = encode(Image.fromarray((arr * 0.55).astype("uint8")))

        body = client.post(
            "/critique",
            files={
                "reference": ("ref.png", scene(240, 240), "image/png"),
                "attempt": ("att.png", dark, "image/png"),
            },
        ).json()
        assert "dark" in body["message"].lower()

    def test_reports_no_score_or_verdict_on_quality(self):
        """The API must not leak an aesthetic judgement the core refuses to make."""
        body = client.post(
            "/critique",
            files={
                "reference": ("ref.png", scene(240, 240), "image/png"),
                "attempt": ("att.png", scene(240, 240), "image/png"),
            },
        ).json()
        text = (body["message"] + " " + (body["next_step"] or "")).lower()
        assert not any(w in text for w in ("score", "/10", "grade", "beautiful", "talented"))

    def test_requires_both_images(self):
        response = client.post(
            "/critique", files={"reference": ("ref.png", scene(240, 240), "image/png")}
        )
        assert response.status_code == 422


class TestArchitecture:
    def test_api_does_not_reimplement_core(self):
        """The API layer must stay a translation layer.

        Importing OpenCV or scikit-image here would mean image processing had
        leaked out of the part that is testable without a server running.
        """
        from pathlib import Path

        import artmentor.api.app as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        for banned in ("import cv2", "from skimage", "import skimage", "delta_E"):
            assert banned not in source, f"{banned} leaked into the API layer"


class TestGallery:
    def test_saving_a_piece_returns_links_to_its_images(self, gallery):
        piece = new_piece("Harbour at dusk", medium="acrylic")
        assert piece["title"] == "Harbour at dusk"
        assert piece["medium"] == "acrylic"
        assert piece["attempts"] == []
        assert piece["image_url"].startswith("/gallery/images/")
        assert piece["thumbnail_url"].endswith("/thumb")

    def test_the_saved_reference_can_be_fetched_back(self, gallery):
        piece = new_piece()
        full = client.get(piece["image_url"])
        thumb = client.get(piece["thumbnail_url"])
        assert full.status_code == 200
        assert full.headers["content-type"] == "image/jpeg"
        assert thumb.status_code == 200
        assert len(thumb.content) < len(full.content)

    def test_uploads_are_sanitised_before_they_are_stored(self, gallery):
        """Nothing a client sent reaches the disk — the stored copy is re-encoded.

        A phone photo of a painting carries the GPS coordinates of the painter's
        home. Storing the upload verbatim would undo the loader's EXIF stripping
        at the one point where it lasts forever.
        """
        img = Image.open(io.BytesIO(scene(240, 240))).convert("RGB")
        exif = img.getexif()
        exif[0x010E] = "SECRET-LOCATION"  # ImageDescription
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", exif=exif.tobytes())
        assert b"SECRET-LOCATION" in buffer.getvalue(), "the fixture must actually carry it"

        response = client.post(
            "/gallery/pieces",
            files={"image": ("photo.jpg", buffer.getvalue(), "image/jpeg")},
            data={"title": "From a phone"},
        )
        stored = client.get(response.json()["image_url"]).content
        assert b"SECRET-LOCATION" not in stored
        assert dict(Image.open(io.BytesIO(stored)).getexif()) == {}

    def test_the_gallery_lists_what_was_saved(self, gallery):
        new_piece("One")
        new_piece("Two")
        listing = client.get("/gallery/pieces").json()
        assert {p["title"] for p in listing} == {"One", "Two"}
        assert all(p["attempts"] == 0 for p in listing)

    def test_a_pattern_can_be_kept_with_the_piece(self, gallery):
        pattern = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"colors": 4, "max_dimension": 240},
        ).json()
        piece = new_piece("With a guide", pattern=json.dumps(pattern))
        reopened = client.get(f"/gallery/pieces/{piece['id']}").json()
        assert reopened["has_pattern"] is True
        assert reopened["pattern"]["svg"] == pattern["svg"]

    def test_rejects_a_pattern_that_is_not_json(self, gallery):
        response = client.post(
            "/gallery/pieces",
            files={"image": ("ref.png", scene(240, 240), "image/png")},
            data={"title": "Bad", "pattern": "{not json"},
        )
        assert response.status_code == 400

    def test_rejects_a_piece_with_no_title(self, gallery):
        response = client.post(
            "/gallery/pieces",
            files={"image": ("ref.png", scene(240, 240), "image/png")},
            data={"title": "   "},
        )
        assert response.status_code == 400

    def test_rejects_a_reference_that_is_not_an_image(self, gallery):
        response = client.post(
            "/gallery/pieces",
            files={"image": ("x.png", b"certainly not a picture", "image/png")},
            data={"title": "Nope"},
        )
        assert response.status_code == 400

    def test_renaming(self, gallery):
        piece = new_piece("Untitled")
        renamed = client.patch(
            f"/gallery/pieces/{piece['id']}", data={"title": "Seascape"}
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Seascape"

    def test_deleting_a_piece_removes_it_and_its_images(self, gallery):
        piece = new_piece()
        assert client.delete(f"/gallery/pieces/{piece['id']}").status_code == 204
        assert client.get(f"/gallery/pieces/{piece['id']}").status_code == 404
        assert client.get(piece["image_url"]).status_code == 404

    def test_unknown_things_are_404_not_400(self, gallery):
        assert client.get("/gallery/pieces/nope").status_code == 404
        assert client.delete("/gallery/pieces/nope").status_code == 404
        assert client.delete("/gallery/attempts/nope").status_code == 404
        assert client.get(f"/gallery/images/{'0' * 64}").status_code == 404

    def test_a_malformed_digest_never_reaches_the_store(self, gallery):
        """The route only admits 64 hex characters, so nothing else is looked up."""
        for bad in ("abc", "Z" * 64, "0" * 63, "0" * 65):
            assert client.get(f"/gallery/images/{bad}").status_code == 422

    def test_a_traversal_attempt_is_refused_by_routing(self, gallery):
        """Two defences, and this one fires first.

        An encoded `../` is decoded before matching, so it stops being a single
        path segment and matches no route at all. Even if it got through, the
        store addresses files by hash and never by a caller's string.
        """
        for bad in ("..%2F..%2Fetc%2Fpasswd", "../../etc/passwd", "%2e%2e%2fsecrets"):
            assert client.get(f"/gallery/images/{bad}").status_code in (404, 422)


class TestGalleryAttempts:
    def test_saving_an_attempt_critiques_it_against_the_reference(self, gallery):
        piece = new_piece()
        response = client.post(
            f"/gallery/pieces/{piece['id']}/attempts",
            files={"image": ("att.png", scene(240, 240), "image/png")},
            data={"note": "first sitting"},
        )
        assert response.status_code == 201, response.text
        attempt = response.json()
        assert attempt["note"] == "first sitting"
        assert attempt["message"]
        assert {m["key"] for m in attempt["measurements"]} == {
            "colour_accuracy",
            "value_structure",
            "lightness_bias",
            "chroma_bias",
        }

    def test_attempts_show_up_on_the_piece_and_in_the_listing(self, gallery):
        piece = new_piece()
        for _ in range(2):
            client.post(
                f"/gallery/pieces/{piece['id']}/attempts",
                files={"image": ("att.png", scene(240, 240), "image/png")},
            )
        assert len(client.get(f"/gallery/pieces/{piece['id']}").json()["attempts"]) == 2
        assert client.get("/gallery/pieces").json()[0]["attempts"] == 2

    def test_an_attempt_against_a_missing_piece_is_404(self, gallery):
        response = client.post(
            "/gallery/pieces/nope/attempts",
            files={"image": ("att.png", scene(240, 240), "image/png")},
        )
        assert response.status_code == 404

    def test_deleting_an_attempt_leaves_the_piece(self, gallery):
        piece = new_piece()
        attempt = client.post(
            f"/gallery/pieces/{piece['id']}/attempts",
            files={"image": ("att.png", scene(240, 240), "image/png")},
        ).json()
        assert client.delete(f"/gallery/attempts/{attempt['id']}").status_code == 204
        assert client.get(f"/gallery/pieces/{piece['id']}").json()["attempts"] == []

    def test_a_stored_critique_is_not_recomputed_on_read(self, gallery):
        """History must keep saying what it said when it was written."""
        piece = new_piece()
        written = client.post(
            f"/gallery/pieces/{piece['id']}/attempts",
            files={"image": ("att.png", scene(240, 240), "image/png")},
        ).json()
        read_back = client.get(f"/gallery/pieces/{piece['id']}").json()["attempts"][0]
        assert read_back["measurements"] == written["measurements"]
        assert read_back["message"] == written["message"]


class TestGalleryProgress:
    def darker(self, factor: float) -> bytes:
        arr = np.asarray(Image.open(io.BytesIO(scene(240, 240))), dtype=np.float64)
        return encode(Image.fromarray(np.clip(arr * factor, 0, 255).astype("uint8")))

    def test_one_attempt_has_nothing_to_compare_yet(self, gallery):
        piece = new_piece()
        client.post(
            f"/gallery/pieces/{piece['id']}/attempts",
            files={"image": ("att.png", scene(240, 240), "image/png")},
        )
        body = client.get(f"/gallery/pieces/{piece['id']}/progress").json()
        assert body["attempts"] == 1
        assert body["trends"] == []
        assert "another attempt" in body["summary"]

    def test_a_closer_second_attempt_reads_as_improvement(self, gallery):
        piece = new_piece()
        for image in (self.darker(0.5), scene(240, 240)):
            client.post(
                f"/gallery/pieces/{piece['id']}/attempts",
                files={"image": ("att.png", image, "image/png")},
            )
        body = client.get(f"/gallery/pieces/{piece['id']}/progress").json()
        trends = {t["key"]: t for t in body["trends"]}
        assert trends["colour_accuracy"]["improved"] is True
        assert trends["colour_accuracy"]["latest"] < trends["colour_accuracy"]["first"]
        assert trends["colour_accuracy"]["description"]

    def test_progress_claims_nothing_about_quality(self, gallery):
        piece = new_piece()
        for image in (self.darker(0.5), scene(240, 240)):
            client.post(
                f"/gallery/pieces/{piece['id']}/attempts",
                files={"image": ("att.png", image, "image/png")},
            )
        body = client.get(f"/gallery/pieces/{piece['id']}/progress").json()
        text = " ".join([body["summary"]] + [t["description"] for t in body["trends"]]).lower()
        assert not any(w in text for w in ("score", "grade", "talent", "beautiful", "/10"))

    def test_progress_on_a_piece_with_no_attempts(self, gallery):
        piece = new_piece()
        assert client.get(f"/gallery/pieces/{piece['id']}/progress").status_code == 400

    def test_progress_on_a_missing_piece_is_404(self, gallery):
        assert client.get("/gallery/pieces/nope/progress").status_code == 404


class TestMediumAndLevel:
    def test_the_pickers_can_be_populated(self):
        mediums = client.get("/mediums").json()
        levels = client.get("/levels").json()
        assert {m["id"] for m in mediums} >= {"acrylic", "watercolour", "pencil"}
        assert [level["id"] for level in levels] == ["beginner", "improving", "confident"]
        assert all(o["name"] for o in mediums + levels)

    def test_guidance_can_be_read_before_any_upload(self):
        body = client.get("/guidance", params={"medium": "watercolour", "level": "beginner"}).json()
        assert body["excluded_paint_ids"] == ["ti-white"]
        assert body["max_components"] == 2
        assert body["notes"]

    def test_an_unknown_medium_is_a_400_not_a_500(self):
        assert client.get("/guidance", params={"medium": "tempera"}).status_code == 400
        response = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"medium": "tempera", "max_dimension": 240},
        )
        assert response.status_code == 400

    def test_watercolour_never_asks_for_white(self):
        """The paper is the white; a tube of it is not in the painter's hand."""
        body = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"medium": "watercolour", "max_dimension": 240, "colors": 5},
        ).json()
        for colour in body["palette"]:
            assert all(step["paint_id"] != "ti-white" for step in colour["mix"])
            assert "water" in colour["instructions"]

    def test_acrylic_still_uses_white(self):
        body = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"medium": "acrylic", "max_dimension": 240, "colors": 6},
        ).json()
        used = {s["paint_id"] for c in body["palette"] for s in c["mix"]}
        assert "ti-white" in used

    def test_a_pencil_pattern_is_a_value_map_not_a_colour_one(self):
        """Graphite has no hue, so naming a pigment would be meaningless."""
        body = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"medium": "pencil", "max_dimension": 240, "colors": 4},
        ).json()
        for colour in body["palette"]:
            assert colour["mix"] is None
            assert "part" not in colour["instructions"].lower()
            assert "tone" in colour["instructions"] or "bare paper" in colour["instructions"]
            # Hue is stripped before quantizing, so every swatch is a grey.
            r, g, b = (int(colour["hex"][i:i + 2], 16) for i in (1, 3, 5))
            assert max(r, g, b) - min(r, g, b) <= 4, colour["hex"]

    def test_the_level_sets_the_colour_count_when_none_is_given(self):
        def colours(level):
            return client.post(
                "/patterns",
                files={"image": ("scene.png", scene(), "image/png")},
                params={"level": level, "max_dimension": 240, "include_recipes": False},
            ).json()["colors"]

        assert colours("beginner") < colours("confident")

    def test_an_explicit_colour_count_overrides_the_level(self):
        body = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"level": "beginner", "colors": 10, "max_dimension": 240,
                    "include_recipes": False},
        ).json()
        assert body["colors"] == 10

    def test_the_response_says_which_rules_it_followed(self):
        body = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"medium": "oil", "level": "beginner", "max_dimension": 240,
                    "include_recipes": False},
        ).json()
        assert body["medium"] == "oil"
        assert body["level"] == "beginner"
        assert any("oil" in n.lower() for n in body["notes"])

    def test_a_palette_emptied_by_the_medium_is_refused_clearly(self):
        """Watercolour plus a white-only palette leaves nothing to mix."""
        response = client.post(
            "/patterns",
            files={"image": ("scene.png", scene(), "image/png")},
            params={"medium": "watercolour", "paints": "ti-white", "max_dimension": 240},
        )
        assert response.status_code == 400
        assert "watercolour" in response.json()["detail"].lower()


class TestCaching:
    def test_the_page_is_never_cached(self):
        """A cached index.html makes every edit look like it silently reverted.

        This cost a long debugging session once: the file on disk was correct,
        the server was serving the new page, and the browser kept showing the
        old one. Asserted here so nobody "optimises" the header back.
        """
        response = client.get("/")
        assert response.status_code == 200
        assert "no-store" in response.headers["cache-control"]

    def test_gallery_images_are_cached_hard(self, gallery):
        """The opposite case, and for the opposite reason.

        An image is addressed by the hash of its own bytes, so its content can
        never change. It is safe to cache forever and wasteful not to.
        """
        piece = new_piece()
        headers = client.get(piece["image_url"]).headers
        assert "immutable" in headers["cache-control"]


class TestLoomisSheet:
    def test_the_head_sheet_is_served(self):
        """Checks only status and type, so the route can be written any way."""
        response = client.get("/loomis-sheet.svg")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/svg+xml")

    def test_it_is_a_real_svg_with_several_heads(self):
        body = client.get("/loomis-sheet.svg").text
        assert body.lstrip().startswith("<?xml")
        assert body.count("turn") >= 8

    def test_it_says_the_proportions_are_a_convention(self):
        """The honesty travels with the sheet, not just with the code."""
        assert "not measured from the book" in client.get("/loomis-sheet.svg").text


class TestAdvancement:
    def attempt(self, piece_id, scale):
        """One saved attempt at a given brightness, to move colour accuracy."""
        import numpy as np
        from PIL import Image

        arr = np.asarray(Image.open(io.BytesIO(scene(240, 240))), dtype=np.float64)
        shot = encode(Image.fromarray(np.clip(arr * scale, 0, 255).astype("uint8")))
        return client.post(
            f"/gallery/pieces/{piece_id}/attempts",
            files={"image": ("a.png", shot, "image/png")},
        )

    def test_one_attempt_asks_for_more_before_judging(self, gallery):
        """A single good photograph under good light must not promote anyone."""
        piece = new_piece()
        self.attempt(piece["id"], 1.0)
        body = client.get(f"/gallery/pieces/{piece['id']}/progress").json()
        assert body["advance"]["suggested"] is False
        assert "more attempt" in body["advance"]["message"]

    def test_a_run_of_close_attempts_suggests_moving_up(self, gallery):
        piece = new_piece()
        for _ in range(3):
            self.attempt(piece["id"], 1.0)
        body = client.get(
            f"/gallery/pieces/{piece['id']}/progress", params={"level": "beginner"}
        ).json()
        assert body["advance"]["suggested"] is True
        assert body["advance"]["next_level"] == "improving"

    def test_attempts_far_from_the_reference_do_not(self, gallery):
        piece = new_piece()
        for _ in range(3):
            self.attempt(piece["id"], 0.35)
        body = client.get(
            f"/gallery/pieces/{piece['id']}/progress", params={"level": "confident"}
        ).json()
        assert body["advance"]["suggested"] is False

    def test_the_top_of_the_ladder_has_nowhere_to_go(self, gallery):
        piece = new_piece()
        for _ in range(3):
            self.attempt(piece["id"], 1.0)
        body = client.get(
            f"/gallery/pieces/{piece['id']}/progress", params={"level": "confident"}
        ).json()
        assert body["advance"]["next_level"] is None
        assert body["advance"]["suggested"] is False

    def test_the_level_is_a_request_parameter_not_a_property_of_the_piece(self, gallery):
        """A skill level belongs to the painter, so two levels read differently."""
        piece = new_piece()
        for _ in range(3):
            self.attempt(piece["id"], 0.75)
        easy = client.get(
            f"/gallery/pieces/{piece['id']}/progress", params={"level": "beginner"}
        ).json()["advance"]
        hard = client.get(
            f"/gallery/pieces/{piece['id']}/progress", params={"level": "confident"}
        ).json()["advance"]
        assert easy["target_delta_e"] > hard["target_delta_e"]

    def test_an_unknown_level_is_a_400(self, gallery):
        piece = new_piece()
        self.attempt(piece["id"], 1.0)
        response = client.get(
            f"/gallery/pieces/{piece['id']}/progress", params={"level": "wizard"}
        )
        assert response.status_code == 400

    def test_it_never_claims_the_painting_is_good(self, gallery):
        piece = new_piece()
        for _ in range(3):
            self.attempt(piece["id"], 1.0)
        text = client.get(
            f"/gallery/pieces/{piece['id']}/progress", params={"level": "beginner"}
        ).json()["advance"]["message"].lower()
        assert not any(w in text for w in ("talented", "beautiful", "gifted", "score"))
