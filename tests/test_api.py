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

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from artmentor.api.app import app

client = TestClient(app)


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