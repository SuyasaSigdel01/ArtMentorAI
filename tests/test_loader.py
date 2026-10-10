"""Tests for the hardened image loader.

This is a security boundary, so these tests assert on rejection behavior as much
as on happy paths. A regression here is not a cosmetic bug.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from artmentor.core.imaging.loader import (
    ImageTooLargeError,
    UnsupportedImageError,
    load_image,
)


def encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def solid(width: int = 64, height: int = 48, color=(200, 80, 40)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


class TestHappyPath:
    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
    def test_accepts_allowed_formats(self, fmt):
        loaded = load_image(encode(solid(), fmt))
        assert loaded.source_format == fmt
        assert loaded.rgb.shape == (48, 64, 3)
        assert loaded.rgb.dtype == np.uint8

    def test_reads_from_path(self, tmp_path):
        path = tmp_path / "x.png"
        path.write_bytes(encode(solid()))
        assert load_image(path).rgb.shape == (48, 64, 3)

    def test_downscales_to_max_dimension(self):
        loaded = load_image(encode(solid(800, 400)), max_dimension=200)
        assert max(loaded.rgb.shape[:2]) == 200
        assert loaded.original_size == (800, 400)  # original size is still reported

    def test_never_upscales(self):
        loaded = load_image(encode(solid(40, 30)), max_dimension=4000)
        assert loaded.rgb.shape == (30, 40, 3)

    def test_rgba_is_flattened_to_rgb(self):
        rgba = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        assert load_image(encode(rgba)).rgb.shape == (10, 10, 3)


class TestRejections:
    def test_rejects_non_image_bytes(self):
        with pytest.raises(UnsupportedImageError):
            load_image(b"this is definitely not an image, but it is long enough")

    def test_rejects_truncated_payload(self):
        with pytest.raises(UnsupportedImageError):
            load_image(b"\x89PNG")

    def test_rejects_disallowed_format_despite_valid_image(self):
        """A GIF is a real image and still must not get through."""
        buffer = io.BytesIO()
        solid().save(buffer, format="GIF")
        with pytest.raises(UnsupportedImageError):
            load_image(buffer.getvalue())

    def test_extension_cannot_override_content(self, tmp_path):
        """Naming a text file .png must not get it past validation."""
        path = tmp_path / "evil.png"
        path.write_bytes(b"GIF89a" + b"\x00" * 64)
        with pytest.raises(UnsupportedImageError):
            load_image(path)

    def test_enforces_pixel_limit(self):
        with pytest.raises(ImageTooLargeError):
            load_image(encode(solid(400, 400)), max_pixels=1000)

    def test_enforces_byte_limit(self):
        with pytest.raises(ImageTooLargeError):
            load_image(encode(solid(400, 400)), max_bytes=50)

    def test_pixel_limit_checked_before_full_decode(self):
        """The guard must trip on header dimensions, not after allocating the image."""
        payload = encode(solid(3000, 3000))
        with pytest.raises(ImageTooLargeError):
            load_image(payload, max_pixels=10_000)


class TestSanitization:
    def test_exif_orientation_is_applied_then_discarded(self):
        """A rotated photo must come back upright, with no metadata attached."""
        img = solid(40, 20)
        exif = img.getexif()
        exif[0x0112] = 6  # orientation: rotate 90 CW
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", exif=exif)

        loaded = load_image(buffer.getvalue())
        # Orientation honoured: the 40x20 landscape becomes 20x40 portrait.
        assert loaded.rgb.shape[:2] == (40, 20)
        # Output is a bare array, so no EXIF (and no GPS) can ride along.
        assert isinstance(loaded.rgb, np.ndarray)

    def test_output_is_a_fresh_contiguous_array(self):
        loaded = load_image(encode(solid()))
        assert loaded.rgb.flags["OWNDATA"]
