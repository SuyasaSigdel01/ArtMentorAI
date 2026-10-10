"""Hardened image loading.

Every image entering this system is untrusted input being fed to C-based parsers.
This module is the only sanctioned entry point for user-supplied images, and it
enforces four things that are easy to skip and expensive to retrofit:

1. Format is validated by magic bytes, never by filename extension or a
   client-supplied content type. Both are attacker-controlled.
2. Decompressed pixel count is capped before decode. A "decompression bomb" is a
   few kilobytes of PNG that expands to tens of gigabytes in RAM.
3. EXIF is stripped. Phone photos of a painting carry the GPS coordinates of the
   user's home. Orientation is applied first, then all metadata is discarded.
4. Every image is re-encoded through a canonical RGB path, which drops trailing
   payloads, embedded thumbnails, and assorted parser weirdness.

Keep this module boring and keep Pillow pinned and patched.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

__all__ = ["ImageTooLargeError", "LoadedImage", "UnsupportedImageError", "load_image"]


# Deliberately short. Every additional format is another parser's attack surface.
_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Magic byte signatures, checked before handing bytes to the decoder.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"RIFF", "WEBP"),  # refined below: bytes 8..12 must be "WEBP"
)

# Hard ceiling on decoded pixels (~40 megapixels). A legitimate phone photo is
# well under this; anything above is either a mistake or an attack.
MAX_PIXELS = 40_000_000

# Hard ceiling on encoded bytes accepted from a client (25 MB).
MAX_BYTES = 25 * 1024 * 1024


class UnsupportedImageError(ValueError):
    """The bytes are not an image in an allowed format."""


class ImageTooLargeError(ValueError):
    """The image exceeds a configured size limit."""


@dataclass(frozen=True)
class LoadedImage:
    """A validated, sanitized image ready for processing."""

    rgb: np.ndarray      # (H, W, 3) uint8
    source_format: str   # as detected, e.g. "JPEG"
    original_size: tuple[int, int]  # (width, height) before any downscale

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def _sniff_format(data: bytes) -> str:
    """Identify format from magic bytes, or raise."""
    for sig, fmt in _SIGNATURES:
        if data.startswith(sig):
            if fmt == "WEBP":
                if len(data) >= 12 and data[8:12] == b"WEBP":
                    return "WEBP"
                continue
            return fmt
    raise UnsupportedImageError(
        "File does not match an allowed image signature (JPEG, PNG, or WEBP)."
    )


def load_image(
    source: str | Path | bytes,
    *,
    max_pixels: int = MAX_PIXELS,
    max_bytes: int = MAX_BYTES,
    max_dimension: int | None = None,
) -> LoadedImage:
    """Validate, sanitize, and decode an image.

    Args:
        source: a filesystem path, or raw bytes from an upload.
        max_pixels: reject images whose decoded pixel count exceeds this.
        max_bytes: reject encoded payloads larger than this.
        max_dimension: if set, downscale so the longest side is at most this.

    Raises:
        UnsupportedImageError: bad signature, disallowed format, or corrupt data.
        ImageTooLargeError: exceeds a byte or pixel limit.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        size = path.stat().st_size
        if size > max_bytes:
            raise ImageTooLargeError(f"File is {size} bytes, limit is {max_bytes}.")
        data = path.read_bytes()
    else:
        data = bytes(source)
        if len(data) > max_bytes:
            raise ImageTooLargeError(f"Payload is {len(data)} bytes, limit is {max_bytes}.")

    if len(data) < 12:
        raise UnsupportedImageError("Payload too short to be an image.")

    sniffed = _sniff_format(data)

    # Pass 1: parse the header only, so dimensions can be checked before any
    # full decode allocates memory.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            declared = (probe.format or "").upper()
            width, height = probe.size
    except Exception as exc:  # Pillow raises a wide variety here
        raise UnsupportedImageError(f"Could not parse image: {exc}") from exc

    if declared not in _ALLOWED_FORMATS:
        raise UnsupportedImageError(f"Format {declared!r} is not allowed.")
    if declared != sniffed:
        # Signature and parser disagree: treat as hostile rather than guessing.
        raise UnsupportedImageError(
            f"Signature says {sniffed}, decoder says {declared}; refusing."
        )
    if width <= 0 or height <= 0:
        raise UnsupportedImageError("Image reports non-positive dimensions.")
    if width * height > max_pixels:
        raise ImageTooLargeError(
            f"Image is {width}x{height} ({width * height} px), limit is {max_pixels} px."
        )

    # Pass 2: full decode, with Pillow's own bomb guard also armed.
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)   # honour rotation...
            img = img.convert("RGB")             # ...then drop EXIF and alpha entirely
            if max_dimension is not None:
                img = _fit_within(img, max_dimension)
            # np.array on a converted copy is the canonical re-encode: no original
            # buffer, no metadata, no trailing bytes survive this.
            rgb = np.array(img, dtype=np.uint8, copy=True)
    except (UnsupportedImageError, ImageTooLargeError):
        raise
    except Exception as exc:
        raise UnsupportedImageError(f"Could not decode image: {exc}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise UnsupportedImageError("Decoded image is not 3-channel RGB.")

    return LoadedImage(rgb=rgb, source_format=declared, original_size=(width, height))


def _fit_within(img: Image.Image, max_dimension: int) -> Image.Image:
    """Downscale so the longest side is at most `max_dimension`. Never upscales."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dimension:
        return img
    scale = max_dimension / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.LANCZOS)
