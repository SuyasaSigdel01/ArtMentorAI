"""Persistence. Everything that has to survive the process exiting.

Kept out of `core` on purpose: `core` is pure computation with no database,
no filesystem layout and no notion of a "user's gallery". That separation is
what lets the colour science be tested in milliseconds with no fixtures, and
what will let this layer be swapped for Postgres and object storage on a
server without touching a single formula.
"""

from .gallery import (
    Attempt,
    Gallery,
    GalleryError,
    NotFound,
    Piece,
    PieceSummary,
    Progress,
    Trend,
)

__all__ = [
    "Attempt",
    "Gallery",
    "GalleryError",
    "NotFound",
    "Piece",
    "PieceSummary",
    "Progress",
    "Trend",
]