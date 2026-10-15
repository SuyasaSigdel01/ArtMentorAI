"""A learner's saved work: references, attempts, and progress over time.

Two of the original seven features were blocked on this one module. The
gallery (feature 7) obviously needs somewhere to keep paintings. Less
obviously, "has my painting improved?" (feature 4) needs it too — a single
critique can only say how far today's attempt sits from the reference, and
"further than last week" is a different sentence that requires last week to
still exist.

Design notes worth keeping
--------------------------
**SQLite, one file.** The whole gallery is `gallery.db` plus a `media/`
folder beside it. It can be copied, backed up or deleted by dragging one
folder, which matters a great deal while this runs on a laptop. The schema is
ordinary SQL with no SQLite-only syntax, so the move to Postgres later is a
connection string and a migration, not a rewrite.

**Images on disk, thumbnails in the database.** Full-size photographs would
bloat the database file and slow every query that doesn't need them.
Thumbnails are small and are needed by exactly the query that must be fast —
listing the gallery — so they live inline and the list renders from one
statement with no disk reads at all.

**Content-addressed media.** A file is stored under the SHA-256 of its bytes,
never under a name the user supplied. That removes path traversal as a class
of bug rather than defending against it, and it deduplicates for free: the
same reference photo used in five pieces is stored once. Deletion is
refcounted against both tables, so removing one piece cannot orphan a file
another piece is still showing.

**Measurements stored as JSON, not columns.** `critique.py` will grow a fifth
and sixth measure. A column per measure would mean a migration each time and
would silently drop measures written by a newer version. The JSON blob keeps
old rows readable and new measures storable without any schema change — and
nothing here queries on an individual measure.

**Single user.** There is no owner column and no access control. This is a
local gallery for the person at the keyboard. Adding accounts later means
adding a `user_id` to two tables and a filter to every query, which is why
every query in this module goes through a method rather than being written
inline at the call site.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from artmentor.core.interfaces import Critique, Measurement

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

SCHEMA_VERSION = 1

#: Long edge of a stored thumbnail, in pixels. Big enough for a retina gallery
#: tile, small enough that a hundred of them is under a megabyte.
THUMB_SIZE = 320

#: Guards against a caller handing us something absurd. The API layer already
#: enforces its own limit before decoding; this is the floor under that.
MAX_MEDIA_BYTES = 30 * 1024 * 1024

_TITLE_MAX = 120
_NOTE_MAX = 500

#: Which direction counts as improvement for each measurement key. Anything
#: not listed here reports its change without claiming it is better or worse —
#: an honest "unknown" beats guessing the sign of a measure added later.
_BETTER = {
    "colour_accuracy": "lower",   # distance from the reference
    "value_structure": "higher",  # correlation with the reference
    "lightness_bias": "zero",     # signed; nearer zero is nearer the target
    "chroma_bias": "zero",
}


class GalleryError(RuntimeError):
    """Something the caller asked for cannot be done.

    Distinct from sqlite3 errors on purpose: the API layer maps this to a 4xx,
    while a raw sqlite3 error is a bug and deserves a 500.
    """


class NotFound(GalleryError):
    """The id or digest does not exist.

    Its own class so the HTTP layer can answer 404 rather than 400 without
    pattern-matching on the text of an error message — which works right up
    until someone rewords one.
    """


# --------------------------------------------------------------------------
# Records returned to callers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Attempt:
    """One photograph of the work in progress, with its critique."""

    id: str
    piece_id: str
    created_at: str
    image_digest: str
    measurements: tuple[Measurement, ...]
    message: str
    next_step: str | None = None
    note: str | None = None

    def value(self, key: str) -> float | None:
        for m in self.measurements:
            if m.key == key:
                return m.value
        return None


@dataclass(frozen=True)
class PieceSummary:
    """Enough to draw a gallery tile without loading the attempts."""

    id: str
    title: str
    medium: str | None
    created_at: str
    updated_at: str
    reference_digest: str
    attempts: int
    has_pattern: bool


@dataclass(frozen=True)
class Piece:
    """A painting being worked on: one reference, many attempts."""

    id: str
    title: str
    medium: str | None
    created_at: str
    reference_digest: str
    pattern: dict[str, Any] | None = None
    attempts: tuple[Attempt, ...] = ()


@dataclass(frozen=True)
class Trend:
    """How one measurement moved between the first attempt and the latest."""

    key: str
    unit: str
    first: float
    latest: float
    change: float
    improved: bool | None  # None when the direction of "better" is unknown

    @property
    def description(self) -> str:
        if self.improved is None:
            return f"{self.key} moved from {self.first:g} to {self.latest:g}."
        word = "closer to the reference" if self.improved else "further from it"
        return f"{self.key} went from {self.first:g} to {self.latest:g} — {word}."


@dataclass(frozen=True)
class Progress:
    """The answer to 'am I getting better at this one?'"""

    piece_id: str
    attempts: int
    first_at: str
    latest_at: str
    trends: tuple[Trend, ...] = field(default=())

    @property
    def summary(self) -> str:
        if self.attempts < 2:
            return "Save another attempt to see whether it is moving closer."
        better = [t for t in self.trends if t.improved is True]
        worse = [t for t in self.trends if t.improved is False]
        if better and not worse:
            return f"All {len(better)} measures improved since your first attempt."
        if better:
            return f"{len(better)} measures improved, {len(worse)} drifted."
        if worse:
            return "This attempt sits further from the reference than your first did."
        return "Nothing has moved much since your first attempt."


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS media (
    digest TEXT PRIMARY KEY,
    format TEXT NOT NULL,
    width  INTEGER NOT NULL,
    height INTEGER NOT NULL,
    bytes  INTEGER NOT NULL,
    thumb  BLOB NOT NULL
);

-- `seq` exists so that insertion order is never in doubt. Sorting by
-- `created_at` alone broke the moment two rows landed in the same clock tick
-- and the tie fell through to a random uuid: a learner's attempts came back
-- shuffled, which would have made the progress timeline nonsense. AUTOINCREMENT
-- rather than a bare rowid, because a plain rowid is reused after a delete and
-- could place a new attempt before an older one.
CREATE TABLE IF NOT EXISTS pieces (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    id               TEXT NOT NULL UNIQUE,
    title            TEXT NOT NULL,
    medium           TEXT,
    created_at       TEXT NOT NULL,
    reference_digest TEXT NOT NULL REFERENCES media(digest),
    pattern_json     TEXT
);

CREATE TABLE IF NOT EXISTS attempts (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    id           TEXT NOT NULL UNIQUE,
    piece_id     TEXT NOT NULL REFERENCES pieces(id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    image_digest TEXT NOT NULL REFERENCES media(digest),
    measurements TEXT NOT NULL,
    message      TEXT NOT NULL,
    next_step    TEXT,
    note         TEXT
);

CREATE INDEX IF NOT EXISTS attempts_by_piece ON attempts(piece_id, seq);
CREATE INDEX IF NOT EXISTS pieces_by_date ON pieces(created_at DESC);
"""


def _now() -> str:
    """UTC, ISO-8601, microseconds.

    Full precision on purpose: truncating to the second made two saves in the
    same second indistinguishable, and this string is what orders a learner's
    timeline. Text sort equals chronological sort for this format, so the
    database can order on it directly.
    """
    return datetime.now(UTC).isoformat()


def _digest(data: bytes) -> str:
    from hashlib import sha256

    return sha256(data).hexdigest()


def _clean(text: str | None, limit: int, label: str) -> str | None:
    if text is None:
        return None
    stripped = " ".join(text.split())
    if len(stripped) > limit:
        raise GalleryError(f"{label} is too long (max {limit} characters)")
    return stripped or None


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------
class Gallery:
    """A saved-work store backed by one SQLite file and a media folder."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.media_dir = self.root / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "gallery.db"
        with self._connect() as db:
            db.executescript(_SCHEMA)
            db.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            stored = db.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
        if int(stored) > SCHEMA_VERSION:
            raise GalleryError(
                f"this gallery was written by a newer version (schema {stored}); "
                f"this build understands {SCHEMA_VERSION}"
            )

    # -- connection ---------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A short-lived connection per operation.

        Not one long-lived connection: FastAPI runs synchronous routes in a
        thread pool, and a sqlite3 connection belongs to the thread that made
        it. Opening per call costs microseconds against a local file and makes
        the whole class thread-safe without a single lock.
        """
        db = sqlite3.connect(self.db_path, timeout=10.0)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA journal_mode = WAL")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # -- media --------------------------------------------------------------
    def _media_path(self, digest: str, suffix: str) -> Path:
        # Two-character shard: a flat folder of 100k files is slow to list on
        # every filesystem that matters.
        return self.media_dir / digest[:2] / f"{digest}{suffix}"

    def _store_media(self, db: sqlite3.Connection, data: bytes) -> str:
        if not data:
            raise GalleryError("the image is empty")
        if len(data) > MAX_MEDIA_BYTES:
            raise GalleryError(f"the image is larger than {MAX_MEDIA_BYTES // (1024 * 1024)}MB")

        digest = _digest(data)
        row = db.execute("SELECT format FROM media WHERE digest = ?", (digest,)).fetchone()
        if row is not None:
            return digest  # identical bytes already stored

        try:
            with Image.open(BytesIO(data)) as img:
                fmt = (img.format or "").upper()
                width, height = img.size
                thumb = self._thumbnail(img)
        except GalleryError:
            raise
        except Exception as exc:  # Pillow raises a zoo of exception types
            raise GalleryError("that file is not a readable image") from exc

        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt)
        if suffix is None:
            raise GalleryError(f"unsupported image format: {fmt or 'unknown'}")

        path = self._media_path(digest, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        db.execute(
            "INSERT INTO media (digest, format, width, height, bytes, thumb) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (digest, fmt, width, height, len(data), thumb),
        )
        return digest

    @staticmethod
    def _thumbnail(img: Image.Image) -> bytes:
        small = img.convert("RGB").copy()
        small.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        buffer = BytesIO()
        small.save(buffer, format="JPEG", quality=80, optimize=True)
        return buffer.getvalue()

    def media_bytes(self, digest: str) -> tuple[bytes, str]:
        """Full-size image and its media type. Raises if unknown."""
        with self._connect() as db:
            row = db.execute("SELECT format FROM media WHERE digest = ?", (digest,)).fetchone()
        if row is None:
            raise NotFound("no such image")
        fmt = row["format"]
        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]
        path = self._media_path(digest, suffix)
        if not path.is_file():
            raise NotFound("that image is recorded but its file is missing")
        return path.read_bytes(), f"image/{'jpeg' if fmt == 'JPEG' else fmt.lower()}"

    def thumbnail(self, digest: str) -> bytes:
        with self._connect() as db:
            row = db.execute("SELECT thumb FROM media WHERE digest = ?", (digest,)).fetchone()
        if row is None:
            raise NotFound("no such image")
        return bytes(row["thumb"])

    def _collect_orphans(self, db: sqlite3.Connection) -> int:
        """Delete media no piece or attempt refers to. Returns how many went."""
        rows = db.execute(
            "SELECT digest, format FROM media WHERE digest NOT IN "
            "(SELECT reference_digest FROM pieces UNION "
            " SELECT image_digest FROM attempts)"
        ).fetchall()
        for row in rows:
            suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[row["format"]]
            self._media_path(row["digest"], suffix).unlink(missing_ok=True)
            db.execute("DELETE FROM media WHERE digest = ?", (row["digest"],))
        return len(rows)

    # -- pieces -------------------------------------------------------------
    def add_piece(
        self,
        title: str,
        reference: bytes,
        *,
        medium: str | None = None,
        pattern: dict[str, Any] | None = None,
    ) -> Piece:
        clean_title = _clean(title, _TITLE_MAX, "title")
        if not clean_title:
            raise GalleryError("a piece needs a title")
        piece_id = uuid.uuid4().hex
        created = _now()
        with self._connect() as db:
            digest = self._store_media(db, reference)
            db.execute(
                "INSERT INTO pieces (id, title, medium, created_at, reference_digest, "
                "pattern_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    piece_id,
                    clean_title,
                    _clean(medium, 40, "medium"),
                    created,
                    digest,
                    json.dumps(pattern) if pattern is not None else None,
                ),
            )
        return Piece(
            id=piece_id,
            title=clean_title,
            medium=_clean(medium, 40, "medium"),
            created_at=created,
            reference_digest=digest,
            pattern=pattern,
        )

    def list_pieces(self, *, limit: int = 100, offset: int = 0) -> list[PieceSummary]:
        """Newest activity first — a piece worked on today belongs at the top."""
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT p.id, p.title, p.medium, p.created_at, p.reference_digest,
                       p.pattern_json IS NOT NULL AS has_pattern,
                       COUNT(a.id) AS attempts,
                       COALESCE(MAX(a.created_at), p.created_at) AS updated_at
                FROM pieces p
                LEFT JOIN attempts a ON a.piece_id = p.id
                GROUP BY p.id
                ORDER BY updated_at DESC, p.seq DESC
                LIMIT ? OFFSET ?
                """,
                (max(1, limit), max(0, offset)),
            ).fetchall()
        return [
            PieceSummary(
                id=r["id"],
                title=r["title"],
                medium=r["medium"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                reference_digest=r["reference_digest"],
                attempts=r["attempts"],
                has_pattern=bool(r["has_pattern"]),
            )
            for r in rows
        ]

    def get_piece(self, piece_id: str) -> Piece:
        with self._connect() as db:
            row = db.execute("SELECT * FROM pieces WHERE id = ?", (piece_id,)).fetchone()
            if row is None:
                raise NotFound("no such piece")
            attempts = self._attempts(db, piece_id)
        return Piece(
            id=row["id"],
            title=row["title"],
            medium=row["medium"],
            created_at=row["created_at"],
            reference_digest=row["reference_digest"],
            pattern=json.loads(row["pattern_json"]) if row["pattern_json"] else None,
            attempts=attempts,
        )

    def rename_piece(self, piece_id: str, title: str) -> Piece:
        clean = _clean(title, _TITLE_MAX, "title")
        if not clean:
            raise GalleryError("a piece needs a title")
        with self._connect() as db:
            changed = db.execute(
                "UPDATE pieces SET title = ? WHERE id = ?", (clean, piece_id)
            ).rowcount
        if not changed:
            raise NotFound("no such piece")
        return self.get_piece(piece_id)

    def delete_piece(self, piece_id: str) -> None:
        with self._connect() as db:
            changed = db.execute("DELETE FROM pieces WHERE id = ?", (piece_id,)).rowcount
            if not changed:
                raise NotFound("no such piece")
            self._collect_orphans(db)

    # -- attempts -----------------------------------------------------------
    def add_attempt(
        self,
        piece_id: str,
        image: bytes,
        critique: Critique,
        *,
        note: str | None = None,
    ) -> Attempt:
        attempt_id = uuid.uuid4().hex
        created = _now()
        clean_note = _clean(note, _NOTE_MAX, "note")
        payload = json.dumps(
            [
                {"key": m.key, "value": m.value, "unit": m.unit, "explanation": m.explanation}
                for m in critique.measurements
            ]
        )
        with self._connect() as db:
            if db.execute("SELECT 1 FROM pieces WHERE id = ?", (piece_id,)).fetchone() is None:
                raise NotFound("no such piece")
            digest = self._store_media(db, image)
            db.execute(
                "INSERT INTO attempts (id, piece_id, created_at, image_digest, "
                "measurements, message, next_step, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attempt_id,
                    piece_id,
                    created,
                    digest,
                    payload,
                    critique.message,
                    critique.suggested_next_step,
                    clean_note,
                ),
            )
        return Attempt(
            id=attempt_id,
            piece_id=piece_id,
            created_at=created,
            image_digest=digest,
            measurements=critique.measurements,
            message=critique.message,
            next_step=critique.suggested_next_step,
            note=clean_note,
        )

    def delete_attempt(self, attempt_id: str) -> None:
        with self._connect() as db:
            changed = db.execute("DELETE FROM attempts WHERE id = ?", (attempt_id,)).rowcount
            if not changed:
                raise NotFound("no such attempt")
            self._collect_orphans(db)

    @staticmethod
    def _attempts(db: sqlite3.Connection, piece_id: str) -> tuple[Attempt, ...]:
        rows = db.execute(
            "SELECT * FROM attempts WHERE piece_id = ? ORDER BY seq",
            (piece_id,),
        ).fetchall()
        return tuple(
            Attempt(
                id=r["id"],
                piece_id=r["piece_id"],
                created_at=r["created_at"],
                image_digest=r["image_digest"],
                measurements=tuple(
                    Measurement(**m) for m in json.loads(r["measurements"])
                ),
                message=r["message"],
                next_step=r["next_step"],
                note=r["note"],
            )
            for r in rows
        )

    # -- progress -----------------------------------------------------------
    def progress(self, piece_id: str) -> Progress:
        """Compare the latest attempt with the first one.

        First versus latest, not latest versus previous. A learner who had one
        bad session should not be told they are getting worse overall, and the
        question being answered is "has the practice paid off", which is about
        the whole arc.
        """
        attempts = self.get_piece(piece_id).attempts
        if not attempts:
            raise GalleryError("this piece has no attempts yet")

        first, latest = attempts[0], attempts[-1]
        trends: list[Trend] = []
        if len(attempts) >= 2:
            first_by_key = {m.key: m for m in first.measurements}
            for m in latest.measurements:
                before = first_by_key.get(m.key)
                if before is None:
                    continue  # a measure added after the first attempt was saved
                trends.append(
                    Trend(
                        key=m.key,
                        unit=m.unit,
                        first=before.value,
                        latest=m.value,
                        change=round(m.value - before.value, 3),
                        improved=_improved(m.key, before.value, m.value),
                    )
                )
        return Progress(
            piece_id=piece_id,
            attempts=len(attempts),
            first_at=first.created_at,
            latest_at=latest.created_at,
            trends=tuple(trends),
        )

    # -- housekeeping -------------------------------------------------------
    def stats(self) -> dict[str, int]:
        with self._connect() as db:
            return {
                "pieces": db.execute("SELECT COUNT(*) FROM pieces").fetchone()[0],
                "attempts": db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
                "images": db.execute("SELECT COUNT(*) FROM media").fetchone()[0],
                "bytes": db.execute("SELECT COALESCE(SUM(bytes), 0) FROM media").fetchone()[0],
            }


def _improved(key: str, before: float, after: float) -> bool | None:
    """Did this measure move the right way? `None` when we cannot say.

    The tolerance matters: a change of 0.001 in a correlation is noise from a
    differently-lit photograph, not progress, and telling a learner they
    improved because of it would make the whole feature untrustworthy.
    """
    direction = _BETTER.get(key)
    if direction is None:
        return None
    if direction == "lower":
        delta = before - after
    elif direction == "higher":
        delta = after - before
    else:  # "zero"
        delta = abs(before) - abs(after)
    if abs(delta) < 1e-3:
        return None
    return delta > 0