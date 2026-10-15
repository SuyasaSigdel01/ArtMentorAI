"""Tests for saved work.

A store is the one layer where "it worked when I tried it" is worthless —
the bugs that matter only appear on the second run, after a restart, or when
two rows share a file. So these tests lean on three things the happy path
never exercises: reopening the database, storing the same bytes twice, and
deleting one of two rows that point at the same image.

`test_deleting_one_piece_keeps_a_shared_image` is the one that would have
bitten in production, where a reference photo reused across pieces would
vanish from all of them the first time one was deleted.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from artmentor.core.critique import critique_attempt
from artmentor.core.interfaces import Critique, Measurement
from artmentor.store import Gallery, GalleryError


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def png(colour: tuple[int, int, int] = (120, 160, 90), size: int = 64) -> bytes:
    arr = np.full((size, size, 3), colour, dtype=np.uint8)
    arr[: size // 2] = tuple(min(255, c + 60) for c in colour)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg(colour: tuple[int, int, int] = (200, 90, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((48, 64, 3), colour, dtype=np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


def fake_critique(**values: float) -> Critique:
    """A critique with exactly the numbers a test wants to assert on."""
    defaults = {
        "colour_accuracy": 10.0,
        "value_structure": 0.8,
        "lightness_bias": -4.0,
        "chroma_bias": 2.0,
    }
    defaults.update(values)
    units = {
        "colour_accuracy": "dE",
        "value_structure": "correlation",
        "lightness_bias": "L*",
        "chroma_bias": "C*",
    }
    return Critique(
        measurements=tuple(
            Measurement(key=k, value=v, unit=units[k], explanation=f"{k} is {v}.")
            for k, v in defaults.items()
        ),
        message="Keep going.",
        suggested_next_step="Push your darks.",
    )


@pytest.fixture
def gallery(tmp_path):
    return Gallery(tmp_path / "studio")


# --------------------------------------------------------------------------
class TestPieces:
    def test_saves_and_reads_back_a_piece(self, gallery):
        piece = gallery.add_piece("Harbour at dusk", png(), medium="acrylic")
        again = gallery.get_piece(piece.id)
        assert again.title == "Harbour at dusk"
        assert again.medium == "acrylic"
        assert again.attempts == ()
        assert again.reference_digest == piece.reference_digest

    def test_survives_reopening_the_database(self, tmp_path):
        """The entire point of the module. Nothing may live only in memory."""
        first = Gallery(tmp_path / "studio")
        piece = first.add_piece("Pear study", png())
        second = Gallery(tmp_path / "studio")
        assert second.get_piece(piece.id).title == "Pear study"
        assert [p.id for p in second.list_pieces()] == [piece.id]

    def test_stores_a_pattern_alongside_the_reference(self, gallery):
        pattern = {"colors": 6, "palette": [{"number": 1, "hex": "#aabbcc"}]}
        piece = gallery.add_piece("Barn", png(), pattern=pattern)
        assert gallery.get_piece(piece.id).pattern == pattern
        assert gallery.list_pieces()[0].has_pattern is True

    def test_titles_are_tidied_not_trusted(self, gallery):
        piece = gallery.add_piece("  two   spaces\nand a newline ", png())
        assert piece.title == "two spaces and a newline"

    @pytest.mark.parametrize("title", ["", "   ", "\n"])
    def test_rejects_an_empty_title(self, gallery, title):
        with pytest.raises(GalleryError):
            gallery.add_piece(title, png())

    def test_rejects_an_overlong_title(self, gallery):
        with pytest.raises(GalleryError):
            gallery.add_piece("x" * 500, png())

    def test_rejects_a_file_that_is_not_an_image(self, gallery):
        with pytest.raises(GalleryError):
            gallery.add_piece("Nope", b"PK\x03\x04 this is a zip")

    def test_rejects_an_empty_image(self, gallery):
        with pytest.raises(GalleryError):
            gallery.add_piece("Nope", b"")

    def test_renaming_keeps_everything_else(self, gallery):
        piece = gallery.add_piece("Untitled", png())
        gallery.add_attempt(piece.id, jpeg(), fake_critique())
        renamed = gallery.rename_piece(piece.id, "Seascape")
        assert renamed.title == "Seascape"
        assert len(renamed.attempts) == 1

    def test_unknown_ids_raise_rather_than_return_none(self, gallery):
        """A silent None here becomes a confusing 200 with an empty body."""
        for call in (
            lambda: gallery.get_piece("nope"),
            lambda: gallery.delete_piece("nope"),
            lambda: gallery.rename_piece("nope", "x"),
            lambda: gallery.progress("nope"),
            lambda: gallery.add_attempt("nope", jpeg(), fake_critique()),
        ):
            with pytest.raises(GalleryError):
                call()


class TestListing:
    def test_most_recently_worked_on_comes_first(self, gallery):
        """Sorted by last attempt, not by creation — the gallery is a workbench."""
        old = gallery.add_piece("Painted long ago", png((10, 20, 30)))
        new = gallery.add_piece("Started today", png((40, 50, 60)))
        gallery.add_attempt(old.id, jpeg((1, 2, 3)), fake_critique())
        assert gallery.list_pieces()[0].id == old.id
        assert {p.id for p in gallery.list_pieces()} == {old.id, new.id}

    def test_counts_attempts_without_loading_them(self, gallery):
        piece = gallery.add_piece("Study", png())
        for i in range(3):
            gallery.add_attempt(piece.id, jpeg((i * 20, 100, 100)), fake_critique())
        assert gallery.list_pieces()[0].attempts == 3

    def test_paginates(self, gallery):
        ids = [gallery.add_piece(f"Piece {i}", png((i, i, i))).id for i in range(5)]
        page = gallery.list_pieces(limit=2, offset=0)
        assert len(page) == 2
        assert len(gallery.list_pieces(limit=2, offset=4)) == 1
        assert len(ids) == 5

    def test_an_empty_gallery_lists_nothing(self, gallery):
        assert gallery.list_pieces() == []


class TestAttempts:
    def test_an_attempt_keeps_its_measurements(self, gallery):
        piece = gallery.add_piece("Still life", png())
        gallery.add_attempt(piece.id, jpeg(), fake_critique(colour_accuracy=13.5), note="too dark")

        attempt = gallery.get_piece(piece.id).attempts[0]
        assert attempt.value("colour_accuracy") == 13.5
        assert attempt.note == "too dark"
        assert attempt.message == "Keep going."
        assert attempt.next_step == "Push your darks."

    def test_measurements_round_trip_through_json_intact(self, gallery):
        """Including the explanation — the number alone is not feedback."""
        piece = gallery.add_piece("Study", png())
        original = critique_attempt(
            np.asarray(Image.open(io.BytesIO(png(size=80)))),
            np.asarray(Image.open(io.BytesIO(png((90, 120, 70), size=80)))),
        )
        gallery.add_attempt(piece.id, jpeg(), original)
        stored = gallery.get_piece(piece.id).attempts[0]
        assert stored.measurements == original.measurements

    def test_attempts_come_back_oldest_first(self, gallery):
        """Chronological, because the UI draws a timeline from them."""
        piece = gallery.add_piece("Study", png())
        for i in range(4):
            gallery.add_attempt(piece.id, jpeg((i * 30, 80, 80)), fake_critique(chroma_bias=i))
        values = [a.value("chroma_bias") for a in gallery.get_piece(piece.id).attempts]
        assert values == [0, 1, 2, 3]

    def test_a_missing_measure_is_simply_absent(self, gallery):
        piece = gallery.add_piece("Study", png())
        gallery.add_attempt(piece.id, jpeg(), fake_critique())
        assert gallery.get_piece(piece.id).attempts[0].value("not_a_measure") is None

    def test_deleting_an_attempt_leaves_the_piece(self, gallery):
        piece = gallery.add_piece("Study", png())
        a = gallery.add_attempt(piece.id, jpeg((5, 5, 5)), fake_critique())
        gallery.add_attempt(piece.id, jpeg((9, 9, 9)), fake_critique())
        gallery.delete_attempt(a.id)
        assert len(gallery.get_piece(piece.id).attempts) == 1

    def test_deleting_a_piece_takes_its_attempts(self, gallery):
        piece = gallery.add_piece("Study", png())
        gallery.add_attempt(piece.id, jpeg(), fake_critique())
        gallery.delete_piece(piece.id)
        assert gallery.stats()["attempts"] == 0


class TestMedia:
    def test_the_full_image_comes_back_byte_for_byte(self, gallery):
        data = png((77, 88, 99))
        piece = gallery.add_piece("Study", data)
        stored, media_type = gallery.media_bytes(piece.reference_digest)
        assert stored == data
        assert media_type == "image/png"

    def test_a_thumbnail_is_generated_and_is_small(self, gallery):
        piece = gallery.add_piece("Study", png(size=1200))
        thumb = gallery.thumbnail(piece.reference_digest)
        with Image.open(io.BytesIO(thumb)) as img:
            assert max(img.size) <= 320
            assert img.format == "JPEG"
        assert len(thumb) < len(png(size=1200))

    def test_identical_images_are_stored_once(self, gallery):
        """Content addressing, and the reason a reference can be reused freely."""
        data = png((11, 22, 33))
        a = gallery.add_piece("One", data)
        b = gallery.add_piece("Two", data)
        assert a.reference_digest == b.reference_digest
        assert gallery.stats()["images"] == 1

    def test_deleting_one_piece_keeps_a_shared_image(self, gallery):
        """The refcount test. Two pieces, one photo, delete one — the other still works."""
        data = png((44, 55, 66))
        a = gallery.add_piece("One", data)
        b = gallery.add_piece("Two", data)
        gallery.delete_piece(a.id)
        assert gallery.media_bytes(b.reference_digest)[0] == data

    def test_deleting_the_last_user_removes_the_file(self, gallery):
        piece = gallery.add_piece("Only", png((1, 1, 1)))
        digest = piece.reference_digest
        gallery.delete_piece(piece.id)
        assert gallery.stats()["images"] == 0
        with pytest.raises(GalleryError):
            gallery.media_bytes(digest)

    def test_files_are_named_by_digest_not_by_anything_a_user_typed(self, gallery):
        """Path traversal is designed out, not filtered out."""
        piece = gallery.add_piece("../../etc/passwd", png())
        files = [p.name for p in gallery.media_dir.rglob("*") if p.is_file()]
        assert files == [f"{piece.reference_digest}.png"]

    def test_unknown_digests_raise(self, gallery):
        with pytest.raises(GalleryError):
            gallery.media_bytes("0" * 64)
        with pytest.raises(GalleryError):
            gallery.thumbnail("0" * 64)

    def test_jpeg_and_png_both_round_trip(self, gallery):
        p1 = gallery.add_piece("png", png())
        p2 = gallery.add_piece("jpg", jpeg())
        assert gallery.media_bytes(p1.reference_digest)[1] == "image/png"
        assert gallery.media_bytes(p2.reference_digest)[1] == "image/jpeg"


class TestProgress:
    def setup_piece(self, gallery, first: dict, latest: dict):
        piece = gallery.add_piece("Study", png())
        gallery.add_attempt(piece.id, jpeg((1, 1, 1)), fake_critique(**first))
        gallery.add_attempt(piece.id, jpeg((2, 2, 2)), fake_critique(**latest))
        return piece

    def test_a_single_attempt_has_no_trend_yet(self, gallery):
        piece = gallery.add_piece("Study", png())
        gallery.add_attempt(piece.id, jpeg(), fake_critique())
        progress = gallery.progress(piece.id)
        assert progress.attempts == 1
        assert progress.trends == ()
        assert "another attempt" in progress.summary

    def test_getting_closer_is_reported_as_improvement(self, gallery):
        piece = self.setup_piece(
            gallery,
            {"colour_accuracy": 20.0, "value_structure": 0.5},
            {"colour_accuracy": 9.0, "value_structure": 0.9},
        )
        trends = {t.key: t for t in gallery.progress(piece.id).trends}
        assert trends["colour_accuracy"].improved is True
        assert trends["value_structure"].improved is True

    def test_lower_is_better_for_distance_higher_for_correlation(self, gallery):
        """The two measures move in opposite directions; the sign must follow."""
        piece = self.setup_piece(
            gallery,
            {"colour_accuracy": 8.0, "value_structure": 0.9},
            {"colour_accuracy": 15.0, "value_structure": 0.6},
        )
        trends = {t.key: t for t in gallery.progress(piece.id).trends}
        assert trends["colour_accuracy"].improved is False
        assert trends["value_structure"].improved is False

    def test_a_signed_bias_improves_by_approaching_zero(self, gallery):
        """-14 to +3 is progress even though the number went up."""
        piece = self.setup_piece(
            gallery, {"lightness_bias": -14.0}, {"lightness_bias": 3.0}
        )
        trend = {t.key: t for t in gallery.progress(piece.id).trends}["lightness_bias"]
        assert trend.change > 0
        assert trend.improved is True

    def test_noise_is_not_called_progress(self, gallery):
        """A change too small to see must not be announced as improvement."""
        piece = self.setup_piece(
            gallery, {"value_structure": 0.800}, {"value_structure": 0.8002}
        )
        trend = {t.key: t for t in gallery.progress(piece.id).trends}["value_structure"]
        assert trend.improved is None

    def test_compares_the_latest_with_the_first_not_the_previous(self, gallery):
        """One bad session in the middle must not read as 'getting worse'."""
        piece = gallery.add_piece("Study", png())
        for value in (20.0, 30.0, 8.0):
            gallery.add_attempt(piece.id, jpeg((int(value), 7, 7)), fake_critique(colour_accuracy=value))
        trend = {t.key: t for t in gallery.progress(piece.id).trends}["colour_accuracy"]
        assert (trend.first, trend.latest) == (20.0, 8.0)
        assert trend.improved is True

    def test_an_unknown_measure_gets_no_verdict(self, gallery):
        """A measure added later must report its change and refuse to grade it."""
        piece = gallery.add_piece("Study", png())
        for value in (4.0, 9.0):
            critique = Critique(
                measurements=(
                    Measurement(key="edge_softness", value=value, unit="px", explanation="x."),
                ),
                message="ok",
            )
            gallery.add_attempt(piece.id, jpeg((int(value) * 10, 3, 3)), critique)
        trend = gallery.progress(piece.id).trends[0]
        assert trend.key == "edge_softness"
        assert trend.change == 5.0
        assert trend.improved is None

    def test_a_measure_absent_from_the_first_attempt_is_skipped(self, gallery):
        piece = gallery.add_piece("Study", png())
        gallery.add_attempt(
            piece.id,
            jpeg((1, 1, 1)),
            Critique(
                measurements=(Measurement("colour_accuracy", 10.0, "dE", "x."),), message="ok"
            ),
        )
        gallery.add_attempt(piece.id, jpeg((2, 2, 2)), fake_critique())
        keys = {t.key for t in gallery.progress(piece.id).trends}
        assert keys == {"colour_accuracy"}

    def test_the_summary_is_a_sentence_a_learner_can_read(self, gallery):
        piece = self.setup_piece(
            gallery,
            {"colour_accuracy": 25.0, "value_structure": 0.4, "lightness_bias": -20.0,
             "chroma_bias": -15.0},
            {"colour_accuracy": 6.0, "value_structure": 0.95, "lightness_bias": -1.0,
             "chroma_bias": 1.0},
        )
        summary = gallery.progress(piece.id).summary
        assert "improved" in summary
        assert summary.endswith(".")

    def test_progress_makes_no_claim_about_quality(self, gallery):
        """Same stance as critique.py: fidelity moved, not 'you got better at art'."""
        piece = self.setup_piece(
            gallery, {"colour_accuracy": 25.0}, {"colour_accuracy": 6.0}
        )
        progress = gallery.progress(piece.id)
        text = " ".join([progress.summary] + [t.description for t in progress.trends]).lower()
        assert not any(w in text for w in ("score", "grade", "talent", "beautiful", "/10"))


class TestDurability:
    def test_two_galleries_on_one_folder_see_each_other(self, tmp_path):
        """Two requests in a thread pool are exactly this situation."""
        a = Gallery(tmp_path / "studio")
        b = Gallery(tmp_path / "studio")
        piece = a.add_piece("Shared", png())
        b.add_attempt(piece.id, jpeg(), fake_critique())
        assert len(a.get_piece(piece.id).attempts) == 1

    def test_a_failed_insert_leaves_nothing_behind(self, gallery):
        """A rejected attempt must not leave its image orphaned in the store."""
        before = gallery.stats()
        with pytest.raises(GalleryError):
            gallery.add_attempt("no-such-piece", jpeg((123, 45, 67)), fake_critique())
        assert gallery.stats() == before

    def test_an_overlong_note_is_rejected_before_anything_is_written(self, gallery):
        piece = gallery.add_piece("Study", png())
        with pytest.raises(GalleryError):
            gallery.add_attempt(piece.id, jpeg(), fake_critique(), note="x" * 5000)
        assert gallery.get_piece(piece.id).attempts == ()

    def test_stats_add_up(self, gallery):
        piece = gallery.add_piece("Study", png())
        gallery.add_attempt(piece.id, jpeg(), fake_critique())
        stats = gallery.stats()
        assert stats["pieces"] == 1
        assert stats["attempts"] == 1
        assert stats["images"] == 2
        assert stats["bytes"] > 0

    def test_a_newer_schema_is_refused_rather_than_corrupted(self, tmp_path):
        """Better a clear error than a silent write with the wrong columns."""
        import sqlite3

        Gallery(tmp_path / "studio")
        db = sqlite3.connect(tmp_path / "studio" / "gallery.db")
        db.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        db.commit()
        db.close()
        with pytest.raises(GalleryError):
            Gallery(tmp_path / "studio")