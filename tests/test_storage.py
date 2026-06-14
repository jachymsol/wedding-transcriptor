"""Tests for transcriptor.storage — SegmentQueue.

All tests use a temporary in-memory or temp-file database so they do not
touch the production ``data/transcript_queue.db``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from transcriptor.stabilization import TranscriptSegment
from transcriptor.storage import SegmentQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(
    segment_id: int = 1,
    text: str = "Hello world",
    language: str = "en",
    event_id: str = "test-event",
) -> TranscriptSegment:
    return TranscriptSegment(
        event_id=event_id,
        segment_id=segment_id,
        sequence_number=segment_id,
        timestamp="2027-06-14T12:00:00Z",
        source_language=language,
        text=text,
    )


@pytest.fixture
def tmp_queue(tmp_path: Path) -> SegmentQueue:
    """Fresh SegmentQueue backed by a temp-dir SQLite file."""
    q = SegmentQueue(db_path=tmp_path / "test_queue.db")
    yield q
    q.close()


# ---------------------------------------------------------------------------
# from_dict roundtrip
# ---------------------------------------------------------------------------

class TestFromDict:
    def test_roundtrip_all_fields(self):
        seg = _make_segment(segment_id=42, text="Test text", language="fr")
        reconstructed = TranscriptSegment.from_dict(seg.to_dict())
        assert reconstructed == seg

    def test_roundtrip_non_ascii(self):
        seg = _make_segment(text="Příliš žluťoučký kůň")
        reconstructed = TranscriptSegment.from_dict(seg.to_dict())
        assert reconstructed.text == "Příliš žluťoučký kůň"

    def test_final_defaults_to_true(self):
        data = _make_segment().to_dict()
        del data["final"]
        seg = TranscriptSegment.from_dict(data)
        assert seg.final is True

    def test_final_false_preserved(self):
        data = _make_segment().to_dict()
        data["final"] = False
        seg = TranscriptSegment.from_dict(data)
        assert seg.final is False


# ---------------------------------------------------------------------------
# SegmentQueue — basic operations
# ---------------------------------------------------------------------------

class TestSegmentQueueBasic:
    def test_starts_empty(self, tmp_queue: SegmentQueue):
        assert tmp_queue.count() == 0
        assert tmp_queue.peek() == []

    def test_enqueue_returns_row_id(self, tmp_queue: SegmentQueue):
        row_id = tmp_queue.enqueue(_make_segment(1))
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_count_increases_after_enqueue(self, tmp_queue: SegmentQueue):
        tmp_queue.enqueue(_make_segment(1))
        tmp_queue.enqueue(_make_segment(2))
        assert tmp_queue.count() == 2

    def test_peek_does_not_remove(self, tmp_queue: SegmentQueue):
        tmp_queue.enqueue(_make_segment(1))
        tmp_queue.peek()
        assert tmp_queue.count() == 1

    def test_delete_removes_entry(self, tmp_queue: SegmentQueue):
        row_id = tmp_queue.enqueue(_make_segment(1))
        tmp_queue.delete(row_id)
        assert tmp_queue.count() == 0

    def test_delete_nonexistent_is_noop(self, tmp_queue: SegmentQueue):
        """Deleting a non-existent row must not raise."""
        tmp_queue.delete(9999)

    def test_count_decreases_after_delete(self, tmp_queue: SegmentQueue):
        r1 = tmp_queue.enqueue(_make_segment(1))
        tmp_queue.enqueue(_make_segment(2))
        tmp_queue.delete(r1)
        assert tmp_queue.count() == 1


# ---------------------------------------------------------------------------
# FIFO ordering
# ---------------------------------------------------------------------------

class TestFifoOrdering:
    def test_peek_returns_insertion_order(self, tmp_queue: SegmentQueue):
        ids = [1, 2, 3]
        for i in ids:
            tmp_queue.enqueue(_make_segment(i, text=f"Segment {i}"))
        rows = tmp_queue.peek()
        assert [seg.segment_id for _, seg in rows] == ids

    def test_delete_middle_preserves_rest(self, tmp_queue: SegmentQueue):
        r1 = tmp_queue.enqueue(_make_segment(1))
        r2 = tmp_queue.enqueue(_make_segment(2))
        r3 = tmp_queue.enqueue(_make_segment(3))

        tmp_queue.delete(r2)

        row_ids, segments = zip(*tmp_queue.peek())
        assert set(row_ids) == {r1, r3}
        assert [s.segment_id for s in segments] == [1, 3]

    def test_drain_in_order(self, tmp_queue: SegmentQueue):
        for i in range(5):
            tmp_queue.enqueue(_make_segment(i + 1, text=f"s{i+1}"))

        drained = []
        for row_id, seg in tmp_queue.peek():
            drained.append(seg.segment_id)
            tmp_queue.delete(row_id)

        assert drained == [1, 2, 3, 4, 5]
        assert tmp_queue.count() == 0


# ---------------------------------------------------------------------------
# Roundtrip through the database
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_text_preserved(self, tmp_queue: SegmentQueue):
        seg = _make_segment(text="Bon anniversaire")
        tmp_queue.enqueue(seg)
        _, recovered = tmp_queue.peek()[0]
        assert recovered.text == seg.text

    def test_all_fields_preserved(self, tmp_queue: SegmentQueue):
        seg = _make_segment(segment_id=7, text="Ahoj světe", language="cs",
                             event_id="wedding-2027")
        tmp_queue.enqueue(seg)
        _, recovered = tmp_queue.peek()[0]
        assert recovered == seg

    def test_non_ascii_czech(self, tmp_queue: SegmentQueue):
        text = "Příliš žluťoučký kůň úpěl ďábelské ódy"
        seg = _make_segment(text=text, language="cs")
        tmp_queue.enqueue(seg)
        _, recovered = tmp_queue.peek()[0]
        assert recovered.text == text

    def test_non_ascii_polish(self, tmp_queue: SegmentQueue):
        text = "Zażółć gęślą jaźń"
        seg = _make_segment(text=text, language="pl")
        tmp_queue.enqueue(seg)
        _, recovered = tmp_queue.peek()[0]
        assert recovered.text == text

    def test_non_ascii_french(self, tmp_queue: SegmentQueue):
        text = "Voilà un événement très élégant"
        seg = _make_segment(text=text, language="fr")
        tmp_queue.enqueue(seg)
        _, recovered = tmp_queue.peek()[0]
        assert recovered.text == text


# ---------------------------------------------------------------------------
# Persistence across close/reopen
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_data_survives_reconnect(self, tmp_path: Path):
        db_path = tmp_path / "persist.db"
        seg = _make_segment(segment_id=99, text="Persisted segment")

        q1 = SegmentQueue(db_path=db_path)
        row_id = q1.enqueue(seg)
        q1.close()

        q2 = SegmentQueue(db_path=db_path)
        assert q2.count() == 1
        rows = q2.peek()
        assert len(rows) == 1
        assert rows[0][0] == row_id
        assert rows[0][1] == seg
        q2.close()

    def test_delete_persists_across_reconnect(self, tmp_path: Path):
        db_path = tmp_path / "persist_del.db"

        q1 = SegmentQueue(db_path=db_path)
        r1 = q1.enqueue(_make_segment(1))
        q1.enqueue(_make_segment(2))
        q1.delete(r1)
        q1.close()

        q2 = SegmentQueue(db_path=db_path)
        assert q2.count() == 1
        _, seg = q2.peek()[0]
        assert seg.segment_id == 2
        q2.close()

    def test_multiple_sessions_accumulate(self, tmp_path: Path):
        db_path = tmp_path / "multi.db"
        for i in range(3):
            q = SegmentQueue(db_path=db_path)
            q.enqueue(_make_segment(i + 1))
            q.close()

        q_final = SegmentQueue(db_path=db_path)
        assert q_final.count() == 3
        q_final.close()


# ---------------------------------------------------------------------------
# WAL mode
# ---------------------------------------------------------------------------

class TestWalMode:
    def test_journal_mode_is_wal(self, tmp_path: Path):
        db_path = tmp_path / "wal.db"
        q = SegmentQueue(db_path=db_path)
        q.close()

        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"


# ---------------------------------------------------------------------------
# Thread safety (smoke test)
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_enqueue(self, tmp_queue: SegmentQueue):
        """Multiple threads enqueuing simultaneously must not corrupt the DB."""
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    tmp_queue.enqueue(
                        _make_segment(segment_id=thread_id * 100 + i)
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert tmp_queue.count() == 50  # 5 threads × 10 enqueues

    def test_concurrent_enqueue_and_delete(self, tmp_queue: SegmentQueue):
        """Interleaved enqueue and delete must not deadlock or corrupt data."""
        for i in range(20):
            tmp_queue.enqueue(_make_segment(i))

        errors: list[Exception] = []

        def drain() -> None:
            try:
                for row_id, _ in tmp_queue.peek():
                    tmp_queue.delete(row_id)
            except Exception as exc:
                errors.append(exc)

        def fill() -> None:
            try:
                for i in range(20, 30):
                    tmp_queue.enqueue(_make_segment(i))
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=drain)
        t2 = threading.Thread(target=fill)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread errors: {errors}"
