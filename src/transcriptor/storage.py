"""Local storage & offline queue — FR-008, spec §8.

Persists finalized :class:`~transcriptor.stabilization.TranscriptSegment`
objects in a SQLite database so they survive network outages and can be
retransmitted when connectivity is restored.

Database location: ``data/transcript_queue.db`` (project root).

Schema
------
Table ``pending_segments``:

    id         INTEGER  PRIMARY KEY AUTOINCREMENT
    payload    JSON     NOT NULL
    created_at DATETIME NOT NULL DEFAULT (datetime('now'))

All public methods are protected by a :class:`threading.Lock` so the
queue can be written from the main pipeline thread and drained from a
separate reconnect/retry thread (Step 7) without data races.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

from transcriptor.stabilization import TranscriptSegment

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH: Path = _PROJECT_ROOT / "data" / "transcript_queue.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS pending_segments (
    id         INTEGER  PRIMARY KEY AUTOINCREMENT,
    payload    JSON     NOT NULL,
    created_at DATETIME NOT NULL DEFAULT (datetime('now'))
)
"""


class SegmentQueue:
    """Thread-safe SQLite-backed queue for offline transcript segments.

    Typical usage::

        queue = SegmentQueue()

        # Transmission failed — persist the segment:
        queue.enqueue(segment)

        # Connectivity restored — drain and retransmit:
        for row_id, segment in queue.peek():
            if server.send(segment):
                queue.delete(row_id)

        # UI status display:
        label = f"Offline Queue: {queue.count()} segments"
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        """
        Parameters
        ----------
        db_path:
            Path to the SQLite file.  Parent directories are created
            automatically.  Defaults to ``data/transcript_queue.db``.
        """
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, segment: TranscriptSegment) -> int:
        """Persist *segment* to the queue.

        Returns
        -------
        int
            The database row id assigned to this entry.
        """
        row_id = self.enqueue_raw(segment.to_json())
        log.info("Segment %d queued for retry (row %d)", segment.segment_id, row_id)
        return row_id

    def enqueue_raw(self, payload: str) -> int:
        """Persist an arbitrary JSON string to the queue.

        Used for control messages (e.g. section breaks) that are not
        :class:`~transcriptor.stabilization.TranscriptSegment` objects but
        must survive network outages and be retransmitted on reconnect.

        Returns
        -------
        int
            The database row id assigned to this entry.
        """
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO pending_segments (payload) VALUES (?)",
                (payload,),
            )
            self._conn.commit()
            row_id: int = cursor.lastrowid  # type: ignore[assignment]
        log.info("Raw payload queued for retry (row %d)", row_id)
        return row_id

    def peek_all(self) -> list[tuple[int, str]]:
        """Return all pending ``(row_id, raw_json)`` pairs in FIFO order.

        Does **not** remove entries — call :meth:`delete` after a
        successful transmission.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, payload FROM pending_segments ORDER BY id ASC"
            ).fetchall()
        return [(row["id"], row["payload"]) for row in rows]

    def peek(self) -> list[tuple[int, TranscriptSegment]]:
        """Return pending ``(row_id, segment)`` pairs for transcript segments only.

        Rows whose payload does not deserialise as a
        :class:`~transcriptor.stabilization.TranscriptSegment` (e.g. control
        messages) are silently skipped.

        .. deprecated::
            Prefer :meth:`peek_all` which returns every queued row regardless
            of type.
        """
        result: list[tuple[int, TranscriptSegment]] = []
        for row_id, payload in self.peek_all():
            try:
                data = json.loads(payload)
                if data.get("type") in (None, "transcript"):
                    result.append((row_id, TranscriptSegment.from_dict(data)))
            except Exception:
                pass
        return result

    def delete(self, row_id: int) -> None:
        """Remove the entry with *row_id* after successful transmission."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM pending_segments WHERE id = ?", (row_id,)
            )
            self._conn.commit()
        log.info("Segment row %d removed from queue", row_id)

    def count(self) -> int:
        """Number of segments currently waiting in the queue."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM pending_segments"
            ).fetchone()
        return row[0]

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()
        log.info("Segment queue closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            # WAL mode allows concurrent reads during a write transaction
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_CREATE_TABLE)
            self._conn.commit()
        log.info("Segment queue ready: %s", self._path)
