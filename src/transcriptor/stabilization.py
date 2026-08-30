"""Transcript stabilization & segment generation — FR-005, FR-006.

Tracks transcript revisions from the Whisper transcriber and emits
finalized :class:`TranscriptSegment` JSON events when one of two
stabilization rules is satisfied:

1. **Silence rule** (FR-005 §"speech pause exceeds 700 ms"):
   The caller passes ``is_final=True`` when VAD has closed a speech
   segment (≥ ``speech_end_ms`` of silence detected).  The current
   pending text is emitted immediately.

2. **Stability rule** (FR-005 §"transcript unchanged for 2 seconds"):
   If the pending text has not changed for ``stable_ms`` milliseconds
   the segment is emitted automatically on the next :meth:`Stabilizer.update`
   call.  This handles streaming scenarios where Whisper is called on
   growing audio while speech is still ongoing.

``TranscriptSegment`` serialises to the exact JSON schema defined in
spec §4, with ``final: true`` always set in the MVP.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from transcriptor.config import AppConfig
from transcriptor.transcription import TranscriptResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TranscriptSegment — spec §4 JSON event
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    """A finalized transcript event matching spec §4.

    Attributes:
        event_id:         Identifies the live event (from ``config.yaml``).
        segment_id:       Unique segment identifier, monotonically increasing.
        sequence_number:  Monotonic ordering field (same as segment_id in MVP).
        timestamp:        UTC emission time, ISO 8601 with Z suffix.
        source_language:  Operator-selected language code (e.g. ``"cs"``).
        text:             Final, stripped transcript text.
        final:            Always ``True`` in MVP.
    """

    event_id: str
    segment_id: int
    sequence_number: int
    timestamp: str
    source_language: str
    text: str
    final: bool = True

    def to_dict(self) -> dict:
        """Return a plain dict matching the spec §4 JSON schema."""
        return {
            "event_id": self.event_id,
            "segment_id": self.segment_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "source_language": self.source_language,
            "text": self.text,
            "final": self.final,
        }

    def to_json(self) -> str:
        """Serialize to a compact JSON string (non-ASCII characters preserved)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptSegment":
        """Reconstruct a TranscriptSegment from a plain dict (e.g. from JSON storage)."""
        return cls(
            event_id=data["event_id"],
            segment_id=data["segment_id"],
            sequence_number=data["sequence_number"],
            timestamp=data["timestamp"],
            source_language=data["source_language"],
            text=data["text"],
            final=data.get("final", True),
        )


# ---------------------------------------------------------------------------
# Stabilizer
# ---------------------------------------------------------------------------

class Stabilizer:
    """Converts :class:`~transcriptor.transcription.TranscriptResult` objects
    into finalized :class:`TranscriptSegment` events.

    Keeps track of a single *pending* text value.  Each call to
    :meth:`update` either refines the pending text or triggers emission.

    Parameters
    ----------
    config:
        Full application config (supplies ``event_id`` and stabilization
        thresholds).
    clock:
        Callable returning the current time in seconds (defaults to
        :func:`time.monotonic`).  Inject a fake clock in tests to make
        time-based assertions deterministic.

    Usage::

        stabilizer = Stabilizer(config)
        # During ongoing speech (streaming mode):
        seg = stabilizer.update(result)
        # When VAD closes the segment:
        seg = stabilizer.update(result, is_final=True)
    """

    def __init__(
        self,
        config: AppConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._event_id: str = config.event_id
        self._silence_ms: int = config.stabilization.silence_ms   # 700 ms default
        self._stable_ms: int = config.stabilization.stable_ms     # 2000 ms default
        self._clock = clock
        self._next_id: int = 1

        # Pending (not-yet-emitted) state
        self._pending_text: str = ""
        self._pending_language: str = ""
        self._last_change_at: float = 0.0   # clock value when pending_text last changed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        result: TranscriptResult,
        *,
        is_final: bool = False,
    ) -> Optional[TranscriptSegment]:
        """Submit a transcript result and apply the stabilization rules.

        Parameters
        ----------
        result:
            Output of :meth:`~transcriptor.transcription.Transcriber.transcribe`.
        is_final:
            ``True`` when the VAD has closed the current speech segment
            (silence rule).  Forces immediate emission of the pending text.

        Returns
        -------
        :class:`TranscriptSegment` when a segment is finalized, else ``None``.
        """
        now = self._clock()
        text = result.text.strip()

        # Refresh pending state whenever the text changes
        if text and text != self._pending_text:
            self._pending_text = text
            self._pending_language = result.language
            self._last_change_at = now

        # Nothing to emit yet
        if not self._pending_text:
            return None

        # Rule 1 — Silence: VAD detected end-of-speech → emit immediately
        if is_final:
            return self._emit()

        # Rule 2 — Stability: text unchanged for stable_ms → emit
        elapsed_ms = (now - self._last_change_at) * 1000.0
        if elapsed_ms >= self._stable_ms:
            return self._emit()

        return None

    @property
    def pending_text(self) -> str:
        """The current candidate text waiting for stabilization."""
        return self._pending_text

    @property
    def next_segment_id(self) -> int:
        """The segment_id that will be assigned to the next emitted segment."""
        return self._next_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self) -> TranscriptSegment:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        segment = TranscriptSegment(
            event_id=self._event_id,
            segment_id=self._next_id,
            sequence_number=self._next_id,
            timestamp=timestamp,
            source_language=self._pending_language,
            text=self._pending_text,
            final=True,
        )
        log.info(
            "Segment %d finalized [%s]: %r",
            self._next_id,
            self._pending_language,
            self._pending_text,
        )
        self._next_id += 1
        self._pending_text = ""
        self._pending_language = ""
        self._last_change_at = 0.0
        return segment
