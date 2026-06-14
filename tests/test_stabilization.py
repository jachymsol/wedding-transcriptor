"""Tests for transcriptor.stabilization — Stabilizer logic and TranscriptSegment.

Time-based tests use an injected FakeClock so no real sleeping is needed.
"""

from __future__ import annotations

import json
import re
import pytest
from unittest.mock import patch

from transcriptor.config import AppConfig, StabilizationConfig
from transcriptor.stabilization import Stabilizer, TranscriptSegment
from transcriptor.transcription import TranscriptResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClock:
    """Controllable monotonic clock for deterministic time-based tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


def _make_config(
    event_id: str = "test-event",
    silence_ms: int = 700,
    stable_ms: int = 2000,
) -> AppConfig:
    return AppConfig(
        event_id=event_id,
        stabilization=StabilizationConfig(silence_ms=silence_ms, stable_ms=stable_ms),
    )


def _make_result(text: str, language: str = "en") -> TranscriptResult:
    return TranscriptResult(text=text, language=language, duration_s=1.0)


def _make_stabilizer(
    event_id: str = "test-event",
    silence_ms: int = 700,
    stable_ms: int = 2000,
    clock: FakeClock | None = None,
) -> tuple[Stabilizer, FakeClock]:
    if clock is None:
        clock = FakeClock()
    config = _make_config(event_id=event_id, silence_ms=silence_ms, stable_ms=stable_ms)
    return Stabilizer(config, clock=clock), clock


# ---------------------------------------------------------------------------
# TranscriptSegment — dataclass and serialization
# ---------------------------------------------------------------------------

class TestTranscriptSegment:
    def _sample(self, **overrides) -> TranscriptSegment:
        defaults = dict(
            event_id="wedding-2027",
            segment_id=101,
            sequence_number=101,
            timestamp="2027-06-15T18:23:45Z",
            source_language="cs",
            text="Mockrát děkujeme",
            final=True,
        )
        defaults.update(overrides)
        return TranscriptSegment(**defaults)

    def test_to_dict_has_all_spec_fields(self):
        d = self._sample().to_dict()
        assert set(d.keys()) == {
            "event_id", "segment_id", "sequence_number",
            "timestamp", "source_language", "text", "final",
        }

    def test_to_dict_values_match_fields(self):
        seg = self._sample()
        d = seg.to_dict()
        assert d["event_id"] == "wedding-2027"
        assert d["segment_id"] == 101
        assert d["sequence_number"] == 101
        assert d["timestamp"] == "2027-06-15T18:23:45Z"
        assert d["source_language"] == "cs"
        assert d["text"] == "Mockrát děkujeme"
        assert d["final"] is True

    def test_to_json_is_valid_json(self):
        payload = self._sample().to_json()
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)

    def test_to_json_preserves_non_ascii(self):
        seg = self._sample(text="Mockrát děkujeme, že jste přišli")
        payload = seg.to_json()
        assert "děkujeme" in payload          # non-ASCII preserved
        assert "\\u" not in payload            # not escaped

    def test_to_json_roundtrip(self):
        seg = self._sample()
        parsed = json.loads(seg.to_json())
        assert parsed == seg.to_dict()

    def test_final_is_always_true_by_default(self):
        seg = TranscriptSegment(
            event_id="e", segment_id=1, sequence_number=1,
            timestamp="t", source_language="en", text="hi",
        )
        assert seg.final is True


# ---------------------------------------------------------------------------
# Stabilizer — initial state
# ---------------------------------------------------------------------------

class TestStabilizerInit:
    def test_pending_text_starts_empty(self):
        s, _ = _make_stabilizer()
        assert s.pending_text == ""

    def test_next_segment_id_starts_at_one(self):
        s, _ = _make_stabilizer()
        assert s.next_segment_id == 1

    def test_event_id_from_config(self):
        s, clock = _make_stabilizer(event_id="my-wedding")
        # Emit a segment and check the event_id
        result = _make_result("Hello")
        seg = s.update(result, is_final=True)
        assert seg.event_id == "my-wedding"


# ---------------------------------------------------------------------------
# Stabilizer — empty / whitespace-only results
# ---------------------------------------------------------------------------

class TestEmptyResults:
    def test_empty_text_produces_no_segment(self):
        s, _ = _make_stabilizer()
        assert s.update(_make_result("")) is None

    def test_whitespace_only_text_produces_no_segment(self):
        s, _ = _make_stabilizer()
        assert s.update(_make_result("   \t\n")) is None

    def test_empty_text_with_is_final_and_no_pending_produces_no_segment(self):
        s, _ = _make_stabilizer()
        assert s.update(_make_result(""), is_final=True) is None


# ---------------------------------------------------------------------------
# Stabilizer — silence rule (is_final=True)
# ---------------------------------------------------------------------------

class TestSilenceRule:
    def test_is_final_emits_segment(self):
        s, _ = _make_stabilizer()
        seg = s.update(_make_result("Hello world"), is_final=True)
        assert seg is not None
        assert isinstance(seg, TranscriptSegment)

    def test_is_final_text_matches_result(self):
        s, _ = _make_stabilizer()
        seg = s.update(_make_result("Hello world"), is_final=True)
        assert seg.text == "Hello world"

    def test_is_final_strips_whitespace(self):
        s, _ = _make_stabilizer()
        seg = s.update(_make_result("  Hello world  "), is_final=True)
        assert seg.text == "Hello world"

    def test_is_final_uses_result_language(self):
        s, _ = _make_stabilizer()
        seg = s.update(_make_result("Ahoj", language="cs"), is_final=True)
        assert seg.source_language == "cs"

    def test_is_final_emits_pending_text_not_new_empty(self):
        """Pending text from a previous update is emitted when is_final arrives."""
        s, _ = _make_stabilizer()
        s.update(_make_result("First text"))            # sets pending
        seg = s.update(_make_result(""), is_final=True)  # empty + is_final → emit pending
        assert seg is not None
        assert seg.text == "First text"

    def test_is_final_emits_most_recent_pending_text(self):
        """If text changed before is_final, the latest text is emitted."""
        s, _ = _make_stabilizer()
        s.update(_make_result("Draft one"))
        s.update(_make_result("Draft two"))
        seg = s.update(_make_result("Final text"), is_final=True)
        assert seg.text == "Final text"

    def test_pending_cleared_after_emission(self):
        s, _ = _make_stabilizer()
        s.update(_make_result("Hello"), is_final=True)
        assert s.pending_text == ""

    def test_no_segment_emitted_without_is_final_before_stable_ms(self):
        """Without is_final, no segment is emitted before the stability window."""
        s, clock = _make_stabilizer(stable_ms=2000)
        s.update(_make_result("Hello"))
        clock.advance_ms(1999)  # just under stable_ms
        assert s.update(_make_result("Hello")) is None


# ---------------------------------------------------------------------------
# Stabilizer — stability rule (elapsed >= stable_ms)
# ---------------------------------------------------------------------------

class TestStabilityRule:
    def test_text_stable_for_stable_ms_emits_segment(self):
        s, clock = _make_stabilizer(stable_ms=2000)
        s.update(_make_result("Hello"))
        clock.advance_ms(2000)
        seg = s.update(_make_result("Hello"))   # text unchanged, time elapsed
        assert seg is not None
        assert seg.text == "Hello"

    def test_text_changed_resets_stability_timer(self):
        s, clock = _make_stabilizer(stable_ms=2000)
        s.update(_make_result("Draft one"))
        clock.advance_ms(1500)
        s.update(_make_result("Draft two"))   # text changed → timer resets
        clock.advance_ms(600)                 # only 600 ms since last change
        assert s.update(_make_result("Draft two")) is None

    def test_text_unchanged_exactly_at_stable_ms_boundary_emits(self):
        s, clock = _make_stabilizer(stable_ms=2000)
        s.update(_make_result("Stable"))
        clock.advance_ms(2000)
        seg = s.update(_make_result("Stable"))
        assert seg is not None

    def test_text_unchanged_just_before_stable_ms_does_not_emit(self):
        s, clock = _make_stabilizer(stable_ms=2000)
        s.update(_make_result("Stable"))
        clock.advance_ms(1999)
        assert s.update(_make_result("Stable")) is None

    def test_stability_rule_uses_clock_not_wall_time(self):
        """Stability is evaluated using the injected clock, not real wall time."""
        clock = FakeClock()
        s, _ = _make_stabilizer(stable_ms=2000, clock=clock)
        s.update(_make_result("Hello"))
        # Advance clock by exactly 2 s without sleeping
        clock.advance(2.0)
        seg = s.update(_make_result("Hello"))
        assert seg is not None


# ---------------------------------------------------------------------------
# Stabilizer — segment metadata
# ---------------------------------------------------------------------------

class TestSegmentMetadata:
    def test_segment_id_starts_at_one(self):
        s, _ = _make_stabilizer()
        seg = s.update(_make_result("Hi"), is_final=True)
        assert seg.segment_id == 1

    def test_sequence_number_equals_segment_id_in_mvp(self):
        s, _ = _make_stabilizer()
        seg = s.update(_make_result("Hi"), is_final=True)
        assert seg.sequence_number == seg.segment_id

    def test_segment_ids_are_monotonically_increasing(self):
        s, _ = _make_stabilizer()
        ids = []
        for text in ["First", "Second", "Third"]:
            seg = s.update(_make_result(text), is_final=True)
            ids.append(seg.segment_id)
        assert ids == [1, 2, 3]

    def test_next_segment_id_increments_after_emission(self):
        s, _ = _make_stabilizer()
        assert s.next_segment_id == 1
        s.update(_make_result("Hi"), is_final=True)
        assert s.next_segment_id == 2

    def test_timestamp_format_matches_spec(self):
        """Timestamp must be ISO 8601 UTC with Z suffix, seconds precision."""
        s, _ = _make_stabilizer()
        seg = s.update(_make_result("Hi"), is_final=True)
        # e.g. "2026-06-14T12:34:56Z"
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", seg.timestamp
        ), f"Unexpected timestamp format: {seg.timestamp!r}"

    def test_event_id_matches_config(self):
        s, _ = _make_stabilizer(event_id="wedding-2027")
        seg = s.update(_make_result("Hi"), is_final=True)
        assert seg.event_id == "wedding-2027"

    def test_final_flag_is_true(self):
        s, _ = _make_stabilizer()
        seg = s.update(_make_result("Hi"), is_final=True)
        assert seg.final is True


# ---------------------------------------------------------------------------
# Stabilizer — consecutive segments
# ---------------------------------------------------------------------------

class TestConsecutiveSegments:
    def test_second_segment_after_first_has_incremented_id(self):
        s, _ = _make_stabilizer()
        s.update(_make_result("First"), is_final=True)
        seg2 = s.update(_make_result("Second"), is_final=True)
        assert seg2.segment_id == 2

    def test_state_resets_between_segments(self):
        """After emitting a segment the stabilizer is ready for the next utterance."""
        s, _ = _make_stabilizer()
        s.update(_make_result("First"), is_final=True)
        assert s.pending_text == ""
        # New text should be tracked fresh
        s.update(_make_result("Second"))
        assert s.pending_text == "Second"

    def test_language_can_change_between_segments(self):
        s, _ = _make_stabilizer()
        seg1 = s.update(_make_result("Hello", language="en"), is_final=True)
        seg2 = s.update(_make_result("Ahoj", language="cs"), is_final=True)
        assert seg1.source_language == "en"
        assert seg2.source_language == "cs"
