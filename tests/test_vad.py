"""Tests for transcriptor.vad — VAD state machine logic.

All tests use a MagicMock in place of the real Silero model so that:
  - No GPU/CPU ML inference is needed
  - Speech/silence probabilities are controlled precisely per test
  - Each test runs in milliseconds

Helper convention
-----------------
``make_vad(**kwargs)`` from conftest returns ``(vad, fake_model)``.
Change ``fake_model.return_value.item.return_value`` to switch the
probability the "model" returns for subsequent VAD windows.
"""

from __future__ import annotations

import numpy as np
import pytest

from transcriptor.vad import SpeechSegment, VoiceActivityDetector, _VAD_WINDOW, _ms_to_windows, _SAMPLE_RATE

from .conftest import CHUNK_SAMPLES, make_vad

# ---------------------------------------------------------------------------
# Reusable audio blocks
# ---------------------------------------------------------------------------

SILENCE = np.zeros(CHUNK_SAMPLES, dtype=np.float32)   # 100 ms silence
SPEECH = np.ones(CHUNK_SAMPLES, dtype=np.float32) * 0.5   # 100 ms "speech"


# ---------------------------------------------------------------------------
# Phase helpers — drive the VAD through speech / silence phases
# ---------------------------------------------------------------------------

def _feed_speech(vad, model, n_chunks: int = 3) -> None:
    """Make the model return speech probability and feed n_chunks."""
    model.return_value.item.return_value = 1.0
    for _ in range(n_chunks):
        vad.process_chunk(SPEECH)


def _feed_silence(vad, model, max_chunks: int = 15) -> SpeechSegment | None:
    """Make the model return silence probability; return first emitted segment."""
    model.return_value.item.return_value = 0.0
    for _ in range(max_chunks):
        seg = vad.process_chunk(SILENCE)
        if seg is not None:
            return seg
    return None


# ---------------------------------------------------------------------------
# _ms_to_windows — pure conversion function
# ---------------------------------------------------------------------------

class TestMsToWindows:
    def test_typical_values(self):
        # 160 ms → 2560 samples → 5 windows of 512
        assert _ms_to_windows(160) == 5
        # 400 ms → 6400 samples → 12 windows
        assert _ms_to_windows(400) == 12
        # 200 ms → 3200 samples → 6 windows
        assert _ms_to_windows(200) == 6

    def test_clamps_to_one_for_very_short_durations(self):
        assert _ms_to_windows(0) == 1
        assert _ms_to_windows(1) == 1
        assert _ms_to_windows(31) == 1  # 31 ms < one 32 ms window

    def test_exact_window_boundary(self):
        # 32 ms = exactly 512 samples = exactly 1 window
        assert _ms_to_windows(32) == 1

    def test_proportional_scaling(self):
        # Doubling ms should approximately double the window count
        w64 = _ms_to_windows(64)
        w128 = _ms_to_windows(128)
        assert w128 == 2 * w64


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_in_silence(self):
        vad, _ = make_vad()
        assert vad.state == "SILENCE"

    def test_leftover_buffer_starts_empty(self):
        vad, _ = make_vad()
        assert len(vad._leftover) == 0

    def test_speech_buffer_starts_empty(self):
        vad, _ = make_vad()
        assert vad._speech_buf == []


# ---------------------------------------------------------------------------
# SILENCE state behaviour
# ---------------------------------------------------------------------------

class TestSilenceState:
    def test_silence_chunks_produce_no_segment(self):
        vad, model = make_vad()
        for _ in range(20):
            seg = vad.process_chunk(SILENCE)
            assert seg is None

    def test_silence_chunks_keep_state_as_silence(self):
        vad, model = make_vad()
        for _ in range(20):
            vad.process_chunk(SILENCE)
        assert vad.state == "SILENCE"

    def test_brief_speech_below_threshold_does_not_start_segment(self):
        """Fewer consecutive speech windows than speech_start_ms → stays SILENCE."""
        # Default speech_start_ms=160 → needs 5 windows.
        # One 100 ms chunk gives 3 windows.  3 < 5, so no transition.
        vad, model = make_vad()
        model.return_value.item.return_value = 1.0
        vad.process_chunk(SPEECH)  # 3 speech windows
        # Switch back to silence so the counter resets on the next chunk
        model.return_value.item.return_value = 0.0
        vad.process_chunk(SILENCE)
        assert vad.state == "SILENCE"


# ---------------------------------------------------------------------------
# SILENCE → SPEECH transition
# ---------------------------------------------------------------------------

class TestSpeechTransition:
    def test_sufficient_speech_enters_speech_state(self):
        """3 chunks = 9+ speech windows > start_wins(5) → state becomes SPEECH."""
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        assert vad.state == "SPEECH"

    def test_speech_buffer_is_populated_on_entry(self):
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        assert len(vad._speech_buf) > 0

    def test_pre_roll_prepended_to_speech_buffer(self):
        """After entering SPEECH, speech_buf contains the pre-roll silence windows."""
        vad, model = make_vad(pre_roll_ms=200)   # default 6 windows pre-roll

        # Feed 2 silence chunks to populate the pre-roll deque
        model.return_value.item.return_value = 0.0
        vad.process_chunk(SILENCE)
        vad.process_chunk(SILENCE)

        # Now enter speech
        _feed_speech(vad, model, n_chunks=3)
        assert vad.state == "SPEECH"

        # Speech buffer must contain pre-roll audio (≥ 200 ms worth = 3200 samples)
        speech_buf_samples = sum(len(w) for w in vad._speech_buf)
        pre_roll_min_samples = 200 * 16_000 // 1000   # 3200
        assert speech_buf_samples >= pre_roll_min_samples, (
            f"Expected ≥{pre_roll_min_samples} samples in speech_buf, "
            f"got {speech_buf_samples}"
        )


# ---------------------------------------------------------------------------
# SPEECH → SILENCE transition (segment emission)
# ---------------------------------------------------------------------------

class TestSegmentEmission:
    def test_segment_emitted_after_sustained_silence(self):
        """Full cycle: speech then sustained silence → SpeechSegment returned."""
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        seg = _feed_silence(vad, model)

        assert seg is not None
        assert isinstance(seg, SpeechSegment)

    def test_emitted_segment_has_float32_audio(self):
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        seg = _feed_silence(vad, model)
        assert seg.audio.dtype == np.float32

    def test_emitted_segment_is_non_empty(self):
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        seg = _feed_silence(vad, model)
        assert len(seg.audio) > 0

    def test_silence_triggered_segment_is_final(self):
        """Segments closed by silence detection must have is_final=True."""
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        seg = _feed_silence(vad, model)
        assert seg is not None
        assert seg.is_final is True

    def test_state_returns_to_silence_after_emission(self):
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        _feed_silence(vad, model)
        assert vad.state == "SILENCE"

    def test_speech_buffer_cleared_after_emission(self):
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        _feed_silence(vad, model)
        assert vad._speech_buf == []

    def test_brief_silence_mid_speech_does_not_emit(self):
        """Silence shorter than speech_end_ms does not close a segment."""
        # Default speech_end_ms=400 → needs 12 windows.
        # 2 silence chunks = 6 windows < 12 → no emission.
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        assert vad.state == "SPEECH"

        seg1 = _feed_silence(vad, model, max_chunks=2)   # only 2 chunks (6 windows)
        assert seg1 is None, "Brief silence (6 windows < 12 end_wins) must not emit"
        assert vad.state == "SPEECH"

    def test_model_reset_states_called_after_emission(self):
        """model.reset_states() is called once per emitted segment."""
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        _feed_silence(vad, model)
        model.reset_states.assert_called_once()


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------

class TestFlush:
    def test_flush_in_speech_emits_segment(self):
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        assert vad.state == "SPEECH"

        seg = vad.flush()
        assert seg is not None
        assert isinstance(seg, SpeechSegment)
        assert vad.state == "SILENCE"

    def test_flush_segment_is_final(self):
        """flush() always produces a final segment (speech ended)."""
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        seg = vad.flush()
        assert seg is not None
        assert seg.is_final is True

    def test_flush_in_silence_returns_none(self):
        vad, _ = make_vad()
        assert vad.flush() is None

    def test_flush_clears_speech_buffer(self):
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        vad.flush()
        assert vad._speech_buf == []


# ---------------------------------------------------------------------------
# speech_segments() convenience generator
# ---------------------------------------------------------------------------

class TestSpeechSegmentsGenerator:
    def test_yields_segment_for_each_utterance(self):
        """Generator yields one SpeechSegment for the speech + silence sequence."""
        vad, model = make_vad()

        speech_chunks = [SPEECH] * 3
        silence_chunks = [SILENCE] * 15
        chunks = iter(speech_chunks + silence_chunks)

        def controlled_chunks():
            for i, c in enumerate(speech_chunks + silence_chunks):
                if i < len(speech_chunks):
                    model.return_value.item.return_value = 1.0
                else:
                    model.return_value.item.return_value = 0.0
                yield c

        segments = list(vad.speech_segments(controlled_chunks()))
        assert len(segments) >= 1
        assert all(isinstance(s, SpeechSegment) for s in segments)

    def test_generator_calls_flush_at_end(self):
        """If speech is ongoing when the chunk iterator is exhausted, flush is called."""
        vad, model = make_vad()

        def only_speech():
            model.return_value.item.return_value = 1.0
            for _ in range(5):   # speak but never silence → no in-loop emission
                yield SPEECH

        segments = list(vad.speech_segments(only_speech()))
        # flush() at the end should yield the trailing segment
        assert len(segments) == 1


# ---------------------------------------------------------------------------
# Leftover buffer — chunk boundary arithmetic
# ---------------------------------------------------------------------------

class TestLeftoverBuffer:
    def test_no_exception_across_many_chunks(self):
        """Processing many chunks does not raise any exception."""
        vad, _ = make_vad()
        for _ in range(100):
            vad.process_chunk(SILENCE)   # must not raise

    def test_leftover_resets_after_eight_chunks(self):
        """After 8 chunks, accumulated 8×64=512 leftover samples form one extra window.

        Chunk n contributes 1600 samples; with 512-sample windows there is a
        64-sample remainder each time.  After 8 chunks, 8×64=512 bytes of
        leftover have accumulated — exactly one window — which is consumed,
        leaving leftover=0.
        """
        vad, _ = make_vad()
        for _ in range(8):
            vad.process_chunk(SILENCE)
        assert len(vad._leftover) == 0

    def test_leftover_correct_after_ten_chunks(self):
        """After 10 chunks, leftover = (10×64) mod 512 = 640 mod 512 = 128 samples."""
        vad, _ = make_vad()
        for _ in range(10):
            vad.process_chunk(SILENCE)
        assert len(vad._leftover) == 128

    def test_segment_audio_length_is_multiple_of_vad_window(self):
        """Emitted segment audio is assembled from complete 512-sample windows."""
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=5)
        seg = _feed_silence(vad, model)
        assert seg is not None
        assert len(seg.audio) % _VAD_WINDOW == 0


# ---------------------------------------------------------------------------
# Known limitation: two segments completing in one chunk
# ---------------------------------------------------------------------------

class TestKnownLimitations:
    def test_only_last_segment_returned_when_two_end_in_one_chunk(self):
        """If two utterances complete within a single process_chunk() call,
        only the last SpeechSegment is returned.  This is a documented
        limitation (rare in practice given 100 ms chunks and 400 ms end_ms).
        """
        # Construct a scenario with tiny thresholds so two segments can fit
        # inside one 1600-sample / 3-window chunk.
        # start_wins=1, end_wins=1 means:
        #   window 0: speech  → SPEECH (start triggered)
        #   window 1: silence → emit segment A, back to SILENCE
        #   window 2: speech  → SPEECH again (start triggered with 1 window)
        # Three windows in one chunk → segment A is completed and silently lost;
        # process_chunk returns None (state is SPEECH after window 2, not yet closed).
        #
        # To get a second *emission* in the same chunk we'd need 4 windows:
        #   window 3: silence → emit segment B
        # But 1600 samples only gives 3 full 512-sample windows (+ 64 leftover),
        # so two full emissions in one chunk aren't achievable with default
        # CHUNK_SAMPLES.  Instead we verify the behaviour with a shorter chunk.

        vad, model = make_vad(speech_start_ms=32, speech_end_ms=32)
        # Probabilities for 4 windows: speech, silence, speech, silence
        probs = iter([1.0, 0.0, 1.0, 0.0])
        model.return_value.item.side_effect = probs

        # Feed a chunk long enough for 4 windows (4 × 512 = 2048 samples)
        big_chunk = np.zeros(2048, dtype=np.float32)
        seg = vad.process_chunk(big_chunk)

        # Only the LAST emitted segment is returned (the first is silently dropped)
        assert isinstance(seg, SpeechSegment)
        assert vad.state == "SILENCE"


# ---------------------------------------------------------------------------
# max_speech_ms — partial-emit behaviour
# ---------------------------------------------------------------------------

# Convenient window count for tests: 4 windows = 4 × 512 = 2048 samples ≈ 128 ms
_FOUR_WINDOWS = np.zeros(4 * _VAD_WINDOW, dtype=np.float32)


def _make_vad_with_max(max_speech_ms: int, overlap_ms: int = 0) -> tuple:
    """VAD with tiny max_speech_ms so tests don't need thousands of chunks."""
    model = make_vad(max_speech_ms=max_speech_ms, overlap_ms=overlap_ms)[1]
    vad = VoiceActivityDetector(
        speech_start_ms=32,    # 1 window to enter SPEECH
        speech_end_ms=32,      # 1 window to leave SPEECH
        pre_roll_ms=0,
        max_speech_ms=max_speech_ms,
        overlap_ms=overlap_ms,
        model=model,
    )
    return vad, model


class TestMaxSpeechMs:
    def test_partial_emitted_when_buffer_exceeds_max(self):
        """When accumulated speech reaches max_speech_ms, a partial is emitted."""
        # max_speech_ms equivalent to 4 windows
        max_ms = 4 * _VAD_WINDOW * 1000 // _SAMPLE_RATE  # ≈ 128 ms
        vad, model = _make_vad_with_max(max_ms, overlap_ms=0)

        # Enter SPEECH
        model.return_value.item.return_value = 1.0
        vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))  # 1 window → SPEECH

        # Feed windows one at a time; partial fires when buffer hits max
        seg = None
        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                seg = result
                break

        assert seg is not None
        assert isinstance(seg, SpeechSegment)

    def test_partial_segment_is_not_final(self):
        """Partial segments emitted by max_speech_ms have is_final=False."""
        max_ms = 4 * _VAD_WINDOW * 1000 // _SAMPLE_RATE
        vad, model = _make_vad_with_max(max_ms, overlap_ms=0)

        model.return_value.item.return_value = 1.0
        vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))

        seg = None
        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                seg = result
                break

        assert seg is not None
        assert seg.is_final is False

    def test_state_stays_speech_after_partial(self):
        """After a partial emit, VAD remains in SPEECH state."""
        max_ms = 4 * _VAD_WINDOW * 1000 // _SAMPLE_RATE
        vad, model = _make_vad_with_max(max_ms, overlap_ms=0)

        model.return_value.item.return_value = 1.0
        vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))

        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                break

        assert vad.state == "SPEECH"

    def test_model_reset_not_called_after_partial(self):
        """GRU state must NOT be reset after a partial emit (speech is ongoing)."""
        max_ms = 4 * _VAD_WINDOW * 1000 // _SAMPLE_RATE
        vad, model = _make_vad_with_max(max_ms, overlap_ms=0)

        model.return_value.item.return_value = 1.0
        vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))

        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                break

        model.reset_states.assert_not_called()

    def test_overlap_windows_kept_in_buffer_after_partial(self):
        """After a partial emit, the buffer retains exactly overlap_windows entries."""
        # 2 overlap windows
        overlap_ms = 2 * _VAD_WINDOW * 1000 // _SAMPLE_RATE  # ≈ 64 ms
        max_ms = 4 * _VAD_WINDOW * 1000 // _SAMPLE_RATE      # ≈ 128 ms
        vad, model = _make_vad_with_max(max_ms, overlap_ms=overlap_ms)

        model.return_value.item.return_value = 1.0
        vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))  # enter SPEECH

        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                break

        assert len(vad._speech_buf) == 2  # exactly overlap_windows

    def test_no_partial_when_max_speech_ms_zero(self):
        """max_speech_ms=0 disables partial emission entirely."""
        vad, model = _make_vad_with_max(max_speech_ms=0, overlap_ms=0)

        model.return_value.item.return_value = 1.0
        # Feed many windows — no partial should fire
        seg = None
        for _ in range(50):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                seg = result
                break

        assert seg is None
        assert vad.state == "SPEECH"

    def test_silence_after_partial_emits_final_segment(self):
        """After a partial emit, sustained silence closes the remaining audio as final."""
        max_ms = 4 * _VAD_WINDOW * 1000 // _SAMPLE_RATE
        vad, model = _make_vad_with_max(max_ms, overlap_ms=0)

        # Enter SPEECH and trigger a partial
        model.return_value.item.return_value = 1.0
        vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
        partial = None
        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                partial = result
                break
        assert partial is not None and partial.is_final is False

        # Now silence → should emit the remaining speech as final
        model.return_value.item.return_value = 0.0
        final = None
        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                final = result
                break

        assert final is not None
        assert final.is_final is True
        assert vad.state == "SILENCE"


# ---------------------------------------------------------------------------
# Latency timestamps — speech_start_mono / emit_mono
# ---------------------------------------------------------------------------

class TestLatencyTimestamps:
    def test_final_segment_speech_start_mono_is_positive(self):
        """speech_start_mono is set (> 0) when a final segment is emitted."""
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        seg = _feed_silence(vad, model)
        assert seg is not None
        assert seg.speech_start_mono > 0

    def test_final_segment_emit_mono_after_speech_start(self):
        """emit_mono is ≥ speech_start_mono for a final segment."""
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        seg = _feed_silence(vad, model)
        assert seg is not None
        assert seg.emit_mono >= seg.speech_start_mono

    def test_flush_segment_has_timestamps_set(self):
        """flush() produces a segment with both timestamp fields set."""
        vad, model = make_vad()
        _feed_speech(vad, model, n_chunks=3)
        seg = vad.flush()
        assert seg is not None
        assert seg.speech_start_mono > 0
        assert seg.emit_mono >= seg.speech_start_mono

    def test_partial_segment_speech_start_mono_is_positive(self):
        """speech_start_mono is set (> 0) when a partial segment is emitted."""
        max_ms = 4 * _VAD_WINDOW * 1000 // _SAMPLE_RATE
        vad, model = _make_vad_with_max(max_ms, overlap_ms=0)

        model.return_value.item.return_value = 1.0
        vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))

        seg = None
        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                seg = result
                break

        assert seg is not None and seg.is_final is False
        assert seg.speech_start_mono > 0

    def test_partial_segment_emit_mono_after_speech_start(self):
        """emit_mono is ≥ speech_start_mono for a partial segment."""
        max_ms = 4 * _VAD_WINDOW * 1000 // _SAMPLE_RATE
        vad, model = _make_vad_with_max(max_ms, overlap_ms=0)

        model.return_value.item.return_value = 1.0
        vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))

        seg = None
        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                seg = result
                break

        assert seg is not None
        assert seg.emit_mono >= seg.speech_start_mono

    def test_second_partial_speech_start_is_prev_emit_minus_overlap(self):
        """Option B: second partial's speech_start_mono = first partial's emit_mono − overlap_s.

        With 2 overlap windows: overlap_s = 2 × 512 / 16000 = 0.064 s.
        After the first partial is emitted at time T, _speech_start_mono is set to
        T − 0.064 s.  The second partial must carry that value as speech_start_mono.
        """
        overlap_windows = 2
        overlap_ms = overlap_windows * _VAD_WINDOW * 1000 // _SAMPLE_RATE  # ≈ 64 ms
        max_ms = 4 * _VAD_WINDOW * 1000 // _SAMPLE_RATE                    # ≈ 128 ms
        overlap_s = overlap_windows * _VAD_WINDOW / _SAMPLE_RATE           # exact

        vad, model = _make_vad_with_max(max_ms, overlap_ms=overlap_ms)
        model.return_value.item.return_value = 1.0

        # Enter SPEECH (1 window with speech_start_ms=32 → 1 window threshold)
        vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))

        # Collect first partial
        seg1 = None
        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                seg1 = result
                break
        assert seg1 is not None and seg1.is_final is False

        # Collect second partial
        seg2 = None
        for _ in range(10):
            result = vad.process_chunk(np.zeros(_VAD_WINDOW, dtype=np.float32))
            if result is not None:
                seg2 = result
                break
        assert seg2 is not None and seg2.is_final is False

        # Key invariant: seg2.speech_start_mono == seg1.emit_mono - overlap_s
        assert seg2.speech_start_mono == pytest.approx(seg1.emit_mono - overlap_s)
