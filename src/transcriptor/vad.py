"""Voice Activity Detection module — FR-003.

Uses Silero VAD to detect speech regions in 16 kHz mono audio.
Emits SpeechSegment objects when a complete utterance is detected.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

# Silero VAD requires exactly 512 samples per window at 16 kHz (≈ 32 ms)
_VAD_WINDOW: int = 512
_SAMPLE_RATE: int = 16_000

# Default VAD parameters
_SPEECH_THRESHOLD: float = 0.5
_SPEECH_START_MS: int = 160   # must hear this long to open a segment  (~5 windows)
_SPEECH_END_MS: int = 400     # this much silence closes a segment     (~12 windows)
_PRE_ROLL_MS: int = 200       # audio prepended before speech onset (captures word start)


def _ms_to_windows(ms: int) -> int:
    """Convert a duration in milliseconds to a number of VAD windows."""
    return max(1, (_SAMPLE_RATE * ms // 1000) // _VAD_WINDOW)


class _State(Enum):
    SILENCE = auto()
    SPEECH = auto()


@dataclass
class SpeechSegment:
    """A complete speech utterance ready for transcription.

    Attributes:
        audio: float32 mono array at 16 kHz, concatenated from VAD windows.
    """
    audio: np.ndarray


class VoiceActivityDetector:
    """Streaming VAD built on Silero VAD.

    Consumes raw 100 ms audio chunks from :class:`AudioCapture` and emits
    :class:`SpeechSegment` objects when a speech utterance ends.

    State machine::

        SILENCE ──(speech_start_ms of speech)──► SPEECH
        SPEECH  ──(speech_end_ms of silence)───► SILENCE  → emit segment

    Usage::

        vad = VoiceActivityDetector()
        for chunk in capture.chunks():
            seg = vad.process_chunk(chunk)
            if seg is not None:
                transcribe(seg.audio)
        # Flush any trailing speech when stopping
        final = vad.flush()

    Or via the convenience generator::

        for seg in vad.speech_segments(capture.chunks()):
            transcribe(seg.audio)
    """

    def __init__(
        self,
        threshold: float = _SPEECH_THRESHOLD,
        speech_start_ms: int = _SPEECH_START_MS,
        speech_end_ms: int = _SPEECH_END_MS,
        pre_roll_ms: int = _PRE_ROLL_MS,
    ) -> None:
        log.info("Loading Silero VAD model …")
        from silero_vad import load_silero_vad  # deferred: heavy import
        self._model = load_silero_vad()
        self._model.eval()
        log.info("Silero VAD model ready")

        self._threshold = threshold
        self._start_wins = _ms_to_windows(speech_start_ms)
        self._end_wins = _ms_to_windows(speech_end_ms)

        # Pre-roll ring buffer: recent silence windows prepended to each segment
        # so word onsets aren't clipped.
        pre_roll_capacity = _ms_to_windows(pre_roll_ms) + self._start_wins
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_capacity)

        self._state = _State.SILENCE
        self._leftover = np.empty(0, dtype=np.float32)  # sub-window tail between chunks
        self._speech_buf: list[np.ndarray] = []
        self._consec_speech = 0
        self._consec_silence = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> Optional[SpeechSegment]:
        """Process one 100 ms audio chunk.

        Returns a :class:`SpeechSegment` when an utterance ends, else ``None``.
        If two utterances happen to end inside the same chunk (rare), the last
        one is returned; the first is still emitted via the log.
        """
        audio = np.concatenate([self._leftover, chunk])
        result: Optional[SpeechSegment] = None
        offset = 0

        while offset + _VAD_WINDOW <= len(audio):
            window = audio[offset: offset + _VAD_WINDOW]
            offset += _VAD_WINDOW
            is_speech = self._speech_prob(window) >= self._threshold
            seg = self._step(window, is_speech)
            if seg is not None:
                result = seg

        self._leftover = audio[offset:].copy()
        return result

    def flush(self) -> Optional[SpeechSegment]:
        """Force-emit any accumulated speech.

        Call this when audio capture stops so trailing speech is not lost.
        """
        if self._state == _State.SPEECH and self._speech_buf:
            log.debug("VAD flush: emitting trailing speech")
            return self._emit()
        return None

    def speech_segments(
        self, chunks: Iterator[np.ndarray]
    ) -> Iterator[SpeechSegment]:
        """Convenience generator that wraps :meth:`process_chunk` and :meth:`flush`.

        Yields each detected speech segment in order.
        """
        for chunk in chunks:
            seg = self.process_chunk(chunk)
            if seg is not None:
                yield seg
        final = self.flush()
        if final is not None:
            yield final

    @property
    def state(self) -> str:
        """Current VAD state as a string (``"SILENCE"`` or ``"SPEECH"``)."""
        return self._state.name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _speech_prob(self, window: np.ndarray) -> float:
        tensor = torch.from_numpy(window)
        return float(self._model(tensor, _SAMPLE_RATE).item())

    def _step(self, window: np.ndarray, is_speech: bool) -> Optional[SpeechSegment]:
        """Advance the state machine by one VAD window."""
        if self._state == _State.SILENCE:
            self._pre_roll.append(window)
            if is_speech:
                self._consec_speech += 1
                self._consec_silence = 0
                if self._consec_speech >= self._start_wins:
                    self._begin_speech()
            else:
                self._consec_speech = 0
            return None

        else:  # SPEECH
            self._speech_buf.append(window)
            if not is_speech:
                self._consec_silence += 1
                self._consec_speech = 0
                if self._consec_silence >= self._end_wins:
                    return self._emit()
            else:
                self._consec_speech += 1
                self._consec_silence = 0
            return None

    def _begin_speech(self) -> None:
        """Transition SILENCE → SPEECH.

        Seeds the speech buffer with the pre-roll so word onsets are kept.
        """
        self._state = _State.SPEECH
        self._speech_buf = list(self._pre_roll)  # pre-roll windows (includes onset)
        self._consec_silence = 0
        log.debug("Speech started")

    def _emit(self) -> SpeechSegment:
        """Transition SPEECH → SILENCE and return the accumulated segment."""
        audio = (
            np.concatenate(self._speech_buf)
            if self._speech_buf
            else np.empty(0, dtype=np.float32)
        )
        duration_s = len(audio) / _SAMPLE_RATE
        log.debug("Speech segment: %.2f s", duration_s)
        # Reset state
        self._state = _State.SILENCE
        self._speech_buf = []
        self._consec_speech = 0
        self._consec_silence = 0
        self._pre_roll.clear()
        self._model.reset_states()  # clear GRU state for next utterance
        return SpeechSegment(audio=audio)


# ---------------------------------------------------------------------------
# Manual smoke-test: python -m transcriptor.vad
# Shows live SPEECH / SILENCE state and prints segment durations.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time
    from transcriptor.audio import AudioCapture
    from transcriptor.config import load_config
    from transcriptor.logging_setup import setup_logging

    setup_logging()
    cfg = load_config()

    capture = AudioCapture(cfg.audio)
    vad = VoiceActivityDetector()

    capture.start()
    print("Listening — speak to test VAD (Ctrl+C to stop).\n")

    prev_state = vad.state
    try:
        for chunk in capture.chunks():
            seg = vad.process_chunk(chunk)

            current = vad.state
            if current != prev_state:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] {prev_state} → {current}")
                prev_state = current

            if seg is not None:
                duration = len(seg.audio) / _SAMPLE_RATE
                print(f"  *** Segment ready: {duration:.2f} s  ({len(seg.audio)} samples)")

    except KeyboardInterrupt:
        pass
    finally:
        final = vad.flush()
        if final is not None:
            duration = len(final.audio) / _SAMPLE_RATE
            print(f"  *** Final segment (flush): {duration:.2f} s")
        capture.stop()
        print("Done.")
