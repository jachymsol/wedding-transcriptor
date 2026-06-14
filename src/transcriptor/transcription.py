"""Whisper transcription module — FR-004.

Wraps faster-whisper with automatic CUDA→CPU fallback and a mutable
language setting that can be changed at runtime by the operator (FR-002).

Model weights are downloaded from Hugging Face on first use and cached in
~/.cache/huggingface/hub/.  Subsequent starts use the local cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from transcriptor.config import TranscriptionConfig

log = logging.getLogger(__name__)

_SAMPLE_RATE: int = 16_000       # must match AudioCapture and VAD
_CUDA_COMPUTE: str = "float16"
_CPU_COMPUTE: str = "int8"
_CPU_MODEL: str = "small"        # spec §4 fallback


@dataclass
class Word:
    """A single word with timing, returned when ``word_timestamps=True``.

    Attributes:
        word:  The recognised word (may include leading/trailing spaces).
        start: Start time in seconds, relative to the start of the audio clip.
        end:   End time in seconds, relative to the start of the audio clip.
    """

    word: str
    start: float
    end: float


@dataclass
class TranscriptResult:
    """Output of a single transcription pass.

    Attributes:
        text:       Joined, stripped transcript text.
        language:   Language code used (e.g. ``"en"``, ``"cs"``).
        duration_s: Duration of the source audio in seconds.
        words:      Per-word timing list (populated when ``word_timestamps``
                    is available; empty list otherwise).
    """

    text: str
    language: str
    duration_s: float
    words: list = field(default_factory=list)  # list[Word]


def _load_whisper_model(model_name: str) -> Any:
    """Load a WhisperModel with CUDA→CPU fallback.

    Tries ``cuda / float16`` first (spec §4 default); falls back to
    ``cpu / int8`` with the *small* model if CUDA is unavailable or
    initialisation fails.

    The ``faster_whisper`` import is deferred so the module can be
    imported (and tested with a mock) without the library installed.
    """
    import torch
    from faster_whisper import WhisperModel

    if torch.cuda.is_available():
        try:
            model = WhisperModel(model_name, device="cuda", compute_type=_CUDA_COMPUTE)
            log.info("Whisper '%s' loaded on CUDA (%s)", model_name, _CUDA_COMPUTE)
            return model
        except Exception as exc:  # noqa: BLE001
            log.warning("CUDA init failed (%s) — falling back to CPU", exc)

    model = WhisperModel(_CPU_MODEL, device="cpu", compute_type=_CPU_COMPUTE)
    log.info("Whisper '%s' loaded on CPU (%s)", _CPU_MODEL, _CPU_COMPUTE)
    return model


class Transcriber:
    """Transcribes speech audio using a local faster-whisper model.

    The active language can be changed at any time via :meth:`set_language`
    so the operator can switch between speakers mid-event (FR-002).

    Usage::

        transcriber = Transcriber(config.transcription)
        result = transcriber.transcribe(segment.audio)
        print(result.text)
    """

    def __init__(self, config: TranscriptionConfig, model: Any = None) -> None:
        """
        Parameters
        ----------
        config:
            Transcription settings from ``config.yaml``.
        model:
            Optional pre-built ``WhisperModel``.  When *None* the model is
            loaded via :func:`_load_whisper_model`.  Pass a mock in tests to
            avoid downloading real weights.
        """
        if model is None:
            model = _load_whisper_model(config.model)
        self._model = model
        self._language: str = config.language

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        """Currently active source language code (e.g. ``"en"``, ``"cs"``)."""
        return self._language

    def set_language(self, language: str) -> None:
        """Change the source language for all subsequent transcriptions."""
        log.info("Language changed: %s → %s", self._language, language)
        self._language = language

    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """Transcribe a mono float32 audio array captured at 16 kHz.

        Parameters
        ----------
        audio:
            1-D ``float32`` numpy array — the ``audio`` field of a
            :class:`~transcriptor.vad.SpeechSegment`.

        Returns
        -------
        TranscriptResult
            Joined transcript text, language used, and audio duration.
        """
        duration_s = len(audio) / _SAMPLE_RATE

        # vad_filter=False: our pipeline already runs Silero VAD externally.
        # word_timestamps=True: needed for partial-commit boundary detection
        # when max_speech_ms forces a mid-speech segment split.
        segments_iter, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=False,
            word_timestamps=True,
        )

        # Materialise the lazy iterator — faster-whisper yields segments on demand.
        all_words: list[Word] = []
        texts: list[str] = []
        for seg in segments_iter:
            if seg.text.strip():
                texts.append(seg.text.strip())
            for w in (seg.words or []):
                all_words.append(Word(word=w.word, start=w.start, end=w.end))

        text = " ".join(texts)

        result = TranscriptResult(
            text=text,
            language=self._language,
            duration_s=duration_s,
            words=all_words,
        )
        log.info(
            "Transcribed %.1f s [%s]: %r (%d words)",
            duration_s, self._language, text, len(all_words),
        )
        return result


# ---------------------------------------------------------------------------
# Manual smoke-test: python -m transcriptor.transcription
# Connects audio → VAD → Whisper and prints live transcription.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transcriptor.audio import AudioCapture
    from transcriptor.config import load_config
    from transcriptor.logging_setup import setup_logging
    from transcriptor.vad import VoiceActivityDetector

    setup_logging()
    cfg = load_config()

    print("Loading Whisper model (first run may download weights) …")
    transcriber = Transcriber(cfg.transcription)
    print(f"Ready — language={transcriber.language}\n")

    capture = AudioCapture(cfg.audio)
    vad = VoiceActivityDetector()
    capture.start()
    print("Speak into the microphone. Press Ctrl+C to stop.\n")

    try:
        for seg in vad.speech_segments(capture.chunks()):
            dur = len(seg.audio) / _SAMPLE_RATE
            print(f"[{dur:.1f} s] transcribing …", end=" ", flush=True)
            result = transcriber.transcribe(seg.audio)
            print(f"→ {result.text!r}")
    except KeyboardInterrupt:
        pass
    finally:
        final = vad.flush()
        if final is not None and len(final.audio) > 0:
            result = transcriber.transcribe(final.audio)
            if result.text:
                print(f"[final] → {result.text!r}")
        capture.stop()
        print("Done.")
