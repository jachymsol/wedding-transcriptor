"""Whisper transcription module — FR-004.

Uses mlx-whisper for Apple MLX GPU-accelerated inference on Apple Silicon
(macOS), and faster-whisper (CPU) on other platforms. The backend is chosen
automatically based on the host OS, or can be forced via
``TranscriptionConfig.backend``. Model weights are downloaded from Hugging
Face on first use and cached locally; subsequent starts use the cache.

The model is pre-warmed during construction by running a short dummy clip
through it, so the first real speech segment does not pay the weight-loading
latency cost.
"""

from __future__ import annotations

import logging
import platform
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from transcriptor.config import TranscriptionConfig

log = logging.getLogger(__name__)

_SAMPLE_RATE: int = 16_000  # must match AudioCapture and VAD

# Short-name → full HF repo mapping for mlx-community models.
_MLX_REPO_MAP: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large": "mlx-community/whisper-large-v3",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3",
}

# Short-name → faster-whisper (CTranslate2) model name mapping. faster-whisper
# accepts these size names directly (auto-downloaded from HF on first use),
# so most short names pass through unchanged; this map only exists for
# consistency and any future renames.
_FASTER_WHISPER_MODEL_MAP: dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large-v3",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
}


def _resolve_mlx_repo(model: str) -> str:
    """Map a short model name to a full HF repo string.

    Names that already contain ``"/"`` are returned unchanged, so users can
    specify any MLX repo directly (e.g. ``"mlx-community/whisper-medium-4bit"``).
    Short names (e.g. ``"medium"``) are looked up in *_MLX_REPO_MAP*;
    unrecognised short names fall back to ``f"mlx-community/whisper-{model}"``.
    """
    if "/" in model:
        return model
    return _MLX_REPO_MAP.get(model, f"mlx-community/whisper-{model}")


def _resolve_fw_model(model: str) -> str:
    """Map a model name to a faster-whisper (CTranslate2) model name/path.

    Names already containing ``"/"`` (a local path or a full HF repo id for a
    CTranslate2-converted model) are returned unchanged. Short names are
    looked up in *_FASTER_WHISPER_MODEL_MAP*; unrecognised short names are
    passed through as-is (faster-whisper accepts arbitrary size strings).
    """
    if "/" in model:
        return model
    return _FASTER_WHISPER_MODEL_MAP.get(model, model)


def _resolve_backend_kind(config: TranscriptionConfig) -> str:
    """Return ``"mlx"`` or ``"faster-whisper"`` for *config.backend*.

    ``"auto"`` (the default) picks mlx-whisper on macOS and faster-whisper
    on every other platform.
    """
    backend = config.backend
    if backend in ("mlx", "faster-whisper"):
        return backend
    if backend != "auto":
        log.warning("Unknown transcription.backend %r — falling back to auto", backend)
    return "mlx" if platform.system() == "Darwin" else "faster-whisper"


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


_HALLUCINATION_MAX_PERIOD: int = 4
_HALLUCINATION_MIN_RUN_TOKENS: int = 6
_HALLUCINATION_MIN_COVERAGE: float = 0.65


def _find_loop_start(tokens: list[str]) -> int | None:
    """Return the index where a repeating (hallucinated) loop begins, or None.

    Detects Whisper decoder loops of period 1..*_HALLUCINATION_MAX_PERIOD*
    (e.g. "the the the the", or "H instagramuawat H instagramuawat …" which
    is a period-2 loop). For each period *p*, a token at index *i* is
    considered a "repeat" if it equals the token *p* positions earlier. The
    longest run of consecutive repeats (for any period) is found; if that
    run covers at least *_HALLUCINATION_MIN_COVERAGE* of the tokens from its
    start to the end of the segment, and spans at least
    *_HALLUCINATION_MIN_RUN_TOKENS* tokens (including the seed span of
    length *p*), the segment is flagged as looping from that point onward.

    When multiple periods qualify, the earliest starting index across all of
    them is returned, so the segment is truncated at the true loop onset
    rather than at the first period checked.

    Returns the token index to truncate at (keep ``tokens[:index]``), or
    ``None`` if no loop is detected.
    """
    n = len(tokens)
    if n == 0:
        return None

    best_start: int | None = None

    for p in range(1, _HALLUCINATION_MAX_PERIOD + 1):
        if n <= p:
            continue
        # is_repeat[i] is True when tokens[i] == tokens[i-p], for i in [p, n).
        run_start: int | None = None
        for i in range(p, n + 1):
            is_repeat = i < n and tokens[i] == tokens[i - p]
            if is_repeat:
                if run_start is None:
                    run_start = i - p  # seed of the run starts p tokens back
            else:
                if run_start is not None:
                    loop_start = run_start
                    run_tokens = i - loop_start
                    coverage = run_tokens / (n - loop_start)
                    if run_tokens >= _HALLUCINATION_MIN_RUN_TOKENS and coverage >= _HALLUCINATION_MIN_COVERAGE:
                        if best_start is None or loop_start < best_start:
                            best_start = loop_start
                    run_start = None

    return best_start


_CJK_PATTERN = re.compile(
    r"[\u3000-\u303F"                              # CJK punctuation (、。「」etc.)
    r"\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F"      # Hiragana, Katakana, halfwidth Katakana
    r"\u4E00-\u9FFF\u3400-\u4DBF"                    # CJK Unified Ideographs (+ Ext A)
    r"\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]"      # Hangul syllables / jamo
)


def _strip_cjk(text: str) -> str:
    """Remove Japanese/Korean/CJK characters (letters and punctuation) from *text*.

    All supported languages (en/fr/cs/pl) are Latin-script, so any
    Hiragana/Katakana/Kanji/Hangul characters appearing in Whisper's output
    are hallucinated garbage — a known small/medium-model failure mode —
    rather than genuine transcription. Whitespace left behind by removed
    characters is collapsed.
    """
    if not _CJK_PATTERN.search(text):
        return text
    return re.sub(r"\s+", " ", _CJK_PATTERN.sub("", text)).strip()


class Transcriber:
    """Transcribes speech audio using a local Whisper model.

    Uses mlx-whisper on macOS (Apple Silicon GPU) and faster-whisper (CPU)
    on other platforms; see :func:`_resolve_backend_kind`.

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
            Optional callable mimicking ``mlx_whisper.transcribe``'s
            signature: ``f(audio, *, path_or_hf_repo, language,
            word_timestamps, beam_size, verbose, ...) -> dict``. When
            provided, the mlx-whisper backend/contract is used regardless of
            platform or ``config.backend`` (this is the injection seam used
            by tests to avoid downloading real weights). When *None*, the
            real backend (mlx-whisper or faster-whisper) is selected per
            :func:`_resolve_backend_kind`, imported, and pre-warmed with a
            short dummy clip so the first real segment isn't delayed by
            weight loading.
        """
        self._config: TranscriptionConfig = config
        self._repo: str = _resolve_mlx_repo(config.model)
        self._fw_model_name: str = _resolve_fw_model(config.model)
        self._language: str = config.language
        self._fw_model: Any = None

        if model is not None:
            # Injected callable — always treated as the mlx-style contract.
            self._backend_kind = "mlx"
            self._model = model
            return

        self._backend_kind = _resolve_backend_kind(config)

        if self._backend_kind == "mlx":
            import mlx_whisper as _mlx  # lazy import — skipped when model injected
            log.info("Pre-warming mlx-whisper model '%s' …", self._repo)
            _mlx.transcribe(
                np.zeros(1600, dtype=np.float32),
                path_or_hf_repo=self._repo,
                language=self._language,
                initial_prompt=self._config.initial_prompt(self._language),
                word_timestamps=False,
                verbose=None,
            )
            log.info("mlx-whisper model ready.")
            self._model = _mlx.transcribe
        else:
            from faster_whisper import WhisperModel  # lazy import

            log.info("Loading faster-whisper model '%s' (CPU)…", self._fw_model_name)
            self._fw_model = WhisperModel(
                self._fw_model_name, device="cpu", compute_type="int8"
            )
            log.info("faster-whisper model ready.")

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

        if self._backend_kind == "mlx":
            segments = self._run_mlx(audio)
        else:
            segments = self._run_faster_whisper(audio)

        all_words: list[Word] = []
        texts: list[str] = []
        for seg_text, seg_words in segments:
            cleaned_text = _strip_cjk(seg_text.strip())
            if cleaned_text:
                texts.append(cleaned_text)
            for w in seg_words:
                cleaned_word = _strip_cjk(w["word"])
                if not cleaned_word.strip():
                    log.warning("Dropping hallucinated CJK word: %r", w["word"])
                    continue
                all_words.append(Word(word=cleaned_word, start=w["start"], end=w["end"]))

        text = " ".join(texts)

        tokens = [w.word.strip().lower() for w in all_words]
        loop_start = _find_loop_start(tokens)
        if loop_start is not None:
            dropped_preview = " ".join(tokens[loop_start:loop_start + 10])
            log.warning(
                "Truncating hallucinated loop at word %d/%d: %r…",
                loop_start, len(tokens), dropped_preview,
            )
            all_words = all_words[:loop_start]
            text = " ".join(w.word.strip() for w in all_words)

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

    # ------------------------------------------------------------------
    # Backend-specific inference calls
    # ------------------------------------------------------------------

    def _run_mlx(self, audio: np.ndarray) -> list[tuple[str, list[dict]]]:
        """Run mlx-whisper (or the injected mock) and normalize its output.

        Returns a list of ``(segment_text, [{"word", "start", "end"}, ...])``.
        """
        # beam_size omitted → defaults to None → greedy decoding.
        # Any non-None beam_size raises NotImplementedError in mlx-whisper 0.4.x.
        # verbose=None suppresses tqdm progress bar (verbose=False still shows it).
        raw: dict = self._model(
            audio,
            path_or_hf_repo=self._repo,
            language=self._language,
            initial_prompt=self._config.initial_prompt(self._language),
            word_timestamps=True,
            verbose=None,
        )
        return [
            (seg.get("text", ""), seg.get("words", []))
            for seg in raw.get("segments", [])
        ]

    def _run_faster_whisper(self, audio: np.ndarray) -> list[tuple[str, list[dict]]]:
        """Run faster-whisper and normalize its output to the same shape as mlx.

        Returns a list of ``(segment_text, [{"word", "start", "end"}, ...])``.
        """
        segments, _info = self._fw_model.transcribe(
            audio,
            language=self._language,
            initial_prompt=self._config.initial_prompt(self._language),
            word_timestamps=True,
            beam_size=1,
        )
        result: list[tuple[str, list[dict]]] = []
        for seg in segments:
            words = [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in (seg.words or [])
            ]
            result.append((seg.text, words))
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
