"""Tests for transcriptor.transcription — Transcriber logic.

All tests inject a MagicMock in place of the real WhisperModel so that
no model weights are downloaded and tests run in milliseconds.

Mock contract
-------------
faster-whisper's WhisperModel.transcribe() returns
    (Iterator[Segment], TranscriptionInfo)
where each Segment has a .text attribute.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, call, patch

from transcriptor.config import TranscriptionConfig
from transcriptor.transcription import Transcriber, TranscriptResult, _SAMPLE_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(text: str) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    return seg


def _make_whisper_model(segment_texts: list[str] | None = None) -> MagicMock:
    """Return a mock WhisperModel whose transcribe() yields the given texts."""
    if segment_texts is None:
        segment_texts = []
    segments = [_make_segment(t) for t in segment_texts]
    info = MagicMock()
    model = MagicMock()
    model.transcribe.return_value = (iter(segments), info)
    return model


def _make_transcriber(
    segment_texts: list[str] | None = None,
    language: str = "en",
    model_name: str = "medium",
) -> tuple[Transcriber, MagicMock]:
    model = _make_whisper_model(segment_texts)
    config = TranscriptionConfig(model=model_name, language=language)
    transcriber = Transcriber(config, model=model)
    return transcriber, model


AUDIO_1S = np.zeros(16_000, dtype=np.float32)   # 1 s of silence
AUDIO_3S = np.zeros(48_000, dtype=np.float32)   # 3 s of silence


# ---------------------------------------------------------------------------
# TranscriptResult dataclass
# ---------------------------------------------------------------------------

class TestTranscriptResult:
    def test_fields_are_accessible(self):
        r = TranscriptResult(text="hello", language="en", duration_s=1.5)
        assert r.text == "hello"
        assert r.language == "en"
        assert r.duration_s == 1.5


# ---------------------------------------------------------------------------
# Transcriber construction
# ---------------------------------------------------------------------------

class TestTranscriberInit:
    def test_language_set_from_config(self):
        t, _ = _make_transcriber(language="cs")
        assert t.language == "cs"

    def test_injected_model_is_used(self):
        model = _make_whisper_model()
        config = TranscriptionConfig(model="medium", language="en")
        t = Transcriber(config, model=model)
        assert t._model is model

    def test_model_factory_not_called_when_model_provided(self):
        """_load_whisper_model should never be called when a model is injected."""
        with patch("transcriptor.transcription._load_whisper_model") as mock_load:
            model = _make_whisper_model()
            config = TranscriptionConfig()
            Transcriber(config, model=model)
            mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# transcribe() — output correctness
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_returns_transcript_result(self):
        t, _ = _make_transcriber(["Hello world"])
        result = t.transcribe(AUDIO_1S)
        assert isinstance(result, TranscriptResult)

    def test_single_segment_text(self):
        t, _ = _make_transcriber(["Hello world"])
        assert t.transcribe(AUDIO_1S).text == "Hello world"

    def test_multiple_segments_joined_with_space(self):
        t, _ = _make_transcriber(["Hello", "world", "today"])
        assert t.transcribe(AUDIO_1S).text == "Hello world today"

    def test_leading_trailing_whitespace_stripped(self):
        t, _ = _make_transcriber(["  Hello  ", "  world  "])
        assert t.transcribe(AUDIO_1S).text == "Hello world"

    def test_whitespace_only_segments_are_filtered(self):
        t, _ = _make_transcriber(["Hello", "   ", "world"])
        assert t.transcribe(AUDIO_1S).text == "Hello world"

    def test_no_segments_returns_empty_string(self):
        t, _ = _make_transcriber([])
        assert t.transcribe(AUDIO_1S).text == ""

    def test_language_in_result_matches_current_language(self):
        t, _ = _make_transcriber(language="fr")
        result = t.transcribe(AUDIO_1S)
        assert result.language == "fr"

    def test_duration_is_correct(self):
        t, _ = _make_transcriber()
        result = t.transcribe(AUDIO_3S)
        assert result.duration_s == pytest.approx(3.0)

    def test_duration_uses_sample_rate_constant(self):
        audio = np.zeros(_SAMPLE_RATE * 2, dtype=np.float32)   # exactly 2 s
        t, _ = _make_transcriber()
        assert t.transcribe(audio).duration_s == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# transcribe() — model is called correctly
# ---------------------------------------------------------------------------

class TestTranscribeModelCall:
    def test_model_transcribe_called_once_per_call(self):
        t, model = _make_transcriber(["hi"])
        t.transcribe(AUDIO_1S)
        assert model.transcribe.call_count == 1

    def test_language_passed_to_model(self):
        t, model = _make_transcriber(["hi"], language="pl")
        t.transcribe(AUDIO_1S)
        _, kwargs = model.transcribe.call_args
        assert kwargs["language"] == "pl"

    def test_vad_filter_disabled(self):
        """vad_filter must be False — our pipeline already runs VAD externally."""
        t, model = _make_transcriber(["hi"])
        t.transcribe(AUDIO_1S)
        _, kwargs = model.transcribe.call_args
        assert kwargs["vad_filter"] is False

    def test_audio_array_passed_to_model(self):
        t, model = _make_transcriber(["hi"])
        t.transcribe(AUDIO_1S)
        args, _ = model.transcribe.call_args
        passed_audio = args[0]
        assert np.array_equal(passed_audio, AUDIO_1S)


# ---------------------------------------------------------------------------
# Language switching
# ---------------------------------------------------------------------------

class TestLanguageSwitching:
    def test_initial_language_from_config(self):
        t, _ = _make_transcriber(language="cs")
        assert t.language == "cs"

    def test_set_language_updates_property(self):
        t, _ = _make_transcriber(language="en")
        t.set_language("fr")
        assert t.language == "fr"

    def test_new_language_used_in_next_transcription(self):
        t, model = _make_transcriber(language="en")
        # Re-arm the mock for a second call
        model.transcribe.return_value = (iter([]), MagicMock())
        t.set_language("pl")
        t.transcribe(AUDIO_1S)
        _, kwargs = model.transcribe.call_args
        assert kwargs["language"] == "pl"

    def test_set_language_multiple_times(self):
        t, _ = _make_transcriber(language="en")
        for lang in ["fr", "cs", "pl", "en"]:
            t.set_language(lang)
        assert t.language == "en"


# ---------------------------------------------------------------------------
# _load_whisper_model — CUDA / CPU fallback logic
# ---------------------------------------------------------------------------

class TestLoadWhisperModel:
    def test_uses_cuda_when_available(self):
        from transcriptor.transcription import _load_whisper_model

        with patch("torch.cuda.is_available", return_value=True), \
             patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value = MagicMock()
            _load_whisper_model("medium")
            MockModel.assert_called_once_with("medium", device="cuda", compute_type="float16")

    def test_falls_back_to_cpu_when_cuda_unavailable(self):
        from transcriptor.transcription import _load_whisper_model

        with patch("torch.cuda.is_available", return_value=False), \
             patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value = MagicMock()
            _load_whisper_model("medium")
            MockModel.assert_called_once_with("small", device="cpu", compute_type="int8")

    def test_falls_back_to_cpu_when_cuda_init_raises(self):
        from transcriptor.transcription import _load_whisper_model

        call_count = 0

        def model_factory(name, device, compute_type):
            nonlocal call_count
            call_count += 1
            if device == "cuda":
                raise RuntimeError("out of memory")
            m = MagicMock()
            return m

        with patch("torch.cuda.is_available", return_value=True), \
             patch("faster_whisper.WhisperModel", side_effect=model_factory):
            model = _load_whisper_model("medium")

        # First call (cuda) failed; second call (cpu) succeeded
        assert call_count == 2
