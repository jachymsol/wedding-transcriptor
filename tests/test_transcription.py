"""Tests for transcriptor.transcription — Transcriber logic.

All tests inject a callable MagicMock in place of the real mlx_whisper.transcribe
so that no model weights are downloaded and tests run in milliseconds.

Mock contract
-------------
mlx_whisper.transcribe(audio, *, path_or_hf_repo, language, word_timestamps,
                        beam_size, verbose, ...) returns a dict::

    {
        "text": str,
        "language": str,
        "segments": [
            {
                "text": str,
                "words": [
                    {"word": str, "start": float, "end": float, "probability": float},
                    ...
                ]
            },
            ...
        ]
    }
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from transcriptor.config import TranscriptionConfig
from transcriptor.transcription import (
    Transcriber,
    TranscriptResult,
    Word,
    _SAMPLE_RATE,
    _find_loop_start,
    _resolve_mlx_repo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_word(word: str, start: float, end: float) -> dict:
    """Return a mock mlx-whisper word dict."""
    return {"word": word, "start": start, "end": end, "probability": 1.0}


def _make_segment(text: str, words=None) -> dict:
    return {"text": text, "words": words if words is not None else []}


def _make_whisper_model(segment_texts: list[str] | None = None) -> MagicMock:
    """Return a callable mock mimicking mlx_whisper.transcribe()."""
    if segment_texts is None:
        segment_texts = []
    segments = [_make_segment(t) for t in segment_texts]
    model = MagicMock()
    model.return_value = {"text": "", "language": "en", "segments": segments}
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
# Word dataclass
# ---------------------------------------------------------------------------

class TestWord:
    def test_fields_are_accessible(self):
        w = Word(word="hello", start=0.0, end=0.5)
        assert w.word == "hello"
        assert w.start == 0.0
        assert w.end == 0.5

    def test_fields_accept_float_timestamps(self):
        w = Word(word="world", start=1.23, end=2.45)
        assert w.start == pytest.approx(1.23)
        assert w.end == pytest.approx(2.45)


# ---------------------------------------------------------------------------
# TranscriptResult dataclass
# ---------------------------------------------------------------------------

class TestTranscriptResult:
    def test_fields_are_accessible(self):
        r = TranscriptResult(text="hello", language="en", duration_s=1.5)
        assert r.text == "hello"
        assert r.language == "en"
        assert r.duration_s == 1.5

    def test_words_defaults_to_empty_list(self):
        r = TranscriptResult(text="hello", language="en", duration_s=1.5)
        assert r.words == []

    def test_words_field_accepted(self):
        words = [Word("hello", 0.0, 0.4), Word("world", 0.5, 0.9)]
        r = TranscriptResult(text="hello world", language="en", duration_s=1.0, words=words)
        assert len(r.words) == 2


# ---------------------------------------------------------------------------
# _resolve_mlx_repo
# ---------------------------------------------------------------------------

class TestResolveMLXRepo:
    def test_known_short_name_medium(self):
        assert _resolve_mlx_repo("medium") == "mlx-community/whisper-medium"

    def test_known_short_name_tiny(self):
        assert _resolve_mlx_repo("tiny") == "mlx-community/whisper-tiny"

    def test_full_repo_passes_through_unchanged(self):
        assert _resolve_mlx_repo("mlx-community/whisper-medium") == "mlx-community/whisper-medium"

    def test_full_repo_with_quantisation_suffix_passes_through(self):
        assert _resolve_mlx_repo("mlx-community/whisper-medium-4bit") == "mlx-community/whisper-medium-4bit"


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

    def test_no_prewarm_when_model_injected(self):
        """Injected callable must not be called during __init__ (no pre-warming)."""
        model = _make_whisper_model()
        config = TranscriptionConfig()
        Transcriber(config, model=model)
        model.assert_not_called()


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
    def test_model_called_once_per_transcription(self):
        t, model = _make_transcriber(["hi"])
        t.transcribe(AUDIO_1S)
        assert model.call_count == 1

    def test_language_passed_to_model(self):
        t, model = _make_transcriber(["hi"], language="pl")
        t.transcribe(AUDIO_1S)
        _, kwargs = model.call_args
        assert kwargs["language"] == "pl"

    def test_word_timestamps_enabled(self):
        """word_timestamps=True must be passed so we can do partial commits."""
        t, model = _make_transcriber(["hi"])
        t.transcribe(AUDIO_1S)
        _, kwargs = model.call_args
        assert kwargs["word_timestamps"] is True

    def test_path_or_hf_repo_passed_to_model(self):
        """path_or_hf_repo must be the resolved HF repo string."""
        t, model = _make_transcriber(["hi"], model_name="medium")
        t.transcribe(AUDIO_1S)
        _, kwargs = model.call_args
        assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-medium"

    def test_verbose_none_passed_to_model(self):
        """verbose=None must be passed to suppress tqdm output."""
        t, model = _make_transcriber(["hi"])
        t.transcribe(AUDIO_1S)
        _, kwargs = model.call_args
        assert kwargs["verbose"] is None

    def test_audio_array_passed_to_model(self):
        t, model = _make_transcriber(["hi"])
        t.transcribe(AUDIO_1S)
        args, _ = model.call_args
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
        model.return_value = {"text": "", "language": "pl", "segments": []}
        t.set_language("pl")
        t.transcribe(AUDIO_1S)
        _, kwargs = model.call_args
        assert kwargs["language"] == "pl"

    def test_set_language_multiple_times(self):
        t, _ = _make_transcriber(language="en")
        for lang in ["fr", "cs", "pl", "en"]:
            t.set_language(lang)
        assert t.language == "en"


# ---------------------------------------------------------------------------
# Word collection — words extracted from segment dicts
# ---------------------------------------------------------------------------

class TestWordCollection:
    def _make_model_with_words(self, text: str, words: list) -> MagicMock:
        seg = _make_segment(text, words=words)
        model = MagicMock()
        model.return_value = {"text": text, "language": "en", "segments": [seg]}
        return model

    def test_words_collected_from_segment(self):
        mock_words = [_make_word(" hello", 0.0, 0.4), _make_word(" world", 0.5, 0.9)]
        model = self._make_model_with_words("hello world", mock_words)
        config = TranscriptionConfig(model="medium", language="en")
        t = Transcriber(config, model=model)
        result = t.transcribe(AUDIO_1S)
        assert len(result.words) == 2

    def test_word_fields_preserved(self):
        mock_words = [_make_word(" hello", 0.1, 0.4)]
        model = self._make_model_with_words("hello", mock_words)
        config = TranscriptionConfig(model="medium", language="en")
        t = Transcriber(config, model=model)
        result = t.transcribe(AUDIO_1S)
        w = result.words[0]
        assert w.word == " hello"
        assert w.start == pytest.approx(0.1)
        assert w.end == pytest.approx(0.4)

    def test_words_collected_across_multiple_segments(self):
        w1 = _make_word(" hello", 0.0, 0.4)
        w2 = _make_word(" world", 0.5, 0.9)
        seg1 = _make_segment("hello", words=[w1])
        seg2 = _make_segment("world", words=[w2])
        model = MagicMock()
        model.return_value = {"text": "hello world", "language": "en", "segments": [seg1, seg2]}
        config = TranscriptionConfig(model="medium", language="en")
        t = Transcriber(config, model=model)
        result = t.transcribe(AUDIO_1S)
        assert len(result.words) == 2
        assert result.words[0].word == " hello"
        assert result.words[1].word == " world"

    def test_empty_words_when_segment_has_no_words(self):
        seg = _make_segment("hello", words=[])
        model = MagicMock()
        model.return_value = {"text": "hello", "language": "en", "segments": [seg]}
        config = TranscriptionConfig(model="medium", language="en")
        t = Transcriber(config, model=model)
        result = t.transcribe(AUDIO_1S)
        assert result.words == []

    def test_words_empty_when_no_segments(self):
        t, _ = _make_transcriber([])
        result = t.transcribe(AUDIO_1S)
        assert result.words == []


# ---------------------------------------------------------------------------
# transcribe() — hallucination loop truncation (end-to-end)
# ---------------------------------------------------------------------------

class TestHallucinationTruncation:
    def _make_model_with_words(self, text: str, words: list) -> MagicMock:
        seg = _make_segment(text, words=words)
        model = MagicMock()
        model.return_value = {"text": text, "language": "en", "segments": [seg]}
        return model

    def test_bigram_loop_truncated_keeping_good_prefix(self):
        good = ["Ja", "jsem", "v", "singingu"]
        loop = (["H", "instagramuawat"] * 5) + ["H"] * 4
        all_tokens = good + loop
        mock_words = [_make_word(f" {w}", i * 0.1, i * 0.1 + 0.09) for i, w in enumerate(all_tokens)]
        text = " ".join(all_tokens)
        model = self._make_model_with_words(text, mock_words)
        config = TranscriptionConfig(model="medium", language="cs")
        t = Transcriber(config, model=model)
        result = t.transcribe(AUDIO_1S)
        assert result.text == "Ja jsem v singingu"
        assert len(result.words) == len(good)
        assert [w.word.strip() for w in result.words] == good

    def test_normal_sentence_unaffected(self):
        words_list = "Ja jsem velmi rada ze jsme se dnes sesli".split()
        mock_words = [_make_word(f" {w}", i * 0.1, i * 0.1 + 0.09) for i, w in enumerate(words_list)]
        text = " ".join(words_list)
        model = self._make_model_with_words(text, mock_words)
        config = TranscriptionConfig(model="medium", language="cs")
        t = Transcriber(config, model=model)
        result = t.transcribe(AUDIO_1S)
        assert result.text == text
        assert len(result.words) == len(words_list)


# ---------------------------------------------------------------------------
# _find_loop_start — periodicity-based hallucination loop detection
# ---------------------------------------------------------------------------

class TestFindLoopStart:
    def test_no_tokens_returns_none(self):
        assert _find_loop_start([]) is None

    def test_short_normal_sentence_not_flagged(self):
        tokens = "ja jsem velmi rada ze jsme se dnes sesli".split()
        assert _find_loop_start(tokens) is None

    def test_unigram_loop_from_start_flagged(self):
        tokens = ["the"] * 8
        assert _find_loop_start(tokens) == 0

    def test_unigram_loop_after_good_prefix(self):
        tokens = "hello there my friend".split() + ["h"] * 8
        assert _find_loop_start(tokens) == 4

    def test_bigram_alternating_loop_matches_reported_case(self):
        prefix = "ja jsem v singingu".split()
        loop = (["h", "instagramuawat"] * 5) + ["h"] * 4
        tokens = prefix + loop
        assert _find_loop_start(tokens) == len(prefix)

    def test_trigram_loop_flagged(self):
        prefix = ["dobry", "den", "vsem"]
        loop = ["a", "b", "c"] * 4
        tokens = prefix + loop
        assert _find_loop_start(tokens) == len(prefix)

    def test_short_legitimate_repeat_not_flagged(self):
        tokens = "jo jo jo dobre tak zacneme".split()
        assert _find_loop_start(tokens) is None

    def test_two_occurrences_of_a_word_not_flagged(self):
        tokens = "ja ja vim ze to bylo tezke".split()
        assert _find_loop_start(tokens) is None
