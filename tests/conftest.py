"""Shared test fixtures and helpers."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from transcriptor.vad import VoiceActivityDetector

# ---------------------------------------------------------------------------
# Reusable audio arrays
# ---------------------------------------------------------------------------

CHUNK_SAMPLES = 1600  # 100 ms at 16 kHz


@pytest.fixture
def silence_chunk() -> np.ndarray:
    """100 ms of silence."""
    return np.zeros(CHUNK_SAMPLES, dtype=np.float32)


@pytest.fixture
def speech_chunk() -> np.ndarray:
    """100 ms of synthetic non-zero audio that the fake model will treat as speech."""
    return np.ones(CHUNK_SAMPLES, dtype=np.float32) * 0.5


# ---------------------------------------------------------------------------
# Fake VAD model factory
# ---------------------------------------------------------------------------

def make_fake_model(prob: float = 0.0) -> MagicMock:
    """Return a MagicMock that mimics the Silero VAD model.

    The mock is callable and its return value exposes .item() → prob.
    Use ``model.return_value.item.return_value = x`` to change the
    probability returned for every subsequent window.
    """
    model = MagicMock()
    model.reset_states = MagicMock()
    model.return_value.item.return_value = prob
    return model


def make_vad(speech_prob: float = 0.0, **kwargs) -> tuple[VoiceActivityDetector, MagicMock]:
    """Construct a VoiceActivityDetector backed by a fake model.

    The model initially returns *speech_prob* for every VAD window.
    Change ``model.return_value.item.return_value`` between phases to
    simulate speech / silence transitions.

    Returns ``(vad, fake_model)`` so tests can reconfigure the model.
    """
    model = make_fake_model(speech_prob)
    vad = VoiceActivityDetector(model=model, **kwargs)
    return vad, model
