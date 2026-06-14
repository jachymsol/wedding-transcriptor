"""Tests for transcriptor.audio — device resolution, callback, and queue behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from transcriptor.audio import (
    AudioCapture,
    _resolve_device,
    CHUNK_MS,
    CHUNK_SAMPLES,
    CHANNELS,
    SAMPLE_RATE,
)
from transcriptor.config import AudioConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_sample_rate(self):
        assert SAMPLE_RATE == 16_000

    def test_channels(self):
        assert CHANNELS == 1

    def test_chunk_ms(self):
        assert CHUNK_MS == 100

    def test_chunk_samples(self):
        assert CHUNK_SAMPLES == 1600  # 16_000 * 100 // 1000


# ---------------------------------------------------------------------------
# _resolve_device — device ID resolution logic
# ---------------------------------------------------------------------------

FAKE_DEVICES = [
    {"index": 0, "name": "Built-in Microphone", "host_api": "Core Audio", "default_samplerate": 44100},
    {"index": 1, "name": "USB Audio Interface", "host_api": "Core Audio", "default_samplerate": 48000},
    {"index": 2, "name": "USB Mixer Pro", "host_api": "Core Audio", "default_samplerate": 48000},
]


@pytest.fixture(autouse=False)
def mock_list_devices(monkeypatch):
    monkeypatch.setattr("transcriptor.audio.list_input_devices", lambda: FAKE_DEVICES)


class TestResolveDevice:
    def test_default_returns_none(self, mock_list_devices):
        assert _resolve_device("default") is None

    def test_numeric_string_returns_int(self, mock_list_devices):
        assert _resolve_device("0") == 0
        assert _resolve_device("2") == 2

    def test_partial_name_match_case_insensitive(self, mock_list_devices):
        # "usb audio" matches "USB Audio Interface" at index 1
        assert _resolve_device("usb audio") == 1

    def test_partial_name_first_match_wins(self, mock_list_devices):
        # "usb" matches both index 1 and 2 — first match (index 1) is returned
        assert _resolve_device("usb") == 1

    def test_partial_name_match_substring(self, mock_list_devices):
        assert _resolve_device("mixer") == 2

    def test_unknown_name_raises_value_error(self, mock_list_devices):
        with pytest.raises(ValueError, match="Audio input device not found"):
            _resolve_device("nonexistent_xyz")

    def test_unknown_name_includes_name_in_error(self, mock_list_devices):
        with pytest.raises(ValueError, match="nonexistent_xyz"):
            _resolve_device("nonexistent_xyz")


# ---------------------------------------------------------------------------
# AudioCapture._sd_callback — queue ingestion logic
# ---------------------------------------------------------------------------

class TestSdCallback:
    def _make_capture(self) -> AudioCapture:
        return AudioCapture(AudioConfig(device_id="default"))

    def test_callback_extracts_channel_zero(self):
        """Only channel 0 is kept; other channels are discarded."""
        capture = self._make_capture()
        # Simulate 2-channel input
        indata = np.zeros((CHUNK_SAMPLES, 2), dtype=np.float32)
        indata[:, 0] = 0.5   # channel 0 — should be kept
        indata[:, 1] = 0.9   # channel 1 — should be discarded

        capture._sd_callback(indata, CHUNK_SAMPLES, None, None)

        chunk = capture._queue.get_nowait()
        assert chunk.shape == (CHUNK_SAMPLES,)
        assert np.allclose(chunk, 0.5)

    def test_callback_produces_copy(self):
        """Chunk in the queue is a copy, not a view into the input buffer."""
        capture = self._make_capture()
        indata = np.ones((CHUNK_SAMPLES, 1), dtype=np.float32)
        capture._sd_callback(indata, CHUNK_SAMPLES, None, None)

        chunk = capture._queue.get_nowait()
        indata[:] = 0.0   # mutate the original buffer
        assert np.allclose(chunk, 1.0), "chunk should not be affected by mutating indata"

    def test_callback_drops_on_full_queue_without_raising(self):
        """When the queue is full, the callback discards the chunk silently."""
        capture = self._make_capture()
        # Fill the queue to capacity
        filler = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        for _ in range(capture._queue.maxsize):
            capture._queue.put_nowait(filler)

        indata = np.ones((CHUNK_SAMPLES, 1), dtype=np.float32)
        capture._sd_callback(indata, CHUNK_SAMPLES, None, None)  # must not raise

        # Queue is still at capacity; the new chunk was dropped
        assert capture._queue.full()

    def test_callback_with_no_status_flags(self):
        """Passing None as status (no error flags) does not raise."""
        capture = self._make_capture()
        indata = np.zeros((CHUNK_SAMPLES, 1), dtype=np.float32)
        capture._sd_callback(indata, CHUNK_SAMPLES, None, None)  # no error


# ---------------------------------------------------------------------------
# AudioCapture.chunks() — iterator behaviour
# ---------------------------------------------------------------------------

class TestChunksIterator:
    def test_yields_item_placed_in_queue(self):
        """chunks() returns items that were put directly into the internal queue."""
        capture = AudioCapture(AudioConfig(device_id="default"))
        expected = np.ones(CHUNK_SAMPLES, dtype=np.float32) * 0.42
        capture._queue.put_nowait(expected.copy())

        gen = capture.chunks()
        chunk = next(gen)
        capture._stop_event.set()  # stop the generator after we've consumed the item

        assert np.allclose(chunk, expected)

    def test_stops_when_stop_event_is_set(self):
        """chunks() exits cleanly once _stop_event is set."""
        capture = AudioCapture(AudioConfig(device_id="default"))
        capture._stop_event.set()  # signal stop before iterating
        result = list(capture.chunks())
        assert result == []


# ---------------------------------------------------------------------------
# AudioCapture._close_stream — defensive cleanup
# ---------------------------------------------------------------------------

class TestCloseStream:
    def test_close_with_none_stream_is_safe(self):
        """_close_stream() is a no-op when _stream is already None."""
        capture = AudioCapture(AudioConfig(device_id="default"))
        capture._stream = None
        capture._close_stream()   # must not raise

    def test_close_sets_stream_to_none(self):
        """After _close_stream(), _stream is None regardless of whether close() raises."""
        capture = AudioCapture(AudioConfig(device_id="default"))
        fake_stream = MagicMock()
        fake_stream.stop.side_effect = OSError("device gone")  # simulate error
        capture._stream = fake_stream

        capture._close_stream()  # must not propagate the OSError

        assert capture._stream is None
