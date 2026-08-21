"""Audio capture module — FR-001.

Captures mono 16 kHz float32 audio in 100 ms chunks from a sounddevice
InputStream and exposes them through a thread-safe iterator.  Device
reconnection is handled automatically (FR reliability §7).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd

from transcriptor.config import AudioConfig

log = logging.getLogger(__name__)

# Audio constants (§6 Input Configuration)
SAMPLE_RATE: int = 16_000          # Hz
CHANNELS: int = 1                  # Mono
CHUNK_MS: int = 100                # ms per chunk
CHUNK_SAMPLES: int = SAMPLE_RATE * CHUNK_MS // 1000  # 1 600 samples

_QUEUE_MAXSIZE: int = 200          # ~20 s of audio before dropping
_RETRY_INTERVAL: float = 5.0      # seconds between reconnect attempts
_MONITOR_POLL: float = 0.5        # seconds between liveness checks


def list_input_devices() -> list[dict]:
    """Return metadata for every available audio input device."""
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append(
                {
                    "index": i,
                    "name": dev["name"],
                    "host_api": sd.query_hostapis(dev["hostapi"])["name"],
                    "default_samplerate": dev["default_samplerate"],
                }
            )
    return devices


def _resolve_device(device_id: str) -> Optional[int]:
    """Map *device_id* config string to a sounddevice index (None = system default)."""
    if device_id == "default":
        return None
    # Numeric index
    try:
        return int(device_id)
    except ValueError:
        pass
    # Partial name match (case-insensitive)
    for dev in list_input_devices():
        if device_id.lower() in dev["name"].lower():
            log.info("Resolved device %r → index %d (%s)", device_id, dev["index"], dev["name"])
            return dev["index"]
    raise ValueError(f"Audio input device not found: {device_id!r}")


class AudioCapture:
    """Continuously captures audio and delivers 100 ms mono float32 chunks.

    Usage::

        capture = AudioCapture(config.audio)
        capture.start()
        for chunk in capture.chunks():   # numpy float32 array, shape (1600,)
            process(chunk)
        capture.stop()
    """

    def __init__(self, config: AudioConfig) -> None:
        self._device_id: str = config.device_id
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._stream: Optional[sd.InputStream] = None
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the audio stream and start the monitor thread."""
        self._stop_event.clear()
        self._open_stream()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="audio-monitor", daemon=True
        )
        self._monitor_thread.start()
        log.info("Audio capture started (device=%s, %d Hz, %d ms chunks)",
                 self._device_id, SAMPLE_RATE, CHUNK_MS)

    def stop(self) -> None:
        """Signal the monitor thread and close the stream."""
        self._stop_event.set()
        self._close_stream()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        log.info("Audio capture stopped")

    @property
    def is_running(self) -> bool:
        """True when the underlying InputStream is active."""
        return self._stream is not None and self._stream.active

    def chunks(self) -> Iterator[np.ndarray]:
        """Yield captured audio chunks.

        Each chunk is a numpy float32 array of shape ``(CHUNK_SAMPLES,)``
        (i.e. 1 600 samples = 100 ms at 16 kHz).  Blocks until a chunk
        arrives or the capture is stopped.
        """
        while not self._stop_event.is_set():
            try:
                yield self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

    def get_chunk(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        """Return the next captured chunk, or ``None`` on timeout.

        Unlike :meth:`chunks`, this method does not check the internal
        stop event — the caller controls the loop lifetime.  Use this
        when the audio device may be hot-swapped without stopping the
        consuming loop.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_stream(self) -> None:
        device = _resolve_device(self._device_id)
        self._stream = sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self._sd_callback,
        )
        self._stream.start()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice on the audio thread for each 100 ms block."""
        if status:
            log.warning("Audio stream status: %s", status)
        # indata shape is (frames, channels); extract mono channel
        chunk: np.ndarray = indata[:, 0].copy()
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            log.warning("Audio queue full — dropping chunk (downstream too slow)")

    def _monitor_loop(self) -> None:
        """Periodically checks stream liveness and reconnects on failure."""
        while not self._stop_event.is_set():
            time.sleep(_MONITOR_POLL)
            if self._stop_event.is_set():
                break
            if self._stream is None or not self._stream.active:
                log.warning(
                    "Audio device lost — retrying in %.0f s", _RETRY_INTERVAL
                )
                self._close_stream()
                time.sleep(_RETRY_INTERVAL)
                if not self._stop_event.is_set():
                    try:
                        self._open_stream()
                        log.info("Audio device reconnected")
                    except Exception as exc:
                        log.error("Audio reconnection failed: %s", exc)


# ---------------------------------------------------------------------------
# Manual smoke-test: python -m transcriptor.audio
# Prints a live RMS level bar for ~10 s then exits.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from transcriptor.config import load_config
    from transcriptor.logging_setup import setup_logging

    setup_logging()

    cfg = load_config()

    print("Available input devices:")
    for d in list_input_devices():
        marker = " *" if d["name"] in (sd.query_devices(kind="input") or {}).get("name", "") else ""
        print(f"  [{d['index']:2d}] {d['name']}  ({d['host_api']}){marker}")
    print()

    capture = AudioCapture(cfg.audio)
    capture.start()

    print("Recording for 10 seconds — speak or make noise to test level meter.")
    print("Press Ctrl+C to stop early.\n")

    BAR_WIDTH = 40
    deadline = time.monotonic() + 10.0
    try:
        for chunk in capture.chunks():
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            filled = int(rms * BAR_WIDTH * 20)  # scale for typical mic levels
            bar = "#" * min(filled, BAR_WIDTH) + "-" * max(BAR_WIDTH - filled, 0)
            sys.stdout.write(f"\r[{bar}] {rms:.4f}")
            sys.stdout.flush()
            if time.monotonic() > deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        print("\nDone.")
