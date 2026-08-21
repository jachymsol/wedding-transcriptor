"""Application entry point — Step 9.

Wires the full pipeline::

    AudioCapture.chunks()
        ↓  (VAD thread)
    VoiceActivityDetector.process_chunk()
        ↓  SpeechSegment → speech_queue
    Transcriber.transcribe()          (transcription thread)
        ↓  TranscriptResult
    Stabilizer.update(is_final=True)
        ↓  TranscriptSegment when stable / final
    ServerClient.send()
        ↓  WebSocket → HTTPS POST → SegmentQueue

Threading model
---------------
* **Main thread** — tkinter event loop (``AppUI.run()``).
* **VAD thread** — audio chunks → RMS meter → Silero VAD → speech_queue (daemon).
* **Transcription thread** — speech_queue → Whisper → Stabilizer → send (daemon).
* **Server thread** — WebSocket reconnect / queue drain (inside
  :class:`~transcriptor.server.ServerClient`, daemon).
* **Audio monitor thread** — device reconnect loop (inside
  :class:`~transcriptor.audio.AudioCapture`, daemon).

Language changes are applied to the :class:`~transcriptor.transcription.Transcriber`
immediately from the UI callback; the next transcription picks up the new code.

Restart
-------
If the transcription engine raises an unhandled exception, the UI shows an
error and enables the Restart button.  Clicking it re-initialises the
:class:`~transcriptor.transcription.Transcriber` so the pipeline can
continue without restarting the whole application.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import threading
import time
from typing import Optional

import numpy as np

from transcriptor.audio import AudioCapture, list_input_devices
from transcriptor.config import AppConfig, AudioConfig, load_config
from transcriptor.logging_setup import setup_logging
from transcriptor.server import ServerClient
from transcriptor.stabilization import Stabilizer, TranscriptSegment
from transcriptor.storage import SegmentQueue
from transcriptor.transcription import Transcriber, TranscriptResult
from transcriptor.ui import AppUI
from transcriptor.vad import VoiceActivityDetector

log = logging.getLogger(__name__)

# RMS scaling: typical mic input is -40 dBFS; multiply to fill the meter.
_RMS_SCALE: float = 10.0
# How often (in chunks) to push UI status updates.
_STATUS_EVERY_N_CHUNKS: int = 10  # ≈ 1 s
# Capacity of the inter-thread speech segment queue.
# VAD (~1 ms/window) produces far faster than Whisper (0.6–4 s/segment) consumes;
# a small buffer lets bursts absorb without blocking audio capture.
_SPEECH_QUEUE_SIZE: int = 10
# Set TRANSCRIPTOR_LATENCY_LOG=1 to emit per-segment latency measurements.
# Logs total latency (speech onset → transmit) and tail latency (emit → transmit).
_LATENCY_LOGGING: bool = os.getenv("TRANSCRIPTOR_LATENCY_LOG", "") == "1"


class Application:
    """Top-level object that owns all pipeline components.

    Parameters
    ----------
    config:
        Pre-built :class:`~transcriptor.config.AppConfig`.  When ``None``
        (the default) the config is loaded from ``config.yaml`` via
        :func:`~transcriptor.config.load_config`.  Pass an explicit config
        when the startup dialog has applied operator overrides.
    ui_factory:
        Optional callable ``() → AppUI`` for dependency injection in tests.
        Defaults to the real :class:`~transcriptor.ui.AppUI`.
    """

    def __init__(self, *, config: Optional[AppConfig] = None, ui_factory=None) -> None:
        self._config = config if config is not None else load_config()
        log.info("Starting Wedding Transcriptor — event_id=%s", self._config.event_id)

        # Storage (persistent offline queue)
        self._queue = SegmentQueue()

        # Audio capture
        self._audio = AudioCapture(self._config.audio)

        # VAD (loads Silero on construction)
        _total_overlap_ms = (
            self._config.vad.end_overlap_ms + self._config.vad.start_overlap_ms
        )
        self._vad = VoiceActivityDetector(
            max_speech_ms=self._config.vad.max_speech_ms,
            overlap_ms=_total_overlap_ms,
        )
        # start_overlap_ms: words at the START of the next window that were
        # already committed by the current window.  They are skipped in the
        # next window to avoid double-sending.
        # end_overlap_ms: the LAST end_overlap_ms of each partial window are
        # NOT committed from that window; instead they are committed by the
        # *next* window, where they benefit from more right-side audio context.
        # The total VAD buffer = start + end, but each half plays a different role.
        self._end_overlap_s: float = self._config.vad.end_overlap_ms / 1000.0
        self._start_overlap_s: float = self._config.vad.start_overlap_ms / 1000.0
        # Seconds to skip at the START of the next segment (= start_overlap_s).
        # Set after every partial emit; reset to 0.0 after each segment.
        self._skip_overlap_s: float = 0.0

        # Transcriber (loads Whisper on construction)
        self._transcriber: Transcriber = Transcriber(self._config.transcription)
        self._transcriber_lock = threading.Lock()

        # Stabilizer
        self._stabilizer = Stabilizer(self._config)

        # Server client
        self._server = ServerClient(self._config, self._queue)

        # UI
        if ui_factory is not None:
            self._ui: AppUI = ui_factory()
        else:
            try:
                devices = list_input_devices()
            except Exception:
                devices = []
            self._ui = AppUI(
                title="Wedding Transcriptor",
                event_id=self._config.event_id,
                devices=devices,
                device_id=self._config.audio.device_id,
            )

        # Pipeline control
        self._stop_event = threading.Event()
        self._speech_queue: queue.Queue = queue.Queue(maxsize=_SPEECH_QUEUE_SIZE)
        self._vad_thread: Optional[threading.Thread] = None
        self._transcription_thread: Optional[threading.Thread] = None
        self._restart_requested: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Start all components and enter the tkinter main loop (blocking).

        Returns ``True`` if the operator clicked "End Event" (caller should
        show the startup dialog again), ``False`` if the window was closed
        normally.
        """
        self._wire_ui_callbacks()
        self._audio.start()
        self._server.start()

        self._vad_thread = threading.Thread(
            target=self._vad_loop,
            name="vad",
            daemon=True,
        )
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop,
            name="transcription",
            daemon=True,
        )
        self._vad_thread.start()
        self._transcription_thread.start()

        # Handle Ctrl+C gracefully (signal fires on main thread)
        signal.signal(signal.SIGINT, self._on_sigint)

        log.info("Entering UI main loop")
        self._ui.run()  # blocks until window is closed

        self._shutdown()
        return self._restart_requested

    def _shutdown(self) -> None:
        log.info("Shutting down …")
        self._stop_event.set()
        self._audio.stop()
        if self._vad_thread is not None:
            self._vad_thread.join(timeout=5.0)
        if self._transcription_thread is not None:
            self._transcription_thread.join(timeout=10.0)
        self._server.send_control("stop")
        self._server.stop()
        self._queue.close()
        log.info("Shutdown complete")

    def _on_sigint(self, _sig, _frame) -> None:
        log.info("SIGINT received — requesting stop")
        self._ui.stop()  # triggers mainloop exit → _shutdown() called

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------

    def _wire_ui_callbacks(self) -> None:
        self._ui.set_on_language_change(self._on_language_change)
        self._ui.set_on_restart(self._on_restart_transcriber)
        self._ui.set_on_device_change(self._on_device_change)
        self._ui.set_on_end_event(self._on_end_event)
        # Close button → shut down gracefully
        try:
            self._ui._root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        except Exception:
            pass

    def _on_window_close(self) -> None:
        log.info("Window close requested")
        self._ui.stop()

    def _on_language_change(self, code: str) -> None:
        with self._transcriber_lock:
            self._transcriber.set_language(code)
        log.info("Language set to: %s", code)

    def _on_restart_transcriber(self) -> None:
        log.info("Restarting transcriber …")
        new_transcriber = Transcriber(self._config.transcription)
        with self._transcriber_lock:
            self._transcriber = new_transcriber
        self._ui.update_transcript("Transcriber restarted.")
        log.info("Transcriber restart complete")

    def _on_device_change(self, device_id: str) -> None:
        log.info("Switching audio device to: %s", device_id)
        self._audio.stop()
        self._audio = AudioCapture(AudioConfig(device_id=device_id))
        self._audio.start()
        log.info("Audio device switched to: %s", device_id)

    def _on_end_event(self) -> None:
        log.info("End Event requested by operator — returning to startup")
        self._restart_requested = True
        self._ui.stop()  # exits mainloop → _shutdown() → run() returns True

    # ------------------------------------------------------------------
    # Overlap-commit helpers
    # ------------------------------------------------------------------

    def _strip_leading_overlap(
        self, result: TranscriptResult
    ) -> Optional[TranscriptResult]:
        """Remove the start_overlap prefix already committed by the previous partial.

        After a partial emit, the current window commits words up to
        ``duration_s − end_overlap_s``.  Those words reappear at
        ``[0 … start_overlap_s]`` in the next window's audio.  Skipping them
        here prevents double-sending.  The ``end_overlap_ms`` region that
        follows (``[start_overlap_s … total_overlap_s]``) is new territory
        and is committed by this segment.

        Resets ``_skip_overlap_s`` to zero unconditionally; call once per
        segment.

        Returns ``None`` when every word falls inside the skipped prefix
        (i.e. nothing new to commit).
        """
        skip_s = self._skip_overlap_s
        self._skip_overlap_s = 0.0
        if skip_s <= 0.0:
            return result
        kept = [w for w in result.words if w.end > skip_s]
        if not kept:
            log.debug("Overlap strip: no words remain after skipping %.2f s", skip_s)
            return None
        text = " ".join(w.word.strip() for w in kept).strip()
        if not text:
            return None
        return TranscriptResult(
            text=text,
            language=result.language,
            duration_s=result.duration_s,
            words=kept,
        )

    def _commit_partial(self, result: TranscriptResult) -> str:
        """Commit the words before the end_overlap boundary and schedule a skip.

        The window is split into three regions::

            [0 … commit_boundary]   committed now (start_overlap already stripped)
            [commit_boundary … end] end_overlap region — NOT committed here;
                                    committed by the *next* window after that
                                    window's start_overlap prefix is stripped

        ``commit_boundary = duration_s − end_overlap_s``

        The end_overlap words are left to the next window so they can be
        transcribed with more right-side audio context.  To avoid re-sending
        the committed words, ``_skip_overlap_s`` is set to ``start_overlap_s``
        so :meth:`_strip_leading_overlap` discards them from the next segment.

        Returns an empty string when no words fall before the commit boundary.
        """
        commit_boundary_s = result.duration_s - self._end_overlap_s
        if commit_boundary_s <= 0 or not result.words:
            return ""
        committed = [w for w in result.words if w.end <= commit_boundary_s]
        if not committed:
            return ""
        self._skip_overlap_s = self._start_overlap_s
        return " ".join(w.word.strip() for w in committed).strip()

    # ------------------------------------------------------------------
    # VAD loop (runs on the VAD thread)
    # ------------------------------------------------------------------

    def _vad_loop(self) -> None:
        log.info("VAD thread started")
        chunk_count = 0

        for chunk in self._audio.chunks():
            if self._stop_event.is_set():
                break

            chunk_count += 1

            # ── Audio level meter ─────────────────────────────────────
            rms = float(np.sqrt(np.mean(chunk ** 2))) * _RMS_SCALE
            self._ui.update_audio_level(min(rms, 1.0))

            # ── Audio connection indicator (periodic) ─────────────────
            if chunk_count % _STATUS_EVERY_N_CHUNKS == 0:
                self._ui.update_audio_status(self._audio.is_running)
                self._ui.update_server_status(
                    self._server.is_connected,
                    self._queue.count(),
                )

            # ── VAD ───────────────────────────────────────────────────
            speech_seg = self._vad.process_chunk(chunk)
            if speech_seg is None:
                continue

            try:
                self._speech_queue.put_nowait(speech_seg)
            except queue.Full:
                log.warning(
                    "Speech queue full — dropping segment (transcription too slow)"
                )

        # Flush any trailing speech before signalling transcription thread
        trailing = self._vad.flush()
        if trailing is not None:
            try:
                self._speech_queue.put(trailing, timeout=2.0)
            except queue.Full:
                log.warning("Speech queue full during flush — trailing segment dropped")

        # Poison pill: unblock transcription thread so it can exit
        self._speech_queue.put(None)
        log.info("VAD thread exiting")

    # ------------------------------------------------------------------
    # Transcription loop (runs on the transcription thread)
    # ------------------------------------------------------------------

    def _transcription_loop(self) -> None:
        log.info("Transcription thread started")
        transcriber_ok = True

        while True:
            seg = self._speech_queue.get()
            if seg is None:
                break  # poison pill — VAD thread has finished

            # ── Transcription ─────────────────────────────────────────
            try:
                with self._transcriber_lock:
                    result = self._transcriber.transcribe(seg.audio)
                if not transcriber_ok:
                    transcriber_ok = True
                    log.info("Transcriber recovered")
            except Exception as exc:
                log.error("Transcriber error: %s", exc, exc_info=True)
                if transcriber_ok:
                    transcriber_ok = False
                    self._ui.update_transcript("[Transcription error — click Restart]")
                continue

            if not result.text:
                continue

            # ── Stabilization & send ──────────────────────────────────
            if seg.is_final:
                # Normal path: VAD detected silence → finalize immediately.
                # Strip any overlap words already committed by the previous
                # partial (they had better decoder context at end-of-window).
                stripped = self._strip_leading_overlap(result)
                if stripped is None:
                    log.debug(
                        "Final segment: all words covered by prior overlap; skipping"
                    )
                    continue
                final_seg: Optional[TranscriptSegment] = self._stabilizer.update(
                    stripped, is_final=True
                )
            else:
                # Partial path: max_speech_ms triggered a mid-speech emit.
                # 1. Strip words from the previous partial's overlap (those
                #    were committed with better end-of-window context).
                # 2. Commit ALL remaining words — including the NEW overlap
                #    region — because they sit at the end of this window
                #    where the decoder has the most accumulated context.
                # 3. Schedule a prefix-skip for the next segment so those
                #    overlap words are not re-committed.
                stripped = self._strip_leading_overlap(result)
                if stripped is None:
                    log.debug("Partial segment: all words in prior overlap; skipping")
                    continue
                committed_text = self._commit_partial(stripped)
                if not committed_text:
                    continue
                committed_result = TranscriptResult(
                    text=committed_text,
                    language=result.language,
                    duration_s=result.duration_s,
                )
                final_seg = self._stabilizer.update(committed_result, is_final=True)

            if final_seg is not None:
                log.info(
                    "Segment %d finalized: %r", final_seg.segment_id, final_seg.text
                )
                self._ui.update_transcript(final_seg.text)
                self._server.send(final_seg)
                if _LATENCY_LOGGING and seg.emit_mono > 0:
                    _transmit_mono = time.monotonic()
                    log.info(
                        "LATENCY seg=%d total=%.3fs tail=%.3fs"
                        " speech_dur=%.3fs is_final=%s",
                        final_seg.segment_id,
                        _transmit_mono - seg.speech_start_mono,
                        _transmit_mono - seg.emit_mono,
                        seg.emit_mono - seg.speech_start_mono,
                        seg.is_final,
                    )

        log.info("Transcription thread exiting")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    from transcriptor.startup import StartupDialog  # deferred: avoids tkinter at import time

    while True:
        dialog = StartupDialog(load_config())
        config = dialog.run()          # blocks until Start or Cancel
        if config is None:
            log.info("Startup cancelled by operator — exiting")
            return

        restart = Application(config=config).run()
        if not restart:
            break


if __name__ == "__main__":
    main()
