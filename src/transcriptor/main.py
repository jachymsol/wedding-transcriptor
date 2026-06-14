"""Application entry point — Step 9.

Wires the full pipeline::

    AudioCapture.chunks()
        ↓  (pipeline thread)
    VoiceActivityDetector.process_chunk()
        ↓  SpeechSegment when speech ends
    Transcriber.transcribe()
        ↓  TranscriptResult
    Stabilizer.update(is_final=True)
        ↓  TranscriptSegment when stable / final
    ServerClient.send()
        ↓  WebSocket → HTTPS POST → SegmentQueue

Threading model
---------------
* **Main thread** — tkinter event loop (``AppUI.run()``).
* **Pipeline thread** — audio → VAD → Whisper → Stabilizer → send (daemon).
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
import signal
import threading
from typing import Optional

import numpy as np

from transcriptor.audio import AudioCapture
from transcriptor.config import AppConfig, load_config
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
        self._vad = VoiceActivityDetector(
            max_speech_ms=self._config.vad.max_speech_ms,
            overlap_ms=self._config.vad.overlap_ms,
        )
        # Pre-compute overlap in seconds for partial-commit boundary calculation.
        self._overlap_s: float = self._config.vad.overlap_ms / 1000.0

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
            self._ui = AppUI(title="Wedding Transcriptor")

        # Pipeline control
        self._stop_event = threading.Event()
        self._pipeline_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all components and enter the tkinter main loop (blocking)."""
        self._wire_ui_callbacks()
        self._audio.start()
        self._server.start()

        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop,
            name="pipeline",
            daemon=True,
        )
        self._pipeline_thread.start()

        # Handle Ctrl+C gracefully (signal fires on main thread)
        signal.signal(signal.SIGINT, self._on_sigint)

        log.info("Entering UI main loop")
        self._ui.run()  # blocks until window is closed

        self._shutdown()

    def _shutdown(self) -> None:
        log.info("Shutting down …")
        self._stop_event.set()
        self._audio.stop()
        self._server.stop()
        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=5.0)
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

    # ------------------------------------------------------------------
    # Partial-commit helper
    # ------------------------------------------------------------------

    def _extract_committed_text(self, result: TranscriptResult) -> str:
        """Return the portion of *result* that lies before the overlap region.

        For a partial segment (``is_final=False``) the last ``overlap_ms`` of
        audio is kept in the VAD buffer for the next window.  We must not send
        words that fall inside that overlap — they will be re-transcribed with
        better context next time.

        Returns an empty string when no words fall before the commit boundary
        (e.g. the speaker started right at the end of the window).
        """
        commit_boundary_s = result.duration_s - self._overlap_s
        if commit_boundary_s <= 0 or not result.words:
            return ""
        committed = [w for w in result.words if w.end <= commit_boundary_s]
        if not committed:
            return ""
        return " ".join(w.word.strip() for w in committed).strip()

    # ------------------------------------------------------------------
    # Pipeline loop (runs on the pipeline thread)
    # ------------------------------------------------------------------

    def _pipeline_loop(self) -> None:
        log.info("Pipeline thread started")
        chunk_count = 0
        transcriber_ok = True

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

            # ── Transcription ─────────────────────────────────────────
            try:
                with self._transcriber_lock:
                    result = self._transcriber.transcribe(speech_seg.audio)
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
            if speech_seg.is_final:
                # Normal path: VAD detected silence → finalize immediately.
                final_seg: Optional[TranscriptSegment] = self._stabilizer.update(
                    result, is_final=True
                )
            else:
                # Partial path: max_speech_ms triggered a mid-speech emit.
                # Commit only the words before the overlap region; words
                # inside the overlap will be re-transcribed next window.
                committed_text = self._extract_committed_text(result)
                if not committed_text:
                    log.debug("Partial segment: no words before commit boundary; skipping")
                    continue
                committed_result = TranscriptResult(
                    text=committed_text,
                    language=result.language,
                    duration_s=result.duration_s - self._overlap_s,
                )
                final_seg = self._stabilizer.update(committed_result, is_final=True)

            if final_seg is not None:
                log.info(
                    "Segment %d finalized: %r", final_seg.segment_id, final_seg.text
                )
                self._ui.update_transcript(final_seg.text)
                self._server.send(final_seg)

        # Flush any trailing speech on shutdown
        trailing = self._vad.flush()
        if trailing is not None and not self._stop_event.is_set():
            try:
                with self._transcriber_lock:
                    result = self._transcriber.transcribe(trailing.audio)
                if result.text:
                    final_seg = self._stabilizer.update(result, is_final=True)
                    if final_seg is not None:
                        self._server.send(final_seg)
            except Exception as exc:
                log.error("Flush transcription error: %s", exc)

        log.info("Pipeline thread exiting")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    from transcriptor.startup import StartupDialog  # deferred: avoids tkinter at import time

    dialog = StartupDialog(load_config())
    config = dialog.run()          # blocks until Start or Cancel
    if config is None:
        log.info("Startup cancelled by operator — exiting")
        return

    Application(config=config).run()


if __name__ == "__main__":
    main()
