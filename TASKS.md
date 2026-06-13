# List of Tasks

For the purposes of easy reviewing, the implementation shall be split into following tasks:

**Step 1 — Project scaffolding & configuration**
- Directory structure (`src/`, `data/`, `logs/`, `config/`)
-  `pyproject.toml` with all dependencies pinned
-  `config.yaml` and Pydantic-settings models
- Logging setup ( `logs/transcriber.log`, INFO/WARNING/ERROR)

**Step 2 — Audio capture**
- `sounddevice`-based capture module
- 16 kHz mono PCM, 100 ms chunks
- Device enumeration and selection
- Continuous streaming loop

**Step 3 — Voice Activity Detection**
- Silero VAD integration
- Speech/silence detection on audio chunks
- Speech buffer accumulation

**Step 4 — Whisper transcription**
- `faster-whisper` integration
- CUDA (`medium/float16`) → CPU (`small`) fallback
- Transcription from accumulated speech buffers

**Step 5 — Transcript stabilization & segment generation**
- Stabilization buffer tracking revisions
- Finalization rules: 700 ms pause OR 2 s stable
- `TranscriptSegment` JSON event generation per schema in §4

**Step 6 — Local storage & offline queue**
- SQLite setup at  `data/transcript_queue.db`
- `pending_segments` table
- Enqueue/dequeue operations

**Step 7 — Server communication**
- WebSocket client with auto-reconnect
- HTTPS POST fallback
- Retry every 5 s; drain queue on reconnect

**Step 8 — Desktop UI**
- using tkinter
- Audio status + level indicator
- Language selector dropdown (EN/FR/CS/PL)
- Live transcript display
- Connection status (connected / offline queue count)

**Step 9 — Application wiring & reliability**
- Main entry point, wires all pipeline stages
- Error handlers: Whisper crash (display + manual restart), audio device disconnect (retry)
- Graceful shutdown

## Step 1 Notes

The project structure should be as follows:

```
wedding-transcriptor/
├── src/
│   └── transcriptor/
│       ├── __init__.py
│       ├── config.py          # Pydantic-settings models
│       ├── logging_setup.py   # Logging configuration
│       ├── audio.py           # Audio capture (Step 2)
│       ├── vad.py             # VAD (Step 3)
│       ├── transcription.py   # Whisper (Step 4)
│       ├── stabilization.py   # Stabilization (Step 5)
│       ├── storage.py         # SQLite queue (Step 6)
│       ├── server.py          # WebSocket/HTTP (Step 7)
│       ├── ui.py              # tkinter UI (Step 8)
```
