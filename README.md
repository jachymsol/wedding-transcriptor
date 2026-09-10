# Wedding Transcriptor

Local speech-to-text client for live events: transcribes speech on-device and
streams it to a central translation server, for wedding ceremonies and other
events with a multilingual audience.

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [Offline operation](#offline-operation)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Development](#development)
- [Project layout](#project-layout)

---

## Requirements

**Hardware / Software**

Two supported paths, chosen automatically by OS (`transcription.backend:
auto`, see [Configuration](#configuration)):

| Platform | Transcription backend | Notes |
|---|---|---|
| macOS, Apple Silicon (M-series) | `mlx-whisper` — GPU-accelerated via Apple's MLX framework | Recommended for real-time performance |
| Windows / Linux / Intel Mac | `faster-whisper` — CPU (int8) | No GPU required; slower than the mlx path, may fall behind on fast speech with larger models |

Both paths additionally require:

- Python 3.12 or later
- A laptop with at least 8 GB RAM
- A USB audio interface or USB microphone (or the built-in microphone)
- `pip install -e ".[dev]"` (below) installs the right backend for your OS
  automatically, plus `torch` (used only for the Silero VAD model),
  `sounddevice`, `websockets`, `httpx`, `pydantic-settings`, and the other
  runtime dependencies

---

## Installation

```bash
git clone <repo-url>
cd wedding-transcriptor

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

The first launch downloads the Silero VAD and Whisper model weights
(~1.5 GB for `medium`). Subsequent launches are instant.

---

## Running

```bash
source .venv/bin/activate
transcriptor
```

Or, without installing the entry point:

```bash
python -m transcriptor.main
```

### Startup dialog

A dialog opens before the main window. Fill in:

| Field | Default | Description |
|---|---|---|
| Event ID | from `config.yaml` | Identifies this event in every transcript segment |
| Server Host | from `config.yaml` | Bare host of the translation server (e.g. `translate.example.com` or `localhost:3000`) |
| API Key | from `config.yaml` | Bearer token sent with every server request; leave blank if not required |
| Audio Device | Default | Input device for this session |
| Save as default | unchecked | Writes Event ID, Server Host, Audio Device, and API Key back to `config.yaml` |

Click **Register** to pre-register the event with the server (`POST
/admin/events/{event_id}`) without starting the session — useful for
creating the event ahead of time. Click **Start** to open the main window or
**Cancel** to exit.

### Operator interface

The main window title bar shows `Wedding Transcriptor: <event_id>`. The
window itself has these areas:

| Area | Description |
|---|---|
| Audio status | "Audio: Connected / Disconnected" and a real-time level bar |
| Language selector | Dropdown to switch source language mid-event (English / French / Czech / Polish) |
| Audio device selector | Dropdown (with a **Refresh** button) to switch input device mid-event |
| Transcript | The most recently finalized segment |
| Server status | "Server Connected" or "Offline Queue: N segments" |

Buttons:

| Button | Description |
|---|---|
| Section Break | Sends a `section_break` control message to the server |
| Pause / Resume | Toggles a `pause`/`start` control message; disables Section Break while paused |
| Restart Transcriber | Reinitialises the transcription engine without restarting the whole application; enabled once a restart handler is registered |
| End Event | Returns to the startup dialog to switch to a different event without quitting the app |

Close the window or press Ctrl+C to shut down gracefully.

---

## Offline operation

If the WebSocket connection drops, transcript segments and control messages
(section breaks, pause/resume, start/stop) are written to
`data/transcript_queue.db` (SQLite WAL mode). On the next successful connection
the client drains the queue before handling live traffic. Nothing is lost as
long as disk space is available.

The client retries the connection every 5 seconds. If WebSocket
(`/ws/ingest`) remains unavailable it falls back to HTTPS POST to `/ingest`
on the same host.

---

## Configuration

Edit `config.yaml` before the first run, or fill in the startup dialog each
time the application is launched.

```yaml
event_id: wedding-2027            # sent in every transcript event
api_key: ""                       # bearer token sent with every server request
                                   # (also used as a WS ?token= query param);
                                   # leave blank if the server doesn't require auth

server:
  host: translate.example.com     # bare host (optionally host:port); "localhost"/
                                   # "127.0.0.1" use ws/http, everything else uses wss/https

audio:
  device_id: default              # "default", a device name substring, or
                                   # a numeric device index from sounddevice

transcription:
  model: medium                   # short name (mapped per-backend, see below)
                                   # or a full backend-specific repo/path id
  language: en                    # initial source language (en/fr/cs/pl)
  backend: auto                   # "auto" (mlx-whisper on macOS, faster-whisper
                                   # elsewhere), or explicit "mlx" / "faster-whisper"
  initial_prompts:                # optional per-language Whisper priming text;
    cs: ""                        # helps lower-resource languages avoid hallucinations

stabilization:
  silence_ms: 700                 # silence gap that finalizes a segment
  stable_ms: 2000                 # unchanged-text window that finalizes a segment

vad:
  max_speech_ms: 10000            # force a segment split after this much speech
  end_overlap_ms: 1000            # already-decoded audio committed as-is, then
                                   # skipped at the start of the next window
  start_overlap_ms: 2000          # extra audio kept at the start of the next
                                   # window purely as decoding context
  overrides:                      # optional per-language overrides for the
    cs: {}                        # three VAD fields above (e.g. longer windows
                                   # for Czech)
```

**`audio.device_id`** accepts:

| Value | Meaning |
|---|---|
| `default` | System default input device |
| `"Focusrite"` | First device whose name contains that substring (case-insensitive) |
| `"2"` | Device index as reported by sounddevice |

Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list
available devices.

---

## How it works

```
Microphone / mixer
      |
  AudioCapture          100 ms PCM chunks at 16 kHz
      |
  VoiceActivityDetector  Silero VAD; accumulates speech, discards silence
      |
  Transcriber            mlx-whisper (macOS) or faster-whisper (other OS);
                          hallucination-loop and non-Western-character filtering
      |
  Stabilizer             emits a segment after 700 ms silence or 2 s stable text
      |
  ServerClient           WebSocket → HTTPS POST fallback → SQLite offline queue
      |
  Translation server
```

Long utterances (> 10 s, configurable) are split mid-speech. Each split keeps
two kinds of overlap with the next window: an "end overlap" of already-decoded
audio that is committed as-is and skipped in the next window, plus a "start
overlap" of extra audio kept purely as decoding context — so word boundaries
are preserved and no speech is lost. Both can be tuned per language (see
`vad.overrides` above), since lower-resource languages (e.g. Czech) often
benefit from longer windows.

If the operator holds an API key (`api_key` in `config.yaml` / the startup
dialog), it is sent as a bearer token on every request to the translation
server, and as a `?token=` query parameter on the WebSocket handshake.
Besides transcript segments, the client can also send **control messages**
(`start` / `stop` / `pause` / `section_break`) over the same channel, e.g.
when the operator clicks Pause or Section Break in the UI.

### Transcript segment format

Each finalized segment is sent as JSON:

```json
{
  "event_id": "wedding-2027",
  "segment_id": 42,
  "sequence_number": 42,
  "timestamp": "2027-06-15T18:23:45Z",
  "source_language": "cs",
  "text": "Mockrát děkujeme, že jste dneska přišli.",
  "final": true
}
```

`segment_id` and `sequence_number` are monotonically increasing integers. On
startup the client calls `GET /admin/events/{event_id}` to fetch the event's
`segmentCount` and resumes numbering from `segmentCount + 1`, so restarting
the app mid-event does not collide with previously sent segments. If the
event is new (server responds `404`) or the lookup fails for any reason
(server unreachable, timeout, error), numbering falls back to starting at
`1`. `final` is always `true` in the current version.

Operator actions (Section Break, Pause/Resume) are sent over the same
WebSocket connection as a separate **control message**, not as a transcript
segment:

```json
{
  "type": "control",
  "action": "section_break",
  "event_id": "wedding-2027"
}
```

`action` is one of `start`, `stop`, `pause`, or `section_break`.

---

## Development

### Run the tests

```bash
source .venv/bin/activate
pytest -v
```

367 tests covering all pipeline stages. No network or audio hardware is required;
all external dependencies are injected via fakes.

### Module overview

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic-settings models; loads `config.yaml` |
| `audio.py` | sounddevice capture loop; device enumeration |
| `vad.py` | Silero VAD state machine; speech buffering; partial-commit splits |
| `transcription.py` | mlx-whisper / faster-whisper backend selection and wrapper; hallucination-loop and non-Western-character filtering |
| `stabilization.py` | Revision tracking; segment emission rules |
| `storage.py` | SQLite offline queue (segments and control messages) |
| `server.py` | WebSocket client; HTTPS POST fallback; reconnect loop; auth; control messages |
| `http_utils.py` | Shared endpoint URL builders (`ws_url`, `http_ingest_url`, `admin_url`) |
| `ui.py` | tkinter main window |
| `startup.py` | tkinter startup dialog |
| `main.py` | Pipeline wiring; threading model; graceful shutdown |

### Threading model

| Thread | Role |
|---|---|
| Main (tkinter) | UI event loop |
| VAD (daemon) | Audio → VAD → speech queue |
| Transcription (daemon) | Speech queue → Whisper → Stabilizer → send |
| Server (daemon) | WebSocket reconnect / queue drain |
| Audio monitor (daemon) | Device reconnect loop (inside `AudioCapture`) |

---

## Project layout

```
wedding-transcriptor/
├── config.yaml
├── pyproject.toml
├── src/
│   └── transcriptor/
│       ├── audio.py
│       ├── config.py
│       ├── http_utils.py
│       ├── logging_setup.py
│       ├── main.py
│       ├── server.py
│       ├── stabilization.py
│       ├── startup.py
│       ├── storage.py
│       ├── transcription.py
│       ├── ui.py
│       └── vad.py
├── tests/
│   ├── conftest.py
│   ├── test_audio.py
│   ├── test_config.py
│   ├── test_http_utils.py
│   ├── test_server.py
│   ├── test_stabilization.py
│   ├── test_startup.py
│   ├── test_storage.py
│   ├── test_transcription.py
│   ├── test_ui.py
│   └── test_vad.py
├── data/
│   └── transcript_queue.db      # created on first run
└── logs/
    └── transcriber.log          # created on first run
```
</content>
