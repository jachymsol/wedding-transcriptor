# Wedding Transcriptor

Local speech-to-text client for live events. Captures audio from a mixer or USB
microphone, transcribes speech on-device with Whisper, and streams finalized
transcript segments to a central translation server over WebSocket — continuing
to queue segments locally if the network is unavailable.

---

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running](#running)
- [Operator interface](#operator-interface)
- [Offline operation](#offline-operation)
- [Development](#development)
- [Project layout](#project-layout)

---

## How it works

```
Microphone / mixer
      |
  AudioCapture          100 ms PCM chunks at 16 kHz
      |
  VoiceActivityDetector  Silero VAD; accumulates speech, discards silence
      |
  Transcriber            faster-whisper (medium/CUDA → small/CPU fallback)
      |
  Stabilizer             emits a segment after 700 ms silence or 2 s stable text
      |
  ServerClient           WebSocket → HTTPS POST fallback → SQLite offline queue
      |
  Translation server
```

Long utterances (> 10 s) are split mid-speech: the last 3 s of each window
overlaps the next one so that word boundaries are preserved and no speech is lost.

---

## Requirements

**Hardware**

- A laptop with at least 8 GB RAM
- A USB audio interface or USB microphone (or the built-in microphone)
- A CUDA-capable GPU for real-time performance with the `medium` model;
  `small`/CPU works but may fall behind on fast speech

**Software**

- Python 3.12 or later
- [PyTorch](https://pytorch.org/get-started/locally/) matching your CUDA version
  (install separately before the steps below; the `torch` wheel bundled with pip
  is CPU-only)

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

## Configuration

Edit `config.yaml` before the first run, or fill in the startup dialog each
time the application is launched.

```yaml
event_id: wedding-2027            # sent in every transcript event

server:
  websocket_url: wss://translate.example.com/ws

audio:
  device_id: default              # "default", a device name substring, or
                                  # a numeric device index from sounddevice

transcription:
  model: medium                   # faster-whisper model size
  language: en                    # initial source language (en/fr/cs/pl)

stabilization:
  silence_ms: 700                 # silence gap that finalizes a segment
  stable_ms: 2000                 # unchanged-text window that finalizes a segment

vad:
  max_speech_ms: 10000            # force a segment split after this much speech
  overlap_ms: 3000                # overlap kept between split windows
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
| Server URL | from `config.yaml` | WebSocket URL of the translation server |
| Audio Device | Default | Input device for this session |
| Save as default | unchecked | Writes the three fields back to `config.yaml` |

Click **Start** to open the main window or **Cancel** to exit.

---

## Operator interface

The main window has four areas:

| Area | Description |
|---|---|
| Audio status | "Audio: Connected / Disconnected" and a real-time level bar |
| Language selector | Dropdown to switch source language mid-event (English / French / Czech / Polish) |
| Transcript | The most recently finalized segment |
| Server status | "Server Connected" or "Offline Queue: N segments" |

A **Restart** button appears if Whisper raises an unhandled exception. Clicking
it reinitialises the transcription engine without restarting the whole
application.

Close the window or press Ctrl+C to shut down gracefully.

---

## Offline operation

If the WebSocket connection drops, segments are written to
`data/transcript_queue.db` (SQLite WAL mode). On the next successful connection
the client drains the queue before handling live traffic. No segments are lost as
long as disk space is available.

The client retries the connection every 5 seconds. If WebSocket remains
unavailable it falls back to HTTPS POST to the same host and path.

---

## Transcript segment format

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

`segment_id` and `sequence_number` are monotonically increasing integers that
reset when the application restarts. `final` is always `true` in the current
version.

---

## Development

### Run the tests

```bash
source .venv/bin/activate
pytest -v
```

293 tests covering all pipeline stages. No network or audio hardware is required;
all external dependencies are injected via fakes.

### Module overview

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic-settings models; loads `config.yaml` |
| `audio.py` | sounddevice capture loop; device enumeration |
| `vad.py` | Silero VAD state machine; speech buffering; partial-commit splits |
| `transcription.py` | faster-whisper wrapper; CUDA/CPU fallback; word timestamps |
| `stabilization.py` | Revision tracking; segment emission rules |
| `storage.py` | SQLite offline queue |
| `server.py` | WebSocket client; HTTPS POST fallback; reconnect loop |
| `ui.py` | tkinter main window |
| `startup.py` | tkinter startup dialog |
| `main.py` | Pipeline wiring; threading model; graceful shutdown |

### Threading model

| Thread | Role |
|---|---|
| Main (tkinter) | UI event loop |
| Pipeline (daemon) | Audio → VAD → Whisper → Stabilizer → send |
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
