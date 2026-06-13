# Wedding Live Translation - Local Transcription Client

## 1. Overview

### Purpose

The Local Transcription Client captures live speech audio during an event, performs local speech-to-text transcription using Whisper, and streams finalized transcript segments to a central translation server.

The application is intended for small-scale live events such as weddings and family gatherings where speakers may alternate between different languages.

### Goals

* Reliable operation during live events
* Local speech recognition (works even with internet interruptions)
* Language selection and switching manually during runtime
* Minimal latency (2–5 seconds)
* Simple operation by a non-technical user
* Low hardware requirements
* Easy deployment on a laptop

### Non-Goals

* Automatic language detection
* Speaker diarization
* Enterprise-scale infrastructure
* Live audio streaming to the cloud
* Recording and archiving audio

---

# 2. Architecture

```text
Audio Input
    ↓
Audio Capture
    ↓
Voice Activity Detection
    ↓
Whisper Transcription
    ↓
Segment Stabilization
    ↓
Transcript Event Generation
    ↓
WebSocket Client
    ↓
Translation Server
```

---

# 3. Functional Requirements

## FR-001 Audio Capture

The system shall capture audio from:

* USB audio interfaces
* Mixer line inputs
* USB mixer devices

### Requirements

* Mono audio
* 16 kHz sampling rate
* PCM audio stream
* Continuous capture

---

## FR-002 Language Selection

The operator shall manually select the source language before a speech begins.

### Supported Languages

Initial MVP:

* English
* French
* Czech
* Polish

### Behavior

Selected language remains active until manually changed.

No automatic language detection shall be performed.

---

## FR-003 Voice Activity Detection

The system shall detect speech regions.

### Requirements

* Ignore silence
* Ignore low-level background noise
* Buffer speech segments

### Recommended Library

* Silero VAD

---

## FR-004 Speech Recognition

The system shall transcribe speech using a local Whisper model.

### Recommended Engine

* faster-whisper

### Model Configuration

Default:

```text
Model: medium
Device: cuda
Compute Type: float16
```

Fallback:

```text
Model: small
Device: cpu
```

### Requirements

* Streaming operation
* Incremental recognition
* Multilingual support

---

## FR-005 Transcript Stabilization

The system shall prevent unstable transcript updates from being transmitted.

### Requirements

The client shall:

* maintain a current transcript buffer
* track transcript revisions
* commit only stabilized text

### Stabilization Rules

A segment becomes finalized when:

* speech pause exceeds 700 ms

OR

* transcript unchanged for 2 seconds

---

## FR-006 Transcript Segmentation

The system shall divide speech into logical segments.

### Target Segment Length

* 1–2 sentences
* 5–20 seconds of speech

### Example

Input:

```text
Thank you all for coming today. We are very happy to see everyone here.
```

Output:

```text
Segment 101
```

---

## FR-007 Server Communication

The system shall transmit finalized transcript segments to the translation server.

### Transport

* WebSocket

### Fallback

* HTTPS POST

---

## FR-008 Offline Operation

If network connectivity is unavailable:

* transcription shall continue
* finalized segments shall be queued locally

When connectivity is restored:

* queued segments shall be transmitted

---

# 4. Transcript Event Format

## Event Type

TranscriptSegment

### JSON Schema

```json
{
  "event_id": "wedding-2027",
  "segment_id": 101,
  "sequence_number": 101,
  "timestamp": "2027-06-15T18:23:45Z",
  "source_language": "cs",
  "text": "Mockrát děkujeme, že jste dneska přišli",
  "final": true
}
```

### Field Definitions

| Field           | Type    | Description                |
| --------------- | ------- | -------------------------- |
| event_id        | string  | Event identifier           |
| segment_id      | integer | Unique segment identifier  |
| sequence_number | integer | Monotonic ordering         |
| timestamp       | string  | UTC timestamp              |
| source_language | string  | Operator-selected language |
| text            | string  | Final transcript           |
| final           | boolean | Always true in MVP         |

---

# 5. User Interface

## Operator Window

Single-page desktop application.

### Components

#### Audio Status

Displays:

* Connected
* Disconnected
* Audio level

---

#### Language Selector

Dropdown list:

```text
English
French
Czech
Polish
```

---

#### Current Transcript

Displays current live transcript.

### Example

```text
Current Speech

Thank you all for coming tonight...
```

---

#### Connection Status

Displays:

```text
Server Connected
```

or

```text
Offline Queue: 4 segments
```

---

# 6. Audio Processing Pipeline

## Input Configuration

```yaml
sample_rate: 16000
channels: 1
sample_format: int16
chunk_duration_ms: 100
```

---

## Internal Pipeline

```text
Audio Capture
    ↓
Resampling
    ↓
VAD
    ↓
Speech Buffer
    ↓
Whisper
    ↓
Stabilization
    ↓
Final Segment
```

---

# 7. Reliability Requirements

## Network Failure

When connection is lost:

* queue transcript segments
* retry every 5 seconds

---

## Whisper Failure

If transcription engine crashes:

* display error message
* allow manual restart

---

## Audio Device Failure

If input device disconnects:

* notify operator
* continuously retry connection

---

# 8. Local Storage

## Purpose

Temporary recovery storage.

### Storage Location

```text
data/transcript_queue.db
```

### Technology

SQLite

---

## Tables

### pending_segments

| Column     | Type     |
| ---------- | -------- |
| id         | integer  |
| payload    | json     |
| created_at | datetime |

---

# 9. Configuration

## config.yaml

```yaml
event_id: wedding-2027

server:
  websocket_url: wss://translate.example.com/ws

audio:
  device_id: default

transcription:
  model: medium
  language: en

stabilization:
  silence_ms: 700
  stable_ms: 2000
```

---

# 10. Logging

## Log Levels

* INFO
* WARNING
* ERROR

### Log File

```text
logs/transcriber.log
```

### Example

```text
[INFO] Audio device connected
[INFO] Segment 101 finalized
[INFO] Segment 101 transmitted
```

---

# 11. Recommended Technology Stack

## Runtime

* Python 3.12+

## Audio

* sounddevice

## VAD

* Silero VAD

## Transcription

* faster-whisper

## Networking

* websockets

## Local Storage

* SQLite

## Configuration

* pydantic-settings
* PyYAML

## Logging

* standard logging module

---

# 12. MVP Acceptance Criteria

The MVP shall be considered complete when:

1. Audio can be captured from a USB mixer.
2. Speech can be transcribed locally.
3. Source language can be selected manually.
4. Finalized transcript segments are generated.
5. Segments are transmitted to the server.
6. Offline queueing functions correctly.
7. End-to-end latency remains below 5 seconds.
8. Application runs continuously for at least 4 hours without operator intervention.
