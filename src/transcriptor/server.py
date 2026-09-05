"""WebSocket/HTTPS server communication — FR-007, FR-008, spec §7.

Transmits finalized :class:`~transcriptor.stabilization.TranscriptSegment`
events to the translation server.

Connection strategy
-------------------
1. **WebSocket** (preferred) — persistent connection to
   ``wss://{config.server.host}/ws/ingest`` (or ``ws://`` for local hosts).
2. **HTTPS POST** (fallback) — stateless POST to
   ``https://{config.server.host}/ingest`` (or ``http://`` for local hosts).

On any failure the background thread waits ``RETRY_INTERVAL_S`` (5 s) and
retries from the top.  The offline
:class:`~transcriptor.storage.SegmentQueue` is drained immediately after
every successful (re-)connection.

Injection seams (for testing)
------------------------------
* *ws_factory* — ``callable(url: str) → connection`` where the object
  exposes ``.send(data: str)``, ``.recv(timeout: float)``, and
  ``.close()``.
* *http_poster* — ``callable(url: str, payload: bytes) → bool``.

Why the WebSocket transport bridges to asyncio
------------------------------------------------
The default *ws_factory* (:func:`_default_ws_factory`) uses ``websockets``'
**asyncio** client (:func:`websockets.connect`) internally, bridged to the
synchronous duck-typed interface above via :class:`_AsyncioWsBridge`. This
is deliberate: the ``websockets.sync.client`` (threading-based) client was
found, via direct reproduction against a real deployment, to hang for
~60 seconds and then fail on every single connection attempt against one
production server/proxy combination — across every tested library version
(12.0 through 16.0) — while the asyncio client connected instantly and
reliably (100s of consecutive successful connects) against the exact same
URL, token, and headers. The root cause appears to be specific to the sync
client's threading implementation; since :class:`_AsyncioWsBridge` isolates
that on a dedicated event-loop thread per connection attempt, the rest of
this module is unaffected and unaware of the underlying asyncio machinery.

Usage::

    client = ServerClient(config, queue)
    client.start()
    ...
    client.send(segment)          # called from the pipeline thread
    ...
    client.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.error
import urllib.request
from enum import Enum, auto
from typing import Any, Callable, Optional

from transcriptor.config import AppConfig
from transcriptor.http_utils import admin_url
from transcriptor.http_utils import http_ingest_url as _http_ingest_url
from transcriptor.http_utils import ws_url as _ws_url
from transcriptor.stabilization import TranscriptSegment
from transcriptor.storage import SegmentQueue

log = logging.getLogger(__name__)

RETRY_INTERVAL_S: float = 5.0
_WS_OPEN_TIMEOUT_S: float = 10.0


# ---------------------------------------------------------------------------
# ConnectionState
# ---------------------------------------------------------------------------

class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTED_WS = auto()
    CONNECTED_HTTP = auto()


# ---------------------------------------------------------------------------
# Default transport implementations
# ---------------------------------------------------------------------------

class _AsyncioWsBridge:
    """Bridges an asyncio ``websockets`` connection to a synchronous
    ``.send(str)`` / ``.recv(timeout)`` / ``.close()`` duck-type.

    Runs a dedicated event loop on its own background thread for the
    lifetime of one WebSocket connection attempt. All calls block the
    caller's thread until the corresponding coroutine completes (or raises),
    preserving the same synchronous contract the rest of this module relies
    on — callers don't need to know asyncio is involved at all.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._ws: Any = None

    def connect(
        self,
        url: str,
        additional_headers: dict,
        open_timeout: float = _WS_OPEN_TIMEOUT_S,
    ) -> None:
        """Open the connection; raises on failure or timeout."""
        ready = threading.Event()

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            ready.set()
            loop.run_forever()
            loop.close()

        self._loop_thread = threading.Thread(
            target=_run_loop, daemon=True, name="ws-asyncio-loop"
        )
        self._loop_thread.start()
        ready.wait()

        import websockets

        async def _connect():
            return await websockets.connect(url, additional_headers=additional_headers)

        future = asyncio.run_coroutine_threadsafe(_connect(), self._loop)
        try:
            self._ws = future.result(timeout=open_timeout)
        except Exception:
            self._shutdown_loop()
            raise

    def send(self, payload: str) -> None:
        future = asyncio.run_coroutine_threadsafe(self._ws.send(payload), self._loop)
        future.result(timeout=_WS_OPEN_TIMEOUT_S)

    def recv(self, timeout: float = 1.0) -> Any:
        async def _recv():
            return await asyncio.wait_for(self._ws.recv(), timeout=timeout)

        future = asyncio.run_coroutine_threadsafe(_recv(), self._loop)
        try:
            return future.result(timeout=timeout + 1.0)
        except asyncio.TimeoutError:
            raise TimeoutError from None

    def close(self) -> None:
        if self._loop is not None and self._ws is not None:
            future = asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
            try:
                future.result(timeout=5.0)
            except Exception:
                pass
        self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)


def _default_ws_factory(url: str, additional_headers: dict | None = None):
    """Open a WebSocket connection (requires websockets >= 14).

    See the module docstring ("Why the WebSocket transport bridges to
    asyncio") for why this uses :class:`_AsyncioWsBridge` instead of
    ``websockets.sync.client`` directly.
    """
    try:
        import websockets  # noqa: F401  (import-checked here for the error message)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "websockets package not installed; run: pip install websockets"
        ) from exc
    bridge = _AsyncioWsBridge()
    bridge.connect(url, additional_headers or {})
    return bridge


def _default_http_poster(url: str, payload: bytes, headers: dict | None = None) -> bool:
    """HTTP POST *payload* (JSON bytes) to *url*; return True on 2xx."""
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        log.warning("HTTP POST to %s failed: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# Resume support — query the server for the last segment_id of an event
# ---------------------------------------------------------------------------

def fetch_last_segment_id(
    config: AppConfig,
    *,
    timeout: float = 5.0,
    opener: Optional[Callable[[urllib.request.Request], Any]] = None,
) -> int:
    """Return the highest ``segment_id`` already sent for ``config.event_id``.

    Calls ``GET /admin/events/{event_id}`` on the translation server so a
    restarted client can resume numbering instead of starting over at 1
    (which would collide with segments from a previous run of the app for
    the same event).

    The endpoint's ``200`` response body looks like::

        {
          "id": "wedding-2027",
          "status": "waiting" | "live" | "paused" | "ended",
          "createdAt": "...", "startedAt": "...", "endedAt": "...",
          "segmentCount": 42,
          "translationCount": 84
        }

    Only ``segmentCount`` is used here (segment_id/sequence_number values
    are assigned sequentially with no gaps, so the count of segments
    persisted for the event equals the highest segment_id sent so far).

    Returns ``0`` (meaning "no prior segments") in every case where a
    resumable number can't be determined:

    * The server responds ``404`` (event has never been registered/started)
      — logged at INFO, this is an expected case for a brand-new event.
    * The server is unreachable, times out, or returns a non-2xx status
      other than 404, or an unparseable body — logged at WARNING, since this
      means we *can't tell* whether prior segments exist.

    Callers should use ``fetch_last_segment_id(config) + 1`` as the starting
    ``segment_id`` for a new :class:`~transcriptor.stabilization.Stabilizer`.
    """
    url = admin_url(config.server.host, config.event_id)
    req = urllib.request.Request(url, method="GET")
    if config.api_key:
        req.add_header("Authorization", f"Bearer {config.api_key}")

    opener = opener or (lambda r: urllib.request.urlopen(r, timeout=timeout))

    try:
        with opener(req) as resp:
            if not (200 <= resp.status < 300):
                log.warning(
                    "Unexpected status %d fetching last segment id for '%s' — "
                    "starting numbering at 1",
                    resp.status, config.event_id,
                )
                return 0
            raw = resp.read()
            body = json.loads(raw.decode("utf-8")) if raw else {}
            return int(body.get("segmentCount", 0) or 0)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            log.info(
                "Event '%s' not found on server — starting numbering at 1",
                config.event_id,
            )
        else:
            log.warning(
                "Failed to fetch last segment id for '%s' (HTTP %d) — "
                "starting numbering at 1",
                config.event_id, exc.code,
            )
        return 0
    except Exception as exc:
        log.warning(
            "Failed to fetch last segment id for '%s' (%s) — starting numbering at 1",
            config.event_id, exc,
        )
        return 0


# ---------------------------------------------------------------------------
# ServerClient
# ---------------------------------------------------------------------------

class ServerClient:
    """Manages the connection to the translation server.

    Runs a background reconnect/retry thread (started by :meth:`start`) that
    is entirely separate from the audio pipeline thread.  All shared state is
    protected by :attr:`_state_lock`.

    Parameters
    ----------
    config:
        Application config — only ``config.server.host`` is used.
    queue:
        Offline :class:`~transcriptor.storage.SegmentQueue` for durable
        persistence during outages.
    ws_factory:
        Optional override for the WebSocket connector (injection seam for
        tests).  Default: :func:`_default_ws_factory`.
    http_poster:
        Optional override for the HTTP POST sender (injection seam for
        tests).  Default: :func:`_default_http_poster`.
    retry_interval_s:
        Seconds to wait between reconnect attempts.  Default: 5.
    """

    def __init__(
        self,
        config: AppConfig,
        queue: SegmentQueue,
        *,
        ws_factory: Optional[Callable] = None,
        http_poster: Optional[Callable] = None,
        retry_interval_s: float = RETRY_INTERVAL_S,
    ) -> None:
        self._event_id = config.event_id
        self._api_key = config.api_key
        self._url_ws = _ws_url(config.server.host, self._api_key)
        self._url_http = _http_ingest_url(config.server.host)
        self._queue = queue
        self._ws_factory = ws_factory or _default_ws_factory
        self._http_poster = http_poster or _default_http_poster
        self._retry_interval_s = retry_interval_s

        self._state: ConnectionState = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()
        self._ws_conn = None  # active WebSocket connection object

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        """Current :class:`ConnectionState` (thread-safe snapshot)."""
        with self._state_lock:
            return self._state

    @property
    def is_connected(self) -> bool:
        """``True`` when a live WebSocket or HTTP connection is established."""
        return self.state != ConnectionState.DISCONNECTED

    def send(self, segment: TranscriptSegment) -> None:
        """Transmit *segment*, persisting it to the offline queue on failure.

        If no connection is active the segment is enqueued immediately so
        it will be retransmitted when connectivity returns.

        Thread-safe: may be called from any thread.
        """
        if self._try_send_raw(segment.to_json()):
            log.info("Segment %d sent", segment.segment_id)
            return
        self._queue.enqueue(segment)

    def send_section_break(self, break_before_sequence_number: int) -> None:
        """Send a ``section_break`` control message, queuing it offline if needed.

        Parameters
        ----------
        break_before_sequence_number:
            The sequence number of the next transcript segment that will
            follow this break.  The receiver uses this to insert a visual
            section break immediately before that segment.

        Thread-safe: may be called from any thread.
        """
        payload = json.dumps({
            "type": "control",
            "action": "section_break",
            "event_id": self._event_id,
            "break_before_sequence_number": break_before_sequence_number,
        })
        if not self._try_send_raw(payload):
            self._queue.enqueue_raw(payload)
            log.info(
                "Section break (before seq %d) queued offline",
                break_before_sequence_number,
            )
        else:
            log.info(
                "Section break sent (break_before_sequence_number=%d)",
                break_before_sequence_number,
            )

    def send_control(self, action: str) -> None:
        """Send a control message over the active WebSocket connection.

        For ``action="start"``: if no WebSocket is currently open the message
        is **not** queued — it will be sent automatically by :meth:`_try_ws`
        on the next (re-)connection, before any queued segments are drained.

        For ``action="stop"``: fire-and-forget.  If the connection is already
        closed the call returns silently without raising.

        Thread-safe: may be called from any thread.
        """
        payload = json.dumps({
            "type": "control",
            "action": action,
            "event_id": self._event_id,
        })
        with self._state_lock:
            state = self._state
            ws = self._ws_conn
        if state == ConnectionState.CONNECTED_WS and ws is not None:
            try:
                ws.send(payload)
                log.info("Control message sent: action=%s", action)
            except Exception as exc:
                log.warning("Failed to send control message (action=%s): %s", action, exc)
                with self._state_lock:
                    self._state = ConnectionState.DISCONNECTED
                    self._ws_conn = None
        else:
            log.debug(
                "Control message (action=%s) skipped — no active WebSocket", action
            )

    def start(self) -> None:
        """Start the background reconnect/drain thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="server-client",
        )
        self._thread.start()
        log.info("ServerClient started (WS: %s)", self._url_ws)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._close_ws()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info("ServerClient stopped")

    # ------------------------------------------------------------------
    # Background reconnect loop
    # ------------------------------------------------------------------

    def _reconnect_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._connect_cycle()
            except Exception as exc:
                log.error("Unhandled error in reconnect loop: %s", exc, exc_info=True)
            if not self._stop_event.is_set():
                log.info("Retrying in %.0f s", self._retry_interval_s)
                self._stop_event.wait(self._retry_interval_s)

    def _connect_cycle(self) -> None:
        """Try WebSocket; fall back to HTTP POST if WS is unavailable."""
        if self._try_ws():
            return
        if not self._stop_event.is_set():
            self._try_http()

    # ------------------------------------------------------------------
    # WebSocket path
    # ------------------------------------------------------------------

    def _try_ws(self) -> bool:
        """Attempt WebSocket connection; return True if it succeeded."""
        try:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            ws = self._ws_factory(self._url_ws, additional_headers=headers)
        except Exception as exc:
            log.warning("WebSocket connect failed (%s): %s", self._url_ws, exc)
            return False

        with self._state_lock:
            self._ws_conn = ws
            self._state = ConnectionState.CONNECTED_WS
        log.info("WebSocket connected to %s", self._url_ws)

        try:
            self._send_control_on_ws(ws, "start")
            self._drain_queue()
            self._ws_recv_loop(ws)
        except Exception as exc:
            log.warning("WebSocket session error: %s", exc)
        finally:
            self._close_ws()

        return True  # we established a connection (even if it dropped later)

    def _send_control_on_ws(self, ws, action: str) -> None:
        """Send a control message directly on *ws* (called from the server thread)."""
        payload = json.dumps({
            "type": "control",
            "action": action,
            "event_id": self._event_id,
        })
        try:
            ws.send(payload)
            log.info("Control message sent: action=%s", action)
        except Exception as exc:
            log.warning("Failed to send control message (action=%s): %s", action, exc)

    def _ws_recv_loop(self, ws) -> None:
        """Block until the WebSocket closes or :meth:`stop` is called."""
        while not self._stop_event.is_set():
            try:
                ws.recv(timeout=1.0)
            except TimeoutError:
                continue  # keep-alive tick
            except Exception:
                break  # disconnected

    def _close_ws(self) -> None:
        with self._state_lock:
            ws = self._ws_conn
            self._ws_conn = None
            self._state = ConnectionState.DISCONNECTED
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # HTTP POST fallback path
    # ------------------------------------------------------------------

    def _try_http(self) -> None:
        """Drain the queue via HTTPS POST (stateless fallback)."""
        with self._state_lock:
            self._state = ConnectionState.CONNECTED_HTTP
        log.info("HTTP POST fallback active (%s)", self._url_http)
        try:
            self._drain_queue()
        except Exception as exc:
            log.warning("HTTP drain error: %s", exc)
        finally:
            with self._state_lock:
                if self._state == ConnectionState.CONNECTED_HTTP:
                    self._state = ConnectionState.DISCONNECTED

    # ------------------------------------------------------------------
    # Queue drain (shared by both paths)
    # ------------------------------------------------------------------

    def _drain_queue(self) -> None:
        """Send all queued payloads over the active connection.

        Stops at the first send failure (connection dropped mid-drain).
        """
        for row_id, payload in self._queue.peek_all():
            if self._stop_event.is_set():
                break
            if self._try_send_raw(payload):
                self._queue.delete(row_id)
                log.info("Queued payload retransmitted (row %d)", row_id)
            else:
                break  # connection dropped — leave remainder for next cycle

    # ------------------------------------------------------------------
    # Internal send helpers
    # ------------------------------------------------------------------

    def _try_send_now(self, segment: TranscriptSegment) -> bool:
        """Attempt immediate send of *segment* over the current active connection.

        Returns ``True`` on success, ``False`` otherwise.
        """
        return self._try_send_raw(segment.to_json())

    def _try_send_raw(self, payload: str) -> bool:
        """Attempt immediate send of a raw JSON *payload* string.

        Returns ``True`` on success, ``False`` if no connection is active or
        if the send fails (in which case the WS state is cleared).
        """
        with self._state_lock:
            state = self._state
            ws = self._ws_conn

        if state == ConnectionState.CONNECTED_WS and ws is not None:
            try:
                ws.send(payload)
                log.debug("Payload sent via WebSocket (%d bytes)", len(payload))
                return True
            except Exception as exc:
                log.warning("WebSocket send failed: %s", exc)
                with self._state_lock:
                    self._state = ConnectionState.DISCONNECTED
                    self._ws_conn = None

        elif state == ConnectionState.CONNECTED_HTTP:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            ok = self._http_poster(self._url_http, payload.encode(), headers)
            if ok:
                log.debug("Payload sent via HTTP POST (%d bytes)", len(payload))
            return ok

        return False
