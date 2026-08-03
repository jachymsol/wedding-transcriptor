"""WebSocket/HTTPS server communication — FR-007, FR-008, spec §7.

Transmits finalized :class:`~transcriptor.stabilization.TranscriptSegment`
events to the translation server.

Connection strategy
-------------------
1. **WebSocket** (preferred) — persistent connection to
   ``config.server.websocket_url`` (``wss://…``).
2. **HTTPS POST** (fallback) — stateless POST to the same host/path with
   ``wss://`` rewritten to ``https://`` (``ws://`` → ``http://``).

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

Usage::

    client = ServerClient(config, queue)
    client.start()
    ...
    client.send(segment)          # called from the pipeline thread
    ...
    client.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from enum import Enum, auto
from typing import Callable, Optional

from transcriptor.config import AppConfig
from transcriptor.stabilization import TranscriptSegment
from transcriptor.storage import SegmentQueue

log = logging.getLogger(__name__)

RETRY_INTERVAL_S: float = 5.0


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

def _default_ws_factory(url: str):
    """Open a synchronous WebSocket connection (requires websockets >= 12)."""
    try:
        from websockets.sync.client import connect
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "websockets package not installed; run: pip install websockets"
        ) from exc
    return connect(url)


def _default_http_poster(url: str, payload: bytes) -> bool:
    """HTTP POST *payload* (JSON bytes) to *url*; return True on 2xx."""
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        log.warning("HTTP POST to %s failed: %s", url, exc)
        return False


def _ws_url_to_http(ws_url: str) -> str:
    """Rewrite ``wss://`` → ``https://`` or ``ws://`` → ``http://``."""
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[6:]
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[5:]
    return ws_url  # already HTTP or unknown scheme


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
        Application config — only ``config.server.websocket_url`` is used.
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
        self._url_ws = config.server.websocket_url
        self._url_http = _ws_url_to_http(self._url_ws)
        self._event_id = config.event_id
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
        if self._try_send_now(segment):
            return
        self._queue.enqueue(segment)

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
            ws = self._ws_factory(self._url_ws)
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
        """Send all queued segments over the active connection.

        Stops at the first send failure (connection dropped mid-drain).
        """
        for row_id, segment in self._queue.peek():
            if self._stop_event.is_set():
                break
            if self._try_send_now(segment):
                self._queue.delete(row_id)
                log.info(
                    "Queued segment %d retransmitted (row %d)",
                    segment.segment_id,
                    row_id,
                )
            else:
                break  # connection dropped — leave remainder for next cycle

    # ------------------------------------------------------------------
    # Internal send helpers
    # ------------------------------------------------------------------

    def _try_send_now(self, segment: TranscriptSegment) -> bool:
        """Attempt immediate send over the current active connection.

        Returns ``True`` on success, ``False`` if no connection or if the
        send fails (in which case the WS state is cleared).
        """
        payload = segment.to_json()

        with self._state_lock:
            state = self._state
            ws = self._ws_conn

        if state == ConnectionState.CONNECTED_WS and ws is not None:
            try:
                ws.send(payload)
                log.info("Segment %d sent via WebSocket", segment.segment_id)
                return True
            except Exception as exc:
                log.warning("WebSocket send failed: %s", exc)
                with self._state_lock:
                    self._state = ConnectionState.DISCONNECTED
                    self._ws_conn = None

        elif state == ConnectionState.CONNECTED_HTTP:
            ok = self._http_poster(self._url_http, payload.encode())
            if ok:
                log.info("Segment %d sent via HTTP POST", segment.segment_id)
            return ok

        return False
