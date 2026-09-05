"""Tests for transcriptor.server — ServerClient.

No real network connections are made.  WebSocket and HTTP behaviour is
injected via fake factories / posters so the tests run instantly without
any I/O.
"""

from __future__ import annotations

import threading
import time
import urllib.error
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from transcriptor.config import AppConfig, ServerConfig, StabilizationConfig
from transcriptor.server import (
    ConnectionState,
    ServerClient,
    fetch_last_segment_id,
)
from transcriptor.stabilization import TranscriptSegment
from transcriptor.storage import SegmentQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(host: str = "example.com") -> AppConfig:
    return AppConfig(server=ServerConfig(host=host))


def _make_segment(
    segment_id: int = 1,
    text: str = "Hello",
    language: str = "en",
) -> TranscriptSegment:
    return TranscriptSegment(
        event_id="test-event",
        segment_id=segment_id,
        sequence_number=segment_id,
        timestamp="2027-06-14T12:00:00Z",
        source_language=language,
        text=text,
    )


@pytest.fixture
def tmp_queue(tmp_path: Path) -> SegmentQueue:
    q = SegmentQueue(db_path=tmp_path / "test_server.db")
    yield q
    q.close()


class _FakeWs:
    """Fake WebSocket connection for tests."""

    def __init__(self, *, recv_raises=None, send_raises=None):
        self.sent: List[str] = []
        self._recv_raises = recv_raises
        self._send_raises = send_raises
        self.closed = False
        self._stop = threading.Event()

    def send(self, data: str) -> None:
        if self._send_raises:
            raise self._send_raises
        self.sent.append(data)

    def recv(self, timeout: float = 1.0):
        # Block briefly then raise TimeoutError so the loop ticks naturally
        if self._stop.wait(min(timeout, 0.05)):
            raise Exception("Connection closed")
        raise TimeoutError

    def close(self) -> None:
        self.closed = True
        self._stop.set()

    def disconnect(self) -> None:
        """Simulate remote close."""
        self._stop.set()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_disconnected(self, tmp_queue):
        client = ServerClient(_make_config(), tmp_queue)
        assert client.state == ConnectionState.DISCONNECTED

    def test_is_connected_false_initially(self, tmp_queue):
        client = ServerClient(_make_config(), tmp_queue)
        assert client.is_connected is False


# ---------------------------------------------------------------------------
# send() — offline (no connection)
# ---------------------------------------------------------------------------

class TestSendOffline:
    def test_segment_queued_when_disconnected(self, tmp_queue):
        client = ServerClient(_make_config(), tmp_queue)
        seg = _make_segment(1)
        client.send(seg)
        assert tmp_queue.count() == 1

    def test_queued_segment_content_correct(self, tmp_queue):
        client = ServerClient(_make_config(), tmp_queue)
        seg = _make_segment(42, text="Queued text")
        client.send(seg)
        _, recovered = tmp_queue.peek()[0]
        assert recovered == seg

    def test_multiple_segments_queued_in_order(self, tmp_queue):
        client = ServerClient(_make_config(), tmp_queue)
        for i in range(3):
            client.send(_make_segment(i + 1, text=f"s{i+1}"))
        segments = [seg for _, seg in tmp_queue.peek()]
        assert [s.segment_id for s in segments] == [1, 2, 3]


# ---------------------------------------------------------------------------
# send() — WebSocket connected
# ---------------------------------------------------------------------------

class TestSendWebSocket:
    def _connected_client(self, tmp_queue, ws):
        """Build a client and manually set it to CONNECTED_WS state."""
        client = ServerClient(_make_config(), tmp_queue)
        with client._state_lock:
            client._state = ConnectionState.CONNECTED_WS
            client._ws_conn = ws
        return client

    def test_send_transmits_via_ws(self, tmp_queue):
        ws = _FakeWs()
        client = self._connected_client(tmp_queue, ws)
        seg = _make_segment(1, text="Hello WS")
        client.send(seg)
        assert len(ws.sent) == 1
        assert "Hello WS" in ws.sent[0]

    def test_send_via_ws_does_not_queue(self, tmp_queue):
        ws = _FakeWs()
        client = self._connected_client(tmp_queue, ws)
        client.send(_make_segment(1))
        assert tmp_queue.count() == 0

    def test_send_via_ws_json_valid(self, tmp_queue):
        import json
        ws = _FakeWs()
        client = self._connected_client(tmp_queue, ws)
        seg = _make_segment(7, text="Test segment", language="cs")
        client.send(seg)
        data = json.loads(ws.sent[0])
        assert data["segment_id"] == 7
        assert data["text"] == "Test segment"
        assert data["source_language"] == "cs"

    def test_send_failure_queues_segment(self, tmp_queue):
        ws = _FakeWs(send_raises=OSError("broken pipe"))
        client = self._connected_client(tmp_queue, ws)
        client.send(_make_segment(1))
        assert tmp_queue.count() == 1

    def test_send_failure_sets_disconnected(self, tmp_queue):
        ws = _FakeWs(send_raises=OSError("broken pipe"))
        client = self._connected_client(tmp_queue, ws)
        client.send(_make_segment(1))
        assert client.state == ConnectionState.DISCONNECTED

    def test_send_non_ascii_via_ws(self, tmp_queue):
        ws = _FakeWs()
        client = self._connected_client(tmp_queue, ws)
        seg = _make_segment(1, text="Příliš žluťoučký kůň")
        client.send(seg)
        assert "Příliš žluťoučký kůň" in ws.sent[0]


# ---------------------------------------------------------------------------
# send() — HTTP fallback connected
# ---------------------------------------------------------------------------

class TestSendHttp:
    def _http_client(self, tmp_queue, poster):
        client = ServerClient(_make_config(), tmp_queue, http_poster=poster)
        with client._state_lock:
            client._state = ConnectionState.CONNECTED_HTTP
        return client

    def test_send_transmits_via_http(self, tmp_queue):
        calls = []
        def poster(url, payload, headers=None):
            calls.append((url, payload))
            return True

        client = self._http_client(tmp_queue, poster)
        client.send(_make_segment(1, text="HTTP segment"))
        assert len(calls) == 1
        assert b"HTTP segment" in calls[0][1]

    def test_send_via_http_does_not_queue_on_success(self, tmp_queue):
        client = self._http_client(tmp_queue, lambda u, p, h=None: True)
        client.send(_make_segment(1))
        assert tmp_queue.count() == 0

    def test_send_via_http_queues_on_failure(self, tmp_queue):
        client = self._http_client(tmp_queue, lambda u, p, h=None: False)
        client.send(_make_segment(1))
        assert tmp_queue.count() == 1

    def test_http_url_derived_from_host(self, tmp_queue):
        captured = []
        def poster(url, payload, headers=None):
            captured.append(url)
            return True

        config = _make_config(host="host:9000")
        client = ServerClient(config, tmp_queue, http_poster=poster)
        with client._state_lock:
            client._state = ConnectionState.CONNECTED_HTTP
        client.send(_make_segment(1))
        assert captured[0] == "https://host:9000/ingest"

    def test_authorization_header_sent_when_api_key_set(self, tmp_queue):
        captured_headers = []
        def poster(url, payload, headers=None):
            captured_headers.append(headers)
            return True

        config = AppConfig(api_key="secret-key", server=ServerConfig(host="example.com"))
        client = ServerClient(config, tmp_queue, http_poster=poster)
        with client._state_lock:
            client._state = ConnectionState.CONNECTED_HTTP
        client.send(_make_segment(1))
        assert captured_headers[0] == {"Authorization": "Bearer secret-key"}

    def test_no_authorization_header_when_api_key_empty(self, tmp_queue):
        captured_headers = []
        def poster(url, payload, headers=None):
            captured_headers.append(headers)
            return True

        config = AppConfig(api_key="", server=ServerConfig(host="example.com"))
        client = ServerClient(config, tmp_queue, http_poster=poster)
        with client._state_lock:
            client._state = ConnectionState.CONNECTED_HTTP
        client.send(_make_segment(1))
        assert captured_headers[0] == {}


# ---------------------------------------------------------------------------
# WebSocket connection URL / headers — auth (FR-007/§7)
# ---------------------------------------------------------------------------

class TestWebSocketAuth:
    def test_ws_url_includes_token_query_param(self, tmp_queue):
        captured = []
        def ws_factory(url, additional_headers=None):
            captured.append(url)
            raise OSError("stop after capture")

        config = AppConfig(api_key="secret-key", server=ServerConfig(host="example.com"))
        client = ServerClient(config, tmp_queue, ws_factory=ws_factory)
        client._try_ws()
        assert captured[0] == "wss://example.com/ws/ingest?token=secret-key"

    def test_ws_no_token_param_when_api_key_empty(self, tmp_queue):
        captured = []
        def ws_factory(url, additional_headers=None):
            captured.append(url)
            raise OSError("stop after capture")

        config = AppConfig(api_key="", server=ServerConfig(host="example.com"))
        client = ServerClient(config, tmp_queue, ws_factory=ws_factory)
        client._try_ws()
        assert captured[0] == "wss://example.com/ws/ingest"

    def test_ws_authorization_header_also_sent_when_api_key_set(self, tmp_queue):
        captured_headers = []
        def ws_factory(url, additional_headers=None):
            captured_headers.append(additional_headers)
            raise OSError("stop after capture")

        config = AppConfig(api_key="secret-key", server=ServerConfig(host="example.com"))
        client = ServerClient(config, tmp_queue, ws_factory=ws_factory)
        client._try_ws()
        assert captured_headers[0] == {"Authorization": "Bearer secret-key"}


# ---------------------------------------------------------------------------
# Queue drain on reconnect
# ---------------------------------------------------------------------------

class TestQueueDrain:
    def test_ws_connect_drains_queue(self, tmp_queue):
        # Pre-fill queue with 3 segments
        for i in range(1, 4):
            tmp_queue.enqueue(_make_segment(i, text=f"queued {i}"))

        ws = _FakeWs()
        connected = threading.Event()

        def ws_factory(url, additional_headers=None):
            # Signal that the factory was called, then return ws immediately.
            # Blocking here before returning ws would prevent _drain_queue()
            # from running, creating a circular dependency with any "wait for
            # drain" logic in the test.
            connected.set()
            return ws

        client = ServerClient(_make_config(), tmp_queue, ws_factory=ws_factory, retry_interval_s=0.05)
        client.start()
        assert connected.wait(timeout=2.0), "ws_factory was never called"

        # Wait for _drain_queue() to empty the queue (runs in the server thread
        # immediately after the factory returns and state is set to CONNECTED_WS).
        deadline = time.monotonic() + 2.0
        while tmp_queue.count() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        client.stop()

        # ws.sent[0] is the "start" control message sent on every new
        # WS connection; the 3 queued segments follow it.
        assert len(ws.sent) == 4
        assert tmp_queue.count() == 0

    def test_http_fallback_drains_queue(self, tmp_queue):
        for i in range(1, 3):
            tmp_queue.enqueue(_make_segment(i))

        http_calls: List[bytes] = []

        ws_attempts = 0

        def ws_factory(url, additional_headers=None):
            nonlocal ws_attempts
            ws_attempts += 1
            raise OSError("no WS")

        def poster(url, payload, headers=None):
            http_calls.append(payload)
            return True

        # retry_interval_s=0 so it loops fast; stop after first HTTP drain
        stop_event = threading.Event()

        def fast_factory(url, additional_headers=None):
            raise OSError("no WS")

        client = ServerClient(
            _make_config(), tmp_queue,
            ws_factory=fast_factory,
            http_poster=poster,
            retry_interval_s=0.05,
        )
        client.start()
        # Wait for queue to drain
        deadline = time.monotonic() + 3.0
        while tmp_queue.count() > 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        client.stop()

        assert tmp_queue.count() == 0
        assert len(http_calls) == 2

    def test_drain_stops_on_send_failure(self, tmp_queue):
        """If the WS drops mid-drain, remaining segments stay in queue."""
        for i in range(1, 4):
            tmp_queue.enqueue(_make_segment(i))

        send_count = 0

        class FailAfterOne:
            def send(self, data):
                nonlocal send_count
                send_count += 1
                if send_count > 1:
                    raise OSError("dropped")

            def recv(self, timeout=1.0):
                time.sleep(timeout)
                raise TimeoutError

            def close(self):
                pass

        client = ServerClient(_make_config(), tmp_queue,
                              ws_factory=lambda url, additional_headers=None: FailAfterOne(),
                              retry_interval_s=60)  # don't retry during test
        client.start()

        deadline = time.monotonic() + 2.0
        # wait until first drain attempt
        while send_count == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.1)  # let drain complete
        client.stop()

        # send_count: 1 = "start" control message (succeeds), 2 = first
        # queued segment (fails, since send_count > 1) — drain stops
        # immediately, so no queued segments are removed.
        assert tmp_queue.count() == 3


# ---------------------------------------------------------------------------
# start() / stop()
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_stop_before_start_is_safe(self, tmp_queue):
        client = ServerClient(_make_config(), tmp_queue)
        client.stop()  # must not raise

    def test_start_creates_background_thread(self, tmp_queue):
        ws = _FakeWs()
        client = ServerClient(
            _make_config(), tmp_queue,
            ws_factory=lambda url, additional_headers=None: ws,
            retry_interval_s=60,
        )
        client.start()
        time.sleep(0.1)
        assert client._thread is not None
        assert client._thread.is_alive()
        client.stop()

    def test_stop_terminates_thread(self, tmp_queue):
        ws = _FakeWs()
        client = ServerClient(
            _make_config(), tmp_queue,
            ws_factory=lambda url, additional_headers=None: ws,
            retry_interval_s=60,
        )
        client.start()
        time.sleep(0.05)
        client.stop()
        assert not client._thread.is_alive()

    def test_stop_sets_disconnected(self, tmp_queue):
        ws = _FakeWs()
        client = ServerClient(
            _make_config(), tmp_queue,
            ws_factory=lambda url, additional_headers=None: ws,
            retry_interval_s=60,
        )
        client.start()
        time.sleep(0.05)
        client.stop()
        assert client.state == ConnectionState.DISCONNECTED


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retry_interval_honoured(self, tmp_queue):
        """Background thread should attempt to connect at least twice
        within 2 × retry_interval + buffer."""
        attempts = []

        def failing_factory(url, additional_headers=None):
            attempts.append(time.monotonic())
            raise OSError("fail")

        client = ServerClient(
            _make_config(), tmp_queue,
            ws_factory=failing_factory,
            http_poster=lambda u, p, h=None: False,
            retry_interval_s=0.1,
        )
        client.start()
        time.sleep(0.35)
        client.stop()

        assert len(attempts) >= 2

    def test_retry_after_ws_disconnect(self, tmp_queue):
        """After a WS drop, the loop should reconnect within retry_interval."""
        connections = []

        def factory(url, additional_headers=None):
            ws = _FakeWs()
            connections.append(ws)
            if len(connections) == 1:
                # First connection: disconnect quickly
                threading.Timer(0.05, ws.disconnect).start()
            return ws

        client = ServerClient(
            _make_config(), tmp_queue,
            ws_factory=factory,
            retry_interval_s=0.1,
        )
        client.start()
        time.sleep(0.5)
        client.stop()

        assert len(connections) >= 2


# ---------------------------------------------------------------------------
# fetch_last_segment_id — resume support (FR-007 extension)
# ---------------------------------------------------------------------------

class _FakeHttpResponse:
    """Fake context-manager response object mimicking urllib's response."""

    def __init__(self, status: int, body: bytes = b""):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class TestFetchLastSegmentId:
    @staticmethod
    def _event_body(segment_count: int = 0, **overrides) -> bytes:
        """Build a realistic GET /admin/events/:event_id response body."""
        import json as _json
        payload = {
            "id": "wedding-2027",
            "status": "live",
            "createdAt": "2026-09-04T12:00:00.000Z",
            "startedAt": "2026-09-04T12:00:00.000Z",
            "endedAt": None,
            "segmentCount": segment_count,
            "translationCount": segment_count * 2,
        }
        payload.update(overrides)
        return _json.dumps(payload).encode("utf-8")

    def test_returns_last_segment_id_on_success(self):
        opener = lambda req: _FakeHttpResponse(200, self._event_body(42))
        config = _make_config()
        assert fetch_last_segment_id(config, opener=opener) == 42

    def test_missing_key_in_body_defaults_to_zero(self):
        opener = lambda req: _FakeHttpResponse(200, b'{"id": "wedding-2027"}')
        config = _make_config()
        assert fetch_last_segment_id(config, opener=opener) == 0

    def test_empty_body_defaults_to_zero(self):
        opener = lambda req: _FakeHttpResponse(200, b"")
        config = _make_config()
        assert fetch_last_segment_id(config, opener=opener) == 0

    def test_404_returns_zero(self):
        def opener(req):
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", {}, None
            )

        config = _make_config()
        assert fetch_last_segment_id(config, opener=opener) == 0

    def test_other_http_error_returns_zero(self):
        def opener(req):
            raise urllib.error.HTTPError(req.full_url, 500, "Server Error", {}, None)

        config = _make_config()
        assert fetch_last_segment_id(config, opener=opener) == 0

    def test_non_2xx_status_returns_zero(self):
        opener = lambda req: _FakeHttpResponse(301, self._event_body(9))
        config = _make_config()
        assert fetch_last_segment_id(config, opener=opener) == 0

    def test_connection_error_returns_zero(self):
        def opener(req):
            raise OSError("connection refused")

        config = _make_config()
        assert fetch_last_segment_id(config, opener=opener) == 0

    def test_url_targets_admin_events_endpoint(self):
        seen_urls = []

        def opener(req):
            seen_urls.append(req.full_url)
            return _FakeHttpResponse(200, self._event_body(0))

        config = AppConfig(
            event_id="wedding-2027",
            server=ServerConfig(host="host:8443"),
        )
        fetch_last_segment_id(config, opener=opener)
        assert seen_urls == ["https://host:8443/admin/events/wedding-2027"]

    def test_authorization_header_added_when_api_key_set(self):
        seen_headers = []

        def opener(req):
            seen_headers.append(dict(req.header_items()))
            return _FakeHttpResponse(200, self._event_body(0))

        config = AppConfig(
            api_key="secret-key",
            server=ServerConfig(host="example.com"),
        )
        fetch_last_segment_id(config, opener=opener)
        assert seen_headers[0].get("Authorization") == "Bearer secret-key"

    def test_no_authorization_header_when_api_key_empty(self):
        seen_headers = []

        def opener(req):
            seen_headers.append(dict(req.header_items()))
            return _FakeHttpResponse(200, self._event_body(0))

        config = _make_config()
        config = config.model_copy(update={"api_key": ""})
        fetch_last_segment_id(config, opener=opener)
        assert "Authorization" not in seen_headers[0]
