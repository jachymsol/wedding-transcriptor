"""Tests for transcriptor.http_utils — shared host-based URL helpers."""

from __future__ import annotations

from transcriptor.http_utils import admin_url, http_ingest_url, ws_url


class TestWsUrl:
    def test_remote_host_uses_wss(self):
        assert ws_url("example.com") == "wss://example.com/ws/ingest"

    def test_localhost_uses_ws(self):
        assert ws_url("localhost:3000") == "ws://localhost:3000/ws/ingest"

    def test_127_0_0_1_uses_ws(self):
        assert ws_url("127.0.0.1:3000") == "ws://127.0.0.1:3000/ws/ingest"

    def test_api_key_appended_as_token_query_param(self):
        assert (
            ws_url("example.com", api_key="secret-key")
            == "wss://example.com/ws/ingest?token=secret-key"
        )

    def test_api_key_url_encoded(self):
        assert (
            ws_url("example.com", api_key="a b/c")
            == "wss://example.com/ws/ingest?token=a%20b%2Fc"
        )

    def test_no_token_param_when_api_key_empty(self):
        assert ws_url("example.com", api_key="") == "wss://example.com/ws/ingest"


class TestHttpIngestUrl:
    def test_remote_host_uses_https(self):
        assert http_ingest_url("example.com") == "https://example.com/ingest"

    def test_localhost_uses_http(self):
        assert http_ingest_url("localhost:3000") == "http://localhost:3000/ingest"

    def test_preserves_port(self):
        assert http_ingest_url("host:8443") == "https://host:8443/ingest"


class TestAdminUrl:
    def test_remote_host_uses_https(self):
        assert (
            admin_url("example.com", "wedding-2027")
            == "https://example.com/admin/events/wedding-2027"
        )

    def test_preserves_host_and_port(self):
        assert (
            admin_url("host:8443", "my-event")
            == "https://host:8443/admin/events/my-event"
        )

    def test_localhost_uses_http(self):
        assert (
            admin_url("localhost:3000", "wedding-2027")
            == "http://localhost:3000/admin/events/wedding-2027"
        )
