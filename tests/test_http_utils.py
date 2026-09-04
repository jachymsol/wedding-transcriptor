"""Tests for transcriptor.http_utils — shared admin-API URL helpers."""

from __future__ import annotations

from transcriptor.http_utils import admin_url, ws_url_to_http


class TestWsUrlToHttp:
    def test_wss_becomes_https(self):
        assert ws_url_to_http("wss://example.com/ws") == "https://example.com/ws"

    def test_ws_becomes_http(self):
        assert ws_url_to_http("ws://example.com/ws") == "http://example.com/ws"

    def test_unknown_scheme_unchanged(self):
        assert ws_url_to_http("https://example.com") == "https://example.com"

    def test_path_and_port_preserved(self):
        assert ws_url_to_http("wss://host:8443/path/ws") == "https://host:8443/path/ws"


class TestAdminUrl:
    def test_strips_ws_path_suffix(self):
        assert (
            admin_url("wss://example.com/ws", "wedding-2027")
            == "https://example.com/admin/events/wedding-2027"
        )

    def test_preserves_host_and_port(self):
        assert (
            admin_url("wss://host:8443/ws/ingest", "my-event")
            == "https://host:8443/admin/events/my-event"
        )

    def test_ws_scheme_becomes_http(self):
        assert (
            admin_url("ws://localhost:3000/ws/ingest", "wedding-2027")
            == "http://localhost:3000/admin/events/wedding-2027"
        )

    def test_no_path_suffix(self):
        assert (
            admin_url("wss://example.com", "my-event")
            == "https://example.com/admin/events/my-event"
        )

    def test_trailing_slash_handled(self):
        assert (
            admin_url("wss://example.com/", "my-event")
            == "https://example.com/admin/events/my-event"
        )
