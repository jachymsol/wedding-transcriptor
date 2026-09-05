"""Shared URL-building helpers for talking to the translation server.

Both :mod:`transcriptor.server` (background WebSocket/HTTP client) and
:mod:`transcriptor.startup` (the operator-facing startup dialog) need to
build the various endpoint URLs — WebSocket ingest, HTTP ingest fallback,
and the admin API — from the configured bare ``server.host`` (e.g.
``"translate.example.com"`` or ``"localhost:3000"``). This module is the
single source of truth for that logic so it isn't duplicated (and doesn't
drift) across call sites.

Scheme inference
-----------------
A host of ``localhost`` or ``127.0.0.1`` (with or without a port) is treated
as a local development server and addressed over unencrypted ``ws://``/
``http://``. Any other host is addressed over ``wss://``/``https://``.
"""

from __future__ import annotations

from urllib.parse import quote

__all__ = ["ws_url", "http_ingest_url", "admin_url"]

_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1"}


def _is_local(host: str) -> bool:
    """Return True if *host* (optionally ``host:port``) is a local dev host."""
    hostname = host.split(":", 1)[0]
    return hostname in _LOCAL_HOSTNAMES


def _ws_scheme(host: str) -> str:
    return "ws" if _is_local(host) else "wss"


def _http_scheme(host: str) -> str:
    return "http" if _is_local(host) else "https"


def ws_url(host: str, api_key: str = "") -> str:
    """Build the WebSocket ingest URL for *host*.

    Includes ``?token=<api_key>`` (URL-encoded) as a query parameter when
    *api_key* is non-empty, so the server can authenticate the WebSocket
    upgrade request even if a proxy in front of it strips the
    ``Authorization`` header (a common limitation for WS handshakes).
    """
    url = f"{_ws_scheme(host)}://{host}/ws/ingest"
    if api_key:
        url += f"?token={quote(api_key, safe='')}"
    return url


def http_ingest_url(host: str) -> str:
    """Build the HTTPS POST ingest URL for *host* (fallback transport)."""
    return f"{_http_scheme(host)}://{host}/ingest"


def admin_url(host: str, event_id: str) -> str:
    """Build ``scheme://host/admin/events/{event_id}`` for *host*."""
    return f"{_http_scheme(host)}://{host}/admin/events/{event_id}"
