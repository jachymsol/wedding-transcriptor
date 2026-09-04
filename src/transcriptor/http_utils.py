"""Shared HTTP URL helpers for talking to the translation server's admin API.

Both :mod:`transcriptor.server` (background WebSocket/HTTP client) and
:mod:`transcriptor.startup` (the operator-facing startup dialog) need to
derive an HTTP(S) base URL from the configured ``wss://``/``ws://``
WebSocket URL, and both need to build ``/admin/events/{event_id}`` URLs.
This module is the single source of truth for that logic so it isn't
duplicated (and doesn't drift) across call sites.
"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

__all__ = ["ws_url_to_http", "admin_url"]


def ws_url_to_http(ws_url: str) -> str:
    """Rewrite ``wss://`` -> ``https://`` or ``ws://`` -> ``http://``.

    URLs that already use an HTTP(S) scheme (or an unrecognised scheme) are
    returned unchanged.
    """
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[6:]
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[5:]
    return ws_url


def admin_url(ws_url: str, event_id: str) -> str:
    """Build ``scheme://host[:port]/admin/events/{event_id}``.

    Any path suffix carried by *ws_url* (e.g. ``wss://host/ws``) is
    stripped, so the admin API is always addressed at the server root
    regardless of what path the WebSocket endpoint lives at.
    """
    http_url = ws_url_to_http(ws_url).rstrip("/")
    parsed = urlparse(http_url)
    origin = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    return f"{origin}/admin/events/{event_id}"
