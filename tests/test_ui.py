"""Tests for transcriptor.ui — AppUI and pure helper functions.

Pure-function tests need no display and run instantly.

AppUI tests create a real ``tk.Tk()`` window but destroy it immediately
after each test.  On headless CI systems tkinter may not be available; those
tests are skipped automatically via the ``tk_root`` fixture.
"""

from __future__ import annotations

import threading
import time
from typing import List

import pytest

from transcriptor.ui import (
    AppUI,
    DEFAULT_LANGUAGE_CODE,
    LANGUAGE_LABELS,
    LANGUAGES,
    audio_status_text,
    code_to_label,
    label_to_code,
    level_bar_color,
    server_status_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tk_root():
    """Provide a real tk.Tk root shared across the session, skipping if no display.

    The first Tk() call in this venv may fail due to a Tcl init quirk on
    macOS; we retry once before giving up.
    """
    import tkinter as tk
    root = None
    for _ in range(2):
        try:
            root = tk.Tk()
            root.withdraw()
            break
        except Exception:
            root = None
    if root is None:
        pytest.skip("tkinter display not available")
    yield root
    root.destroy()


@pytest.fixture
def app_ui(tk_root):
    """AppUI backed by the session-scoped tk_root."""
    ui = AppUI(root=tk_root)
    yield ui
    for widget in tk_root.winfo_children():
        widget.destroy()


# ---------------------------------------------------------------------------
# label_to_code / code_to_label
# ---------------------------------------------------------------------------

class TestLanguageMapping:
    def test_english_code(self):
        assert label_to_code("English") == "en"

    def test_french_code(self):
        assert label_to_code("French") == "fr"

    def test_czech_code(self):
        assert label_to_code("Czech") == "cs"

    def test_polish_code(self):
        assert label_to_code("Polish") == "pl"

    def test_unknown_label_raises(self):
        with pytest.raises(KeyError):
            label_to_code("Klingon")

    def test_code_to_label_en(self):
        assert code_to_label("en") == "English"

    def test_code_to_label_fr(self):
        assert code_to_label("fr") == "French"

    def test_code_to_label_cs(self):
        assert code_to_label("cs") == "Czech"

    def test_code_to_label_pl(self):
        assert code_to_label("pl") == "Polish"

    def test_unknown_code_raises(self):
        with pytest.raises(KeyError):
            code_to_label("xx")

    def test_roundtrip_label_code_label(self):
        for label, _ in LANGUAGES:
            assert code_to_label(label_to_code(label)) == label

    def test_roundtrip_code_label_code(self):
        for _, code in LANGUAGES:
            assert label_to_code(code_to_label(code)) == code

    def test_default_language_is_english(self):
        assert DEFAULT_LANGUAGE_CODE == "en"

    def test_all_language_labels_present(self):
        assert LANGUAGE_LABELS == ["English", "French", "Czech", "Polish"]


# ---------------------------------------------------------------------------
# server_status_text
# ---------------------------------------------------------------------------

class TestServerStatusText:
    def test_connected(self):
        assert server_status_text(True) == "Server Connected"

    def test_connected_ignores_queue_count(self):
        assert server_status_text(True, queue_count=5) == "Server Connected"

    def test_disconnected_no_queue(self):
        assert server_status_text(False) == "Server Disconnected"

    def test_disconnected_with_queue(self):
        assert server_status_text(False, queue_count=4) == "Offline Queue: 4 segments"

    def test_disconnected_with_one_segment(self):
        assert server_status_text(False, queue_count=1) == "Offline Queue: 1 segments"

    def test_disconnected_zero_queue_shows_disconnected(self):
        assert server_status_text(False, queue_count=0) == "Server Disconnected"


# ---------------------------------------------------------------------------
# audio_status_text
# ---------------------------------------------------------------------------

class TestAudioStatusText:
    def test_connected(self):
        assert audio_status_text(True) == "Audio: Connected"

    def test_disconnected(self):
        assert audio_status_text(False) == "Audio: Disconnected"


# ---------------------------------------------------------------------------
# level_bar_color
# ---------------------------------------------------------------------------

class TestLevelBarColor:
    def test_zero_is_green(self):
        assert level_bar_color(0.0) == "#2ecc71"

    def test_low_is_green(self):
        assert level_bar_color(0.3) == "#2ecc71"

    def test_just_below_0_5_is_green(self):
        assert level_bar_color(0.49) == "#2ecc71"

    def test_0_5_is_orange(self):
        assert level_bar_color(0.5) == "#f39c12"

    def test_mid_range_is_orange(self):
        assert level_bar_color(0.7) == "#f39c12"

    def test_just_below_0_8_is_orange(self):
        assert level_bar_color(0.79) == "#f39c12"

    def test_0_8_is_red(self):
        assert level_bar_color(0.8) == "#e74c3c"

    def test_max_is_red(self):
        assert level_bar_color(1.0) == "#e74c3c"


# ---------------------------------------------------------------------------
# AppUI — widget state tests (require display)
# ---------------------------------------------------------------------------

class TestAppUILanguage:
    def test_default_language_is_english(self, app_ui):
        assert app_ui.get_language() == "en"

    def test_set_language_fr(self, app_ui):
        app_ui._lang_var.set("French")
        assert app_ui.get_language() == "fr"

    def test_set_language_cs(self, app_ui):
        app_ui._lang_var.set("Czech")
        assert app_ui.get_language() == "cs"

    def test_set_language_pl(self, app_ui):
        app_ui._lang_var.set("Polish")
        assert app_ui.get_language() == "pl"

    def test_unknown_code_falls_back_to_english(self, app_ui):
        app_ui._lang_var.set("Klingon")  # not in mapping
        assert app_ui.get_language() == "en"  # falls back to default

    def test_on_language_change_callback_fired(self, app_ui):
        received: List[str] = []
        app_ui.set_on_language_change(received.append)
        app_ui._lang_var.set("French")
        app_ui._on_lang_selected()
        assert received == ["fr"]

    def test_language_callback_receives_correct_code_for_cs(self, app_ui):
        received: List[str] = []
        app_ui.set_on_language_change(received.append)
        app_ui._lang_var.set("Czech")
        app_ui._on_lang_selected()
        assert received == ["cs"]

    def test_no_callback_registered_does_not_raise(self, app_ui):
        app_ui._lang_var.set("Polish")
        app_ui._on_lang_selected()  # must not raise


class TestAppUITranscript:
    def test_set_transcript_sets_text(self, app_ui):
        app_ui._set_transcript("Hello world")
        content = app_ui._transcript_text.get("1.0", "end-1c")
        assert content == "Hello world"

    def test_set_transcript_replaces_previous(self, app_ui):
        app_ui._set_transcript("First")
        app_ui._set_transcript("Second")
        content = app_ui._transcript_text.get("1.0", "end-1c")
        assert content == "Second"

    def test_set_transcript_empty(self, app_ui):
        app_ui._set_transcript("Something")
        app_ui._set_transcript("")
        content = app_ui._transcript_text.get("1.0", "end-1c")
        assert content == ""

    def test_set_transcript_non_ascii(self, app_ui):
        text = "Příliš žluťoučký kůň"
        app_ui._set_transcript(text)
        content = app_ui._transcript_text.get("1.0", "end-1c")
        assert content == text


class TestAppUIAudioStatus:
    def test_audio_connected(self, app_ui):
        app_ui._set_audio_status(True)
        assert "Connected" in app_ui._audio_label.cget("text")

    def test_audio_disconnected(self, app_ui):
        app_ui._set_audio_status(False)
        assert "Disconnected" in app_ui._audio_label.cget("text")


class TestAppUIServerStatus:
    def test_server_connected_text(self, app_ui):
        app_ui._set_server_status(True, 0)
        assert app_ui._server_label.cget("text") == "Server Connected"

    def test_server_disconnected_text(self, app_ui):
        app_ui._set_server_status(False, 0)
        assert app_ui._server_label.cget("text") == "Server Disconnected"

    def test_offline_queue_text(self, app_ui):
        app_ui._set_server_status(False, 7)
        assert app_ui._server_label.cget("text") == "Offline Queue: 7 segments"

    def test_server_connected_fg_green(self, app_ui):
        app_ui._set_server_status(True, 0)
        assert app_ui._server_label.cget("fg") == "#27ae60"

    def test_server_disconnected_fg_red(self, app_ui):
        app_ui._set_server_status(False, 0)
        assert app_ui._server_label.cget("fg") == "#c0392b"


class TestAppUIRestart:
    def test_restart_callback_invoked(self, app_ui):
        called: List[bool] = []
        app_ui.set_on_restart(lambda: called.append(True))
        app_ui._on_restart_clicked()
        assert called == [True]

    def test_no_restart_callback_does_not_raise(self, app_ui):
        app_ui._on_restart_clicked()  # must not raise


class TestAppUILevelBar:
    def test_level_clamps_below_zero(self, app_ui):
        app_ui._set_level(-0.5)
        coords = app_ui._level_canvas.coords(app_ui._level_bar)
        assert coords[2] == 0  # width = 0

    def test_level_clamps_above_one(self, app_ui):
        app_ui._set_level(2.0)
        coords = app_ui._level_canvas.coords(app_ui._level_bar)
        assert coords[2] == app_ui._LEVEL_BAR_W

    def test_level_half(self, app_ui):
        app_ui._set_level(0.5)
        coords = app_ui._level_canvas.coords(app_ui._level_bar)
        assert coords[2] == app_ui._LEVEL_BAR_W // 2
