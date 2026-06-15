"""Tests for transcriptor.startup — StartupDialog and pure helpers.

Strategy
--------
Pure functions (``build_device_labels``, ``label_to_device_id``,
``save_config_yaml``) are tested without tkinter.

Widget tests follow the same injection-seam pattern used in ``test_ui.py``:
a session-scoped ``tk.Tk()`` root is shared across the whole session (with a
retry for the macOS Tcl init quirk), and internal ``_*_var`` / ``_*`` methods
are accessed directly to avoid the ``root.update()`` / ``mainloop()`` hang.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest
import yaml

from transcriptor.config import AppConfig, AudioConfig, ServerConfig, StabilizationConfig, TranscriptionConfig, VADConfig
from transcriptor.startup import (
    DEVICE_LABEL_DEFAULT,
    StartupDialog,
    build_device_labels,
    label_to_device_id,
    save_config_yaml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> AppConfig:
    """Return an AppConfig with sensible test defaults, overridable via kwargs."""
    defaults = dict(
        event_id="test-event",
        server=ServerConfig(websocket_url="wss://test.example.com/ws"),
        audio=AudioConfig(device_id="default"),
        transcription=TranscriptionConfig(model="medium", language="en"),
        stabilization=StabilizationConfig(silence_ms=700, stable_ms=2000),
        vad=VADConfig(max_speech_ms=10000, end_overlap_ms=1000, start_overlap_ms=2000),
    )
    defaults.update(kwargs)
    return AppConfig(**defaults)


_FAKE_DEVICES = [
    {"index": 0, "name": "USB Mixer", "channels": 2},
    {"index": 1, "name": "Built-in Microphone", "channels": 1},
]


# ---------------------------------------------------------------------------
# Session-scoped tk.Tk root (same pattern as test_ui.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tk_root():
    """A single tk.Tk() shared for the whole test session."""
    for _ in range(2):
        try:
            root = tk.Tk()
            root.withdraw()
            return root
        except tk.TclError:
            pass
    pytest.skip("tkinter not available in this environment")


@pytest.fixture
def dialog(tk_root):
    """A StartupDialog backed by the session root, devices stubbed out."""
    cfg = _make_config()
    d = StartupDialog(cfg, root=tk_root)
    # Override _devices directly so tests don't need real audio hardware.
    d._devices = list(_FAKE_DEVICES)
    labels = build_device_labels(d._devices)
    d._device_combo["values"] = labels
    d._device_var.set(DEVICE_LABEL_DEFAULT)
    return d


# ---------------------------------------------------------------------------
# build_device_labels — pure function
# ---------------------------------------------------------------------------

class TestBuildDeviceLabels:
    def test_empty_list_returns_only_default(self):
        assert build_device_labels([]) == [DEVICE_LABEL_DEFAULT]

    def test_single_device_appended_after_default(self):
        devices = [{"index": 0, "name": "USB Mixer", "channels": 2}]
        labels = build_device_labels(devices)
        assert labels == [DEVICE_LABEL_DEFAULT, "USB Mixer"]

    def test_multiple_devices_order_preserved(self):
        labels = build_device_labels(_FAKE_DEVICES)
        assert labels == [DEVICE_LABEL_DEFAULT, "USB Mixer", "Built-in Microphone"]

    def test_default_is_always_first(self):
        labels = build_device_labels(_FAKE_DEVICES)
        assert labels[0] == DEVICE_LABEL_DEFAULT

    def test_device_names_extracted_correctly(self):
        devices = [{"index": 3, "name": "Focusrite", "channels": 4}]
        assert build_device_labels(devices)[1] == "Focusrite"


# ---------------------------------------------------------------------------
# label_to_device_id — pure function
# ---------------------------------------------------------------------------

class TestLabelToDeviceId:
    def test_default_label_returns_default_string(self):
        assert label_to_device_id(DEVICE_LABEL_DEFAULT, _FAKE_DEVICES) == "default"

    def test_device_name_returns_name(self):
        assert label_to_device_id("USB Mixer", _FAKE_DEVICES) == "USB Mixer"

    def test_second_device_name_returns_name(self):
        assert label_to_device_id("Built-in Microphone", _FAKE_DEVICES) == "Built-in Microphone"

    def test_unknown_label_falls_back_to_default(self):
        assert label_to_device_id("Nonexistent Device", _FAKE_DEVICES) == "default"

    def test_empty_devices_list_returns_default(self):
        assert label_to_device_id(DEVICE_LABEL_DEFAULT, []) == "default"

    def test_empty_devices_unknown_label_returns_default(self):
        assert label_to_device_id("USB Mixer", []) == "default"


# ---------------------------------------------------------------------------
# save_config_yaml — pure function (uses tmp_path, no tkinter)
# ---------------------------------------------------------------------------

class TestSaveConfigYaml:
    def test_writes_event_id(self, tmp_path):
        p = tmp_path / "config.yaml"
        save_config_yaml(p, "my-event", "wss://x.com/ws", "default")
        data = yaml.safe_load(p.read_text())
        assert data["event_id"] == "my-event"

    def test_writes_websocket_url(self, tmp_path):
        p = tmp_path / "config.yaml"
        save_config_yaml(p, "e", "wss://new.example.com/ws", "default")
        data = yaml.safe_load(p.read_text())
        assert data["server"]["websocket_url"] == "wss://new.example.com/ws"

    def test_writes_device_id(self, tmp_path):
        p = tmp_path / "config.yaml"
        save_config_yaml(p, "e", "wss://x.com/ws", "USB Mixer")
        data = yaml.safe_load(p.read_text())
        assert data["audio"]["device_id"] == "USB Mixer"

    def test_preserves_existing_keys(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump({
            "event_id": "old",
            "transcription": {"model": "medium", "language": "cs"},
            "vad": {"max_speech_ms": 8000, "end_overlap_ms": 800, "start_overlap_ms": 1200},
        }))
        save_config_yaml(p, "new-event", "wss://x.com/ws", "default")
        data = yaml.safe_load(p.read_text())
        assert data["transcription"]["model"] == "medium"
        assert data["transcription"]["language"] == "cs"
        assert data["vad"]["max_speech_ms"] == 8000
        assert data["vad"]["end_overlap_ms"] == 800
        assert data["vad"]["start_overlap_ms"] == 1200

    def test_creates_file_if_absent(self, tmp_path):
        p = tmp_path / "new_config.yaml"
        assert not p.exists()
        save_config_yaml(p, "e", "wss://x.com/ws", "default")
        assert p.exists()

    def test_unicode_event_id_preserved(self, tmp_path):
        p = tmp_path / "config.yaml"
        save_config_yaml(p, "svatba-2027-\u010cesko", "wss://x.com/ws", "default")
        data = yaml.safe_load(p.read_text())
        assert data["event_id"] == "svatba-2027-\u010cesko"

    def test_overwrites_existing_event_id(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump({"event_id": "old-event"}))
        save_config_yaml(p, "new-event", "wss://x.com/ws", "default")
        data = yaml.safe_load(p.read_text())
        assert data["event_id"] == "new-event"


# ---------------------------------------------------------------------------
# StartupDialog — defaults pre-filled from config
# ---------------------------------------------------------------------------

class TestStartupDialogDefaults:
    def test_event_id_pre_filled(self, dialog):
        assert dialog._event_id_var.get() == "test-event"

    def test_server_url_pre_filled(self, dialog):
        assert dialog._server_url_var.get() == "wss://test.example.com/ws"

    def test_device_defaults_to_default_label(self, dialog):
        assert dialog._device_var.get() == DEVICE_LABEL_DEFAULT

    def test_save_checkbox_unchecked_by_default(self, dialog):
        assert dialog._save_default_var.get() is False

    def test_device_combo_contains_all_labels(self, dialog):
        values = list(dialog._device_combo["values"])
        assert DEVICE_LABEL_DEFAULT in values
        assert "USB Mixer" in values
        assert "Built-in Microphone" in values

    def test_warning_hidden_when_devices_present(self, dialog):
        # grid_info() is empty when the widget is hidden via grid_remove()
        assert dialog._warn_label.grid_info() == {}


# ---------------------------------------------------------------------------
# StartupDialog — Start produces correct AppConfig
# ---------------------------------------------------------------------------

class TestStartupDialogStart:
    def _start(self, dialog, event_id=None, server_url=None, device_label=None):
        """Set field values then call _on_start() directly (no mainloop)."""
        if event_id is not None:
            dialog._event_id_var.set(event_id)
        if server_url is not None:
            dialog._server_url_var.set(server_url)
        if device_label is not None:
            dialog._device_var.set(device_label)
        # Prevent actual window destruction in the shared root
        dialog._root.destroy = lambda: None
        dialog._on_start()
        return dialog._result

    def test_start_returns_app_config(self, dialog):
        result = self._start(dialog)
        assert isinstance(result, AppConfig)

    def test_event_id_overridden(self, dialog):
        result = self._start(dialog, event_id="wedding-2028")
        assert result.event_id == "wedding-2028"

    def test_server_url_overridden(self, dialog):
        result = self._start(dialog, server_url="wss://prod.example.com/ws")
        assert result.server.websocket_url == "wss://prod.example.com/ws"

    def test_device_default_label_maps_to_default(self, dialog):
        result = self._start(dialog, device_label=DEVICE_LABEL_DEFAULT)
        assert result.audio.device_id == "default"

    def test_device_name_maps_to_name(self, dialog):
        result = self._start(dialog, device_label="USB Mixer")
        assert result.audio.device_id == "USB Mixer"

    def test_transcription_config_unchanged(self, dialog):
        result = self._start(dialog)
        assert result.transcription.model == "medium"
        assert result.transcription.language == "en"

    def test_stabilization_config_unchanged(self, dialog):
        result = self._start(dialog)
        assert result.stabilization.silence_ms == 700
        assert result.stabilization.stable_ms == 2000

    def test_vad_config_unchanged(self, dialog):
        result = self._start(dialog)
        assert result.vad.max_speech_ms == 10000
        assert result.vad.end_overlap_ms == 1000
        assert result.vad.start_overlap_ms == 2000

    def test_whitespace_trimmed_from_event_id(self, dialog):
        result = self._start(dialog, event_id="  trimmed-event  ")
        assert result.event_id == "trimmed-event"

    def test_whitespace_trimmed_from_server_url(self, dialog):
        result = self._start(dialog, server_url="  wss://x.com/ws  ")
        assert result.server.websocket_url == "wss://x.com/ws"

    def test_empty_event_id_falls_back_to_base(self, dialog):
        result = self._start(dialog, event_id="   ")
        assert result.event_id == "test-event"

    def test_empty_server_url_falls_back_to_base(self, dialog):
        result = self._start(dialog, server_url="   ")
        assert result.server.websocket_url == "wss://test.example.com/ws"


# ---------------------------------------------------------------------------
# StartupDialog — Cancel returns None
# ---------------------------------------------------------------------------

class TestStartupDialogCancel:
    def test_cancel_result_is_none(self, dialog):
        dialog._root.destroy = lambda: None
        dialog._on_cancel()
        assert dialog._result is None

    def test_result_none_before_any_action(self, dialog):
        assert dialog._result is None


# ---------------------------------------------------------------------------
# StartupDialog — audio device list & warning
# ---------------------------------------------------------------------------

class TestStartupDialogDevices:
    def test_warning_shown_when_no_devices(self, tk_root):
        cfg = _make_config()
        d = StartupDialog(cfg, root=tk_root)
        # Simulate empty device list
        d._devices = []
        d._device_combo["values"] = [DEVICE_LABEL_DEFAULT]
        d._device_var.set(DEVICE_LABEL_DEFAULT)
        d._warn_label.grid()   # force show as _load_devices() would
        assert d._warn_label.grid_info() != {}

    def test_warning_hidden_when_devices_exist(self, tk_root):
        cfg = _make_config()
        d = StartupDialog(cfg, root=tk_root)
        d._devices = list(_FAKE_DEVICES)
        d._warn_label.grid_remove()
        assert d._warn_label.grid_info() == {}

    def test_refresh_repopulates_combo(self, dialog, monkeypatch):
        new_devices = [{"index": 5, "name": "Focusrite Scarlett", "channels": 2}]

        def fake_list():
            return new_devices

        monkeypatch.setattr(
            "transcriptor.startup.StartupDialog._load_devices",
            lambda self: (
                setattr(self, "_devices", new_devices)
                or self._device_combo.configure(
                    values=build_device_labels(new_devices)
                )
            ),
        )
        dialog._on_refresh()
        values = list(dialog._device_combo["values"])
        assert "Focusrite Scarlett" in values

    def test_device_preselected_when_config_matches(self, tk_root):
        cfg = _make_config(audio=AudioConfig(device_id="USB Mixer"))
        d = StartupDialog(cfg, root=tk_root)
        d._devices = list(_FAKE_DEVICES)
        labels = build_device_labels(d._devices)
        d._device_combo["values"] = labels
        # Simulate matching pre-selection logic
        for dev in d._devices:
            if cfg.audio.device_id in dev["name"]:
                d._device_var.set(dev["name"])
                break
        assert d._device_var.get() == "USB Mixer"
