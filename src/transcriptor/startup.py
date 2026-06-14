"""Startup configuration dialog — shown once before the main pipeline starts.

Lets the operator review and override the three settings that vary between
events (event ID, server URL, audio device) without editing ``config.yaml``.
All other settings (model, VAD tuning, stabilization) are carried forward
unchanged from the loaded configuration.

Layout::

    ┌──────────────────────────────────────────────┐
    │        Wedding Transcriptor — Setup          │
    │                                              │
    │  Event ID                                    │
    │  [wedding-2027                             ] │
    │                                              │
    │  Server URL                                  │
    │  [wss://translate.example.com/ws           ] │
    │                                              │
    │  Audio Device                                │
    │  [Default                               ▾ ] [Refresh]
    │  ⚠ No audio devices found. Using default.   │
    │                                              │
    │  ☐ Save as default (update config.yaml)     │
    │                                              │
    │                    [Cancel]      [Start →]   │
    └──────────────────────────────────────────────┘

Injection seam
--------------
Pass ``root=<tk.Tk instance>`` to supply a pre-existing root window (used in
tests).  When omitted a new ``tk.Tk()`` is created.

Usage::

    dialog = StartupDialog(load_config())
    config = dialog.run()   # blocks until Start or Cancel
    if config is None:
        return              # operator cancelled
    Application(config=config).run()
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

import yaml

from transcriptor.config import (
    AppConfig,
    AudioConfig,
    CONFIG_FILE,
    ServerConfig,
)

log = logging.getLogger(__name__)

# Label shown in the audio-device dropdown for "use system default".
DEVICE_LABEL_DEFAULT = "Default"

# Warning shown when no input devices are detected.
_NO_DEVICES_WARNING = "\u26a0 No audio devices found. Using default."

_LABEL_FONT = ("Helvetica", 11, "bold")
_BODY_FONT = ("Helvetica", 11)
_WARN_FONT = ("Helvetica", 10)
_TITLE_FONT = ("Helvetica", 14, "bold")


# ---------------------------------------------------------------------------
# Pure helper functions (testable without tkinter)
# ---------------------------------------------------------------------------

def build_device_labels(devices: list[dict]) -> list[str]:
    """Return a dropdown label list from *devices*.

    The first entry is always :data:`DEVICE_LABEL_DEFAULT`; subsequent entries
    are the ``"name"`` field of each device dict, in the order given.

    Parameters
    ----------
    devices:
        List of device dicts as returned by
        :func:`~transcriptor.audio.list_input_devices`.
    """
    return [DEVICE_LABEL_DEFAULT] + [d["name"] for d in devices]


def label_to_device_id(label: str, devices: list[dict]) -> str:
    """Convert a selected dropdown *label* to a ``device_id`` config string.

    * ``DEVICE_LABEL_DEFAULT`` → ``"default"``
    * Any device name → that name (used for partial matching in
      :func:`~transcriptor.audio._resolve_device`)
    * Unknown label → ``"default"`` (safe fallback)
    """
    if label == DEVICE_LABEL_DEFAULT:
        return "default"
    for d in devices:
        if d["name"] == label:
            return d["name"]
    return "default"


def save_config_yaml(
    path: Path,
    event_id: str,
    websocket_url: str,
    device_id: str,
) -> None:
    """Persist the three operator-facing fields to *path* (config.yaml).

    Loads the existing file (or starts from an empty dict if absent), patches
    only the three specified keys, and writes back.  All other keys
    (``model``, ``vad``, ``stabilization``, …) are preserved.

    Parameters
    ----------
    path:
        Filesystem path to ``config.yaml``.
    event_id:
        New value for the top-level ``event_id`` key.
    websocket_url:
        New value for ``server.websocket_url``.
    device_id:
        New value for ``audio.device_id``.
    """
    try:
        data: dict = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    except Exception:
        data = {}

    data["event_id"] = event_id
    data.setdefault("server", {})["websocket_url"] = websocket_url
    data.setdefault("audio", {})["device_id"] = device_id

    path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    log.info("Config saved to %s", path)


# ---------------------------------------------------------------------------
# StartupDialog
# ---------------------------------------------------------------------------

class StartupDialog:
    """Pre-flight configuration window shown before the main pipeline starts.

    Parameters
    ----------
    config:
        Application configuration loaded from ``config.yaml``; used to
        pre-fill the form fields.
    root:
        Optional pre-existing ``tk.Tk()`` instance (injection seam for tests).
        When ``None`` a new root window is created.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        root: Optional[tk.Tk] = None,
    ) -> None:
        self._base_config = config
        self._root: tk.Tk = root if root is not None else tk.Tk()
        self._root.title("Wedding Transcriptor — Setup")
        self._root.resizable(False, False)

        # Runtime state
        self._result: Optional[AppConfig] = None
        self._devices: list[dict] = []

        # StringVars — created before _build_ui so tests can read them
        self._event_id_var = tk.StringVar(value=config.event_id)
        self._server_url_var = tk.StringVar(value=config.server.websocket_url)
        self._device_var = tk.StringVar()
        self._save_default_var = tk.BooleanVar(value=False)

        self._build_ui()
        # Populate device list (may be empty on headless/test machines)
        self._load_devices()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Optional[AppConfig]:
        """Show the dialog and block until the operator clicks Start or Cancel.

        Returns
        -------
        AppConfig
            New configuration with the operator's overrides applied, or
            ``None`` if the operator cancelled.
        """
        self._root.mainloop()
        return self._result

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._root.configure(padx=20, pady=16)

        # ── Title ─────────────────────────────────────────────────────
        tk.Label(
            self._root,
            text="Wedding Transcriptor",
            font=_TITLE_FONT,
        ).grid(row=0, column=0, columnspan=3, pady=(0, 16), sticky="w")

        # ── Event ID ──────────────────────────────────────────────────
        tk.Label(self._root, text="Event ID", font=_LABEL_FONT).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )
        tk.Entry(
            self._root,
            textvariable=self._event_id_var,
            font=_BODY_FONT,
            width=46,
        ).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        # ── Server URL ────────────────────────────────────────────────
        tk.Label(self._root, text="Server URL", font=_LABEL_FONT).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )
        tk.Entry(
            self._root,
            textvariable=self._server_url_var,
            font=_BODY_FONT,
            width=46,
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        # ── Audio Device ──────────────────────────────────────────────
        tk.Label(self._root, text="Audio Device", font=_LABEL_FONT).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )

        self._device_combo = ttk.Combobox(
            self._root,
            textvariable=self._device_var,
            state="readonly",
            width=38,
            font=_BODY_FONT,
        )
        self._device_combo.grid(row=6, column=0, columnspan=2, sticky="ew")

        tk.Button(
            self._root,
            text="Refresh",
            command=self._on_refresh,
        ).grid(row=6, column=2, padx=(6, 0), sticky="w")

        self._warn_label = tk.Label(
            self._root,
            text=_NO_DEVICES_WARNING,
            font=_WARN_FONT,
            fg="#c0392b",
            anchor="w",
        )
        self._warn_label.grid(row=7, column=0, columnspan=3, sticky="w")
        self._warn_label.grid_remove()   # hidden until needed

        # ── Save as default ───────────────────────────────────────────
        tk.Checkbutton(
            self._root,
            text="Save as default (update config.yaml)",
            variable=self._save_default_var,
            font=_BODY_FONT,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 0))

        # ── Buttons ───────────────────────────────────────────────────
        btn_frame = tk.Frame(self._root)
        btn_frame.grid(row=9, column=0, columnspan=3, sticky="e", pady=(16, 0))

        tk.Button(
            btn_frame,
            text="Cancel",
            width=10,
            command=self._on_cancel,
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_frame,
            text="Start \u2192",
            width=10,
            command=self._on_start,
            default=tk.ACTIVE,
        ).pack(side=tk.LEFT)

        # Bind Enter key to Start
        self._root.bind("<Return>", lambda _e: self._on_start())
        self._root.bind("<Escape>", lambda _e: self._on_cancel())

    # ------------------------------------------------------------------
    # Device list management
    # ------------------------------------------------------------------

    def _load_devices(self) -> None:
        """Populate the audio device dropdown from the host's input devices."""
        try:
            from transcriptor.audio import list_input_devices  # deferred import
            self._devices = list_input_devices()
        except Exception as exc:
            log.warning("Could not enumerate audio devices: %s", exc)
            self._devices = []

        labels = build_device_labels(self._devices)
        self._device_combo["values"] = labels

        # Try to pre-select the device from config
        current_id = self._base_config.audio.device_id
        selected = DEVICE_LABEL_DEFAULT
        if current_id != "default":
            for d in self._devices:
                if current_id in d["name"] or d["name"] in current_id:
                    selected = d["name"]
                    break
        self._device_var.set(selected)

        # Show/hide warning
        if self._devices:
            self._warn_label.grid_remove()
        else:
            self._warn_label.grid()

    def _on_refresh(self) -> None:
        """Re-enumerate audio devices and repopulate the dropdown."""
        self._load_devices()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        event_id = self._event_id_var.get().strip() or self._base_config.event_id
        server_url = (
            self._server_url_var.get().strip()
            or self._base_config.server.websocket_url
        )
        device_id = label_to_device_id(self._device_var.get(), self._devices)

        self._result = AppConfig(
            event_id=event_id,
            server=ServerConfig(websocket_url=server_url),
            audio=AudioConfig(device_id=device_id),
            transcription=self._base_config.transcription,
            stabilization=self._base_config.stabilization,
            vad=self._base_config.vad,
        )

        if self._save_default_var.get():
            try:
                save_config_yaml(CONFIG_FILE, event_id, server_url, device_id)
            except Exception as exc:
                log.error("Failed to save config.yaml: %s", exc)

        log.info(
            "Startup: event_id=%r server=%r device=%r save_default=%s",
            event_id, server_url, device_id, self._save_default_var.get(),
        )
        self._root.destroy()

    def _on_cancel(self) -> None:
        log.info("Startup cancelled by operator")
        self._result = None
        self._root.destroy()
