"""Operator UI — FR-002, spec §5.

Single-page tkinter desktop application.

Layout (top to bottom)
-----------------------
* **Audio status** row — "Connected" / "Disconnected" label + level meter
  (canvas bar that fills proportionally to signal level, colour-coded
  green → yellow → red).
* **Language selector** — labelled ``ttk.Combobox`` (English / French /
  Czech / Polish).
* **Current speech** — large read-only ``tk.Text`` box that shows the
  live in-progress transcript (updated by the pipeline thread).
* **Server status** bar — "Server Connected" or "Offline Queue: N segments".
* **Restart** button — re-initialise the transcription engine after a crash
  (wired by ``set_on_restart``).

All updates from background threads must go through :meth:`update_*`
methods, which schedule the actual widget update on the Tk event loop via
``root.after(0, …)`` — tkinter widgets are not thread-safe.

Injection seam
--------------
Pass ``root=<tk.Tk instance>`` to the constructor to supply a pre-existing
root window (used in tests that provide a mock).  When omitted a real
``tk.Tk()`` is created.

Usage::

    ui = AppUI()
    ui.set_on_language_change(lambda code: pipeline.set_language(code))
    ui.set_on_restart(pipeline.restart_transcriber)
    # pipeline calls ui.update_* from its own thread
    ui.run()  # blocks until window is closed
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from transcriptor.startup import build_device_labels, label_to_device_id, DEVICE_LABEL_DEFAULT

# ---------------------------------------------------------------------------
# Language table
# ---------------------------------------------------------------------------

#: Ordered list of (display label, ISO-639-1 code) pairs for the MVP.
LANGUAGES: list[tuple[str, str]] = [
    ("English", "en"),
    ("French", "fr"),
    ("Czech", "cs"),
    ("Polish", "pl"),
]

_LABEL_TO_CODE: dict[str, str] = {label: code for label, code in LANGUAGES}
_CODE_TO_LABEL: dict[str, str] = {code: label for label, code in LANGUAGES}

LANGUAGE_LABELS: list[str] = [label for label, _ in LANGUAGES]
DEFAULT_LANGUAGE_CODE: str = LANGUAGES[0][1]  # "en"


# ---------------------------------------------------------------------------
# Pure helper functions (testable without tkinter)
# ---------------------------------------------------------------------------

def label_to_code(label: str) -> str:
    """Return the ISO-639-1 code for a display *label*.

    Raises ``KeyError`` for unknown labels.
    """
    return _LABEL_TO_CODE[label]


def code_to_label(code: str) -> str:
    """Return the display label for an ISO-639-1 *code*.

    Raises ``KeyError`` for unknown codes.
    """
    return _CODE_TO_LABEL[code]


def server_status_text(connected: bool, queue_count: int = 0) -> str:
    """Return the server-status bar string per spec §5."""
    if connected:
        return "Server Connected"
    if queue_count:
        return f"Offline Queue: {queue_count} segments"
    return "Server Disconnected"


def audio_status_text(connected: bool) -> str:
    """Return the audio-status label string."""
    return "Audio: Connected" if connected else "Audio: Disconnected"


def level_bar_color(level: float) -> str:
    """Return a tkinter colour string for *level* in ``[0.0, 1.0]``."""
    if level < 0.5:
        return "#2ecc71"   # green
    if level < 0.8:
        return "#f39c12"   # orange/yellow
    return "#e74c3c"       # red


# ---------------------------------------------------------------------------
# AppUI
# ---------------------------------------------------------------------------

class AppUI:
    """Operator-facing tkinter UI window.

    Parameters
    ----------
    title:
        Window title shown in the OS title bar.
    root:
        Optional pre-existing ``tk.Tk()`` instance.  When ``None`` (the
        default) a new root window is created by :meth:`__init__`.
    """

    _LEVEL_BAR_W = 200
    _LEVEL_BAR_H = 18
    _TRANSCRIPT_FONT = ("Helvetica", 14)
    _STATUS_FONT = ("Helvetica", 11)
    _LABEL_FONT = ("Helvetica", 11, "bold")

    def __init__(
        self,
        title: str = "Wedding Transcriptor",
        *,
        event_id: str = "",
        devices: Optional[list[dict]] = None,
        device_id: str = "default",
        root: Optional[tk.Tk] = None,
    ) -> None:
        self._root: tk.Tk = root if root is not None else tk.Tk()
        self._root.title(f"{title}: {event_id}" if event_id else title)
        self._root.resizable(True, True)
        self._event_id: str = event_id
        self._devices: list[dict] = devices if devices is not None else []
        self._current_device_id: str = device_id

        # Callbacks registered by the caller
        self._on_language_change: Optional[Callable[[str], None]] = None
        self._on_restart: Optional[Callable[[], None]] = None
        self._on_device_change: Optional[Callable[[str], None]] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API — called from pipeline threads
    # ------------------------------------------------------------------

    def update_audio_level(self, level: float) -> None:
        """Set the level meter to *level* (0.0–1.0)."""
        level = max(0.0, min(1.0, float(level)))
        self._root.after(0, self._set_level, level)

    def update_audio_status(self, connected: bool) -> None:
        """Update the audio connection indicator."""
        self._root.after(0, self._set_audio_status, connected)

    def update_transcript(self, text: str) -> None:
        """Replace the live transcript display with *text*."""
        self._root.after(0, self._set_transcript, text)

    def update_server_status(self, connected: bool, queue_count: int = 0) -> None:
        """Update the server connection / offline-queue status bar."""
        self._root.after(0, self._set_server_status, connected, queue_count)

    # ------------------------------------------------------------------
    # Public API — setters / getters (safe from any thread if tkinter
    # StringVar access is protected by after(); read-only from caller)
    # ------------------------------------------------------------------

    def get_language(self) -> str:
        """Return the currently selected ISO-639-1 language code."""
        label = self._lang_var.get()
        return _LABEL_TO_CODE.get(label, DEFAULT_LANGUAGE_CODE)

    def set_language(self, code: str) -> None:
        """Programmatically select a language by its ISO-639-1 code."""
        label = _CODE_TO_LABEL.get(code, LANGUAGE_LABELS[0])
        self._root.after(0, self._lang_var.set, label)

    def set_on_language_change(self, callback: Callable[[str], None]) -> None:
        """Register *callback(code)* invoked when the operator changes language."""
        self._on_language_change = callback

    def set_on_restart(self, callback: Callable[[], None]) -> None:
        """Register *callback()* invoked when the operator clicks Restart."""
        self._on_restart = callback

    def set_on_device_change(self, callback: Callable[[str], None]) -> None:
        """Register *callback(device_id)* invoked when the operator selects a device."""
        self._on_device_change = callback

    def get_device(self) -> str:
        """Return the currently selected device_id string."""
        return label_to_device_id(self._device_var.get(), self._devices)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Enter the tkinter main loop (blocking until the window is closed)."""
        self._root.mainloop()

    def stop(self) -> None:
        """Request that the window be destroyed (safe to call from any thread)."""
        self._root.after(0, self._root.destroy)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._root.configure(padx=12, pady=10)

        # ── Audio status row ──────────────────────────────────────────
        audio_frame = tk.Frame(self._root)
        audio_frame.pack(fill=tk.X, pady=(0, 6))

        self._audio_label = tk.Label(
            audio_frame,
            text=audio_status_text(False),
            font=self._STATUS_FONT,
            anchor="w",
        )
        self._audio_label.pack(side=tk.LEFT)

        self._level_canvas = tk.Canvas(
            audio_frame,
            width=self._LEVEL_BAR_W,
            height=self._LEVEL_BAR_H,
            bg="#ecf0f1",
            highlightthickness=1,
            highlightbackground="#bdc3c7",
        )
        self._level_canvas.pack(side=tk.RIGHT)
        self._level_bar = self._level_canvas.create_rectangle(
            0, 0, 0, self._LEVEL_BAR_H, fill="#2ecc71", outline=""
        )

        # ── Language selector ─────────────────────────────────────────
        lang_frame = tk.Frame(self._root)
        lang_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(lang_frame, text="Language:", font=self._LABEL_FONT).pack(
            side=tk.LEFT, padx=(0, 8)
        )

        self._lang_var = tk.StringVar(value=LANGUAGE_LABELS[0])
        self._lang_combo = ttk.Combobox(
            lang_frame,
            textvariable=self._lang_var,
            values=LANGUAGE_LABELS,
            state="readonly",
            width=12,
        )
        self._lang_combo.pack(side=tk.LEFT)
        self._lang_combo.bind("<<ComboboxSelected>>", self._on_lang_selected)

        # ── Audio device selector ─────────────────────────────────────
        device_frame = tk.Frame(self._root)
        device_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(device_frame, text="Audio Device:", font=self._LABEL_FONT).pack(
            side=tk.LEFT, padx=(0, 8)
        )

        self._device_var = tk.StringVar()
        self._device_combo = ttk.Combobox(
            device_frame,
            textvariable=self._device_var,
            state="readonly",
            width=30,
        )
        self._device_combo.pack(side=tk.LEFT)
        self._device_combo.bind("<<ComboboxSelected>>", self._on_device_selected)

        tk.Button(
            device_frame,
            text="Refresh",
            command=self._on_device_refresh,
        ).pack(side=tk.LEFT, padx=(6, 0))

        self._populate_device_combo(self._devices, self._current_device_id)

        # ── Current speech ────────────────────────────────────────────
        tk.Label(self._root, text="Current Speech", font=self._LABEL_FONT).pack(
            anchor="w"
        )

        self._transcript_text = tk.Text(
            self._root,
            font=self._TRANSCRIPT_FONT,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bg="#fdfefe",
        )
        self._transcript_text.pack(fill=tk.BOTH, expand=True, pady=(2, 8))

        # ── Restart button ────────────────────────────────────────────
        self._restart_btn = tk.Button(
            self._root,
            text="Restart Transcriber",
            command=self._on_restart_clicked,
            state=tk.DISABLED,
        )
        self._restart_btn.pack(anchor="e", pady=(0, 4))

        # ── Server status bar ─────────────────────────────────────────
        self._server_label = tk.Label(
            self._root,
            text=server_status_text(False),
            font=self._STATUS_FONT,
            anchor="w",
            relief=tk.SUNKEN,
            bd=1,
            padx=4,
        )
        self._server_label.pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------
    # Internal update implementations (always called on the Tk thread)
    # ------------------------------------------------------------------

    def _set_level(self, level: float) -> None:
        level = max(0.0, min(1.0, float(level)))
        width = int(level * self._LEVEL_BAR_W)
        color = level_bar_color(level)
        self._level_canvas.coords(self._level_bar, 0, 0, width, self._LEVEL_BAR_H)
        self._level_canvas.itemconfigure(self._level_bar, fill=color)

    def _set_audio_status(self, connected: bool) -> None:
        self._audio_label.configure(text=audio_status_text(connected))
        # Enable/disable restart button when audio is disconnected
        self._restart_btn.configure(
            state=tk.NORMAL if self._on_restart is not None else tk.DISABLED
        )

    def _set_transcript(self, text: str) -> None:
        self._transcript_text.configure(state=tk.NORMAL)
        self._transcript_text.delete("1.0", tk.END)
        self._transcript_text.insert(tk.END, text)
        self._transcript_text.configure(state=tk.DISABLED)

    def _set_server_status(self, connected: bool, queue_count: int) -> None:
        text = server_status_text(connected, queue_count)
        fg = "#27ae60" if connected else "#c0392b"
        self._server_label.configure(text=text, fg=fg)

    def _on_lang_selected(self, _event=None) -> None:
        label = self._lang_var.get()
        code = _LABEL_TO_CODE.get(label, DEFAULT_LANGUAGE_CODE)
        if self._on_language_change is not None:
            self._on_language_change(code)

    def _on_device_selected(self, _event=None) -> None:
        device_id = label_to_device_id(self._device_var.get(), self._devices)
        if self._on_device_change is not None:
            self._on_device_change(device_id)

    def _on_device_refresh(self) -> None:
        try:
            from transcriptor.audio import list_input_devices
            self._devices = list_input_devices()
        except Exception:
            self._devices = []
        self._populate_device_combo(self._devices, self._current_device_id)

    def _populate_device_combo(self, devices: list[dict], device_id: str) -> None:
        """Repopulate the device combobox and pre-select *device_id*."""
        labels = build_device_labels(devices)
        self._device_combo.configure(values=labels)
        # Find matching label for the current device_id
        selected = DEVICE_LABEL_DEFAULT
        for d in devices:
            if d["name"] == device_id or str(d["index"]) == device_id:
                selected = d["name"]
                break
        self._device_var.set(selected)

    def _on_restart_clicked(self) -> None:
        if self._on_restart is not None:
            self._on_restart()
