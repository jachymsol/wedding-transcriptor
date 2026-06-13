"""Logging setup for the transcription client."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = _PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "transcriber.log"

# Match the format shown in the spec: [INFO] message
_FORMAT = "[%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Max 10 MB per log file, keep 3 rotated backups
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 3


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with file rotation and console output."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
