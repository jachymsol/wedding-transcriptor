"""Application entry point — wired up fully in Step 9."""

from __future__ import annotations

import logging

from transcriptor.config import load_config
from transcriptor.logging_setup import setup_logging


def main() -> None:
    setup_logging()
    config = load_config()
    logging.info("Configuration loaded: event_id=%s", config.event_id)
    logging.info("Server URL: %s", config.server.websocket_url)


if __name__ == "__main__":
    main()
