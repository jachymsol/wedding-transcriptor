"""Configuration models loaded from config.yaml via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Resolve config.yaml relative to the project root (two levels above this file).
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_FILE = _PROJECT_ROOT / "config.yaml"


class ServerConfig(BaseModel):
    websocket_url: str = "wss://translate.example.com/ws"


class AudioConfig(BaseModel):
    device_id: str = "default"


class TranscriptionConfig(BaseModel):
    model: str = "medium"
    language: str = "en"


class StabilizationConfig(BaseModel):
    silence_ms: int = 700
    stable_ms: int = 2000


class AppConfig(BaseSettings):
    event_id: str = "wedding-2027"
    server: ServerConfig = ServerConfig()
    audio: AudioConfig = AudioConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    stabilization: StabilizationConfig = StabilizationConfig()

    model_config = SettingsConfigDict(yaml_file=str(_CONFIG_FILE))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # Priority: explicit init kwargs > environment variables > config.yaml
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


def load_config() -> AppConfig:
    """Load and return the application configuration."""
    return AppConfig()
