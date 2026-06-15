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
CONFIG_FILE: Path = _PROJECT_ROOT / "config.yaml"   # public — used by startup dialog
_CONFIG_FILE = CONFIG_FILE                            # keep old name for internal use


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


class VADConfig(BaseModel):
    #: Emit a partial segment and reset the buffer when speech exceeds this
    #: duration (milliseconds).  Set to 0 to disable — VAD will accumulate
    #: indefinitely until silence is detected.
    max_speech_ms: int = 10_000   # 10 s
    #: Words at the *end* of a partial window have accumulated decoder context
    #: and are therefore more reliable.  This many milliseconds of audio from
    #: the end of the window are committed as-is and then *skipped* at the
    #: start of the next window (to avoid re-committing with worse context).
    end_overlap_ms: int = 1_000   # 1 s
    #: Additional audio kept at the start of the next window purely as
    #: transcription context.  These words appear right after the skipped
    #: ``end_overlap_ms`` region and benefit from more right-side audio
    #: than they had in the previous window.
    start_overlap_ms: int = 2_000 # 2 s


class AppConfig(BaseSettings):
    event_id: str = "wedding-2027"
    server: ServerConfig = ServerConfig()
    audio: AudioConfig = AudioConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    stabilization: StabilizationConfig = StabilizationConfig()
    vad: VADConfig = VADConfig()

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
