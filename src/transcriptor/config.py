"""Configuration models loaded from config.yaml via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Type

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
    #: Bare host (optionally with port), e.g. "translate.example.com" or
    #: "localhost:3000". Scheme (ws/wss, http/https) is inferred: "localhost"
    #: and "127.0.0.1" use unencrypted ws/http; everything else uses wss/https.
    host: str = "translate.example.com"


class AudioConfig(BaseModel):
    device_id: str = "default"


class TranscriptionConfig(BaseModel):
    model: str = "medium"
    language: str = "en"
    #: Per-language Whisper "initial_prompt" priming text, keyed by
    #: ISO-639-1 code. Whisper is less confident on lower-resource
    #: languages (e.g. Czech); a short in-language priming snippet biases
    #: decoding toward the expected vocabulary/orthography and reduces
    #: hallucinated segments. Languages with no entry get no prompt.
    initial_prompts: dict[str, str] = {}

    def initial_prompt(self, language: str) -> Optional[str]:
        """Return the configured initial_prompt for *language*, or None.

        An empty-string entry is treated the same as "not set".
        """
        return self.initial_prompts.get(language) or None


class StabilizationConfig(BaseModel):
    silence_ms: int = 700
    stable_ms: int = 2000


class VADOverride(BaseModel):
    """Per-language overrides for :class:`VADConfig` timing fields.

    Any field left ``None`` falls back to the corresponding global
    ``VADConfig`` value (see :meth:`VADConfig.effective`).
    """

    max_speech_ms: Optional[int] = None
    end_overlap_ms: Optional[int] = None
    start_overlap_ms: Optional[int] = None


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
    #: Per-language overrides, keyed by ISO-639-1 code (e.g. ``"cs"``).
    #: Languages with sparser Whisper training data (e.g. Czech) often
    #: benefit from longer segments / more overlap context; this lets
    #: those languages use different timings without changing the global
    #: defaults used by every other language.
    overrides: dict[str, VADOverride] = {}

    def effective(self, language: str) -> Tuple[int, int, int]:
        """Return ``(max_speech_ms, end_overlap_ms, start_overlap_ms)`` for *language*.

        Falls back to the global fields for any override field left unset
        (or when no override is defined for *language* at all).
        """
        o = self.overrides.get(language)
        return (
            o.max_speech_ms if o and o.max_speech_ms is not None else self.max_speech_ms,
            o.end_overlap_ms if o and o.end_overlap_ms is not None else self.end_overlap_ms,
            o.start_overlap_ms if o and o.start_overlap_ms is not None else self.start_overlap_ms,
        )


class AppConfig(BaseSettings):
    event_id: str = "wedding-2027"
    api_key: str = ""
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
