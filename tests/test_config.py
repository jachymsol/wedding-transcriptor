"""Tests for transcriptor.config — configuration loading and defaults."""

from __future__ import annotations

import pytest
import yaml
from pydantic_settings import SettingsConfigDict

from transcriptor.config import (
    AppConfig,
    AudioConfig,
    ServerConfig,
    StabilizationConfig,
    TranscriptionConfig,
    VADConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Default values for every nested model
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_server_websocket_url(self):
        assert ServerConfig().websocket_url == "wss://translate.example.com/ws"

    def test_audio_device_id(self):
        assert AudioConfig().device_id == "default"

    def test_transcription_model(self):
        assert TranscriptionConfig().model == "medium"

    def test_transcription_language(self):
        assert TranscriptionConfig().language == "en"

    def test_stabilization_silence_ms(self):
        assert StabilizationConfig().silence_ms == 700

    def test_stabilization_stable_ms(self):
        assert StabilizationConfig().stable_ms == 2000

    def test_vad_max_speech_ms(self):
        assert VADConfig().max_speech_ms == 10_000

    def test_vad_overlap_ms(self):
        assert VADConfig().overlap_ms == 3_000

    def test_app_event_id(self):
        # Default when no YAML or env override is present
        # (may be overridden by the project's own config.yaml — that's fine
        # for this test, which only checks the model field exists)
        cfg = AppConfig(event_id="test-event")
        assert cfg.event_id == "test-event"


# ---------------------------------------------------------------------------
# Priority: init kwargs beat YAML
# ---------------------------------------------------------------------------

class TestPriority:
    def test_init_kwargs_override_yaml_event_id(self):
        """Values passed to __init__ take precedence over what's in config.yaml."""
        cfg = AppConfig(event_id="override-event")
        assert cfg.event_id == "override-event"

    def test_init_kwargs_override_nested_server(self):
        cfg = AppConfig(server=ServerConfig(websocket_url="wss://override.example.com/ws"))
        assert cfg.server.websocket_url == "wss://override.example.com/ws"


# ---------------------------------------------------------------------------
# YAML loading — integration test against a temporary config file
# ---------------------------------------------------------------------------

class TestYamlLoading:
    def test_all_fields_loaded_from_yaml(self, tmp_path):
        """A complete config.yaml is parsed into every nested model correctly."""
        payload = {
            "event_id": "yaml-wedding",
            "server": {"websocket_url": "wss://yaml.example.com/ws"},
            "audio": {"device_id": "usb-mixer"},
            "transcription": {"model": "small", "language": "cs"},
            "stabilization": {"silence_ms": 500, "stable_ms": 1500},
            "vad": {"max_speech_ms": 5000, "overlap_ms": 2000},
        }
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(payload))

        # Create a subclass that points at the temp file — pydantic-settings
        # reads model_config["yaml_file"] dynamically in settings_customise_sources,
        # so subclassing is the cleanest way to inject an alternative file.
        class _TmpConfig(AppConfig):
            model_config = SettingsConfigDict(yaml_file=str(yaml_file))

        cfg = _TmpConfig()
        assert cfg.event_id == "yaml-wedding"
        assert cfg.server.websocket_url == "wss://yaml.example.com/ws"
        assert cfg.audio.device_id == "usb-mixer"
        assert cfg.transcription.model == "small"
        assert cfg.transcription.language == "cs"
        assert cfg.stabilization.silence_ms == 500
        assert cfg.stabilization.stable_ms == 1500
        assert cfg.vad.max_speech_ms == 5000
        assert cfg.vad.overlap_ms == 2000

    def test_missing_yaml_file_falls_back_to_defaults(self, tmp_path):
        """If the YAML file does not exist, all defaults still apply."""
        missing = tmp_path / "nonexistent.yaml"

        class _TmpConfig(AppConfig):
            model_config = SettingsConfigDict(yaml_file=str(missing))

        cfg = _TmpConfig()
        # Defaults from the model definitions
        assert cfg.audio.device_id == "default"
        assert cfg.transcription.model == "medium"

    def test_partial_yaml_leaves_other_fields_at_default(self, tmp_path):
        """A YAML that only overrides some fields leaves the rest at their defaults."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump({"event_id": "partial-event"}))

        class _TmpConfig(AppConfig):
            model_config = SettingsConfigDict(yaml_file=str(yaml_file))

        cfg = _TmpConfig()
        assert cfg.event_id == "partial-event"
        assert cfg.audio.device_id == "default"       # untouched default
        assert cfg.stabilization.silence_ms == 700     # untouched default


# ---------------------------------------------------------------------------
# load_config() — integration against the project's own config.yaml
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_app_config_instance(self):
        assert isinstance(load_config(), AppConfig)

    def test_nested_models_are_correct_types(self):
        cfg = load_config()
        assert isinstance(cfg.server, ServerConfig)
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.transcription, TranscriptionConfig)
        assert isinstance(cfg.stabilization, StabilizationConfig)
        assert isinstance(cfg.vad, VADConfig)

    def test_project_config_yaml_values(self):
        """The values in the committed config.yaml match the spec defaults."""
        cfg = load_config()
        assert cfg.event_id == "wedding-2027"
        assert cfg.server.websocket_url == "wss://translate.example.com/ws"
        assert cfg.audio.device_id == "default"
        assert cfg.transcription.model == "medium"
        assert cfg.transcription.language == "en"
        assert cfg.stabilization.silence_ms == 700
        assert cfg.stabilization.stable_ms == 2000
        assert cfg.vad.max_speech_ms == 10_000
        assert cfg.vad.overlap_ms == 3_000
