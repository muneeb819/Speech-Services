from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from SPEECH_* environment variables (or .env)."""

    model_config = SettingsConfigDict(env_prefix="SPEECH_", env_file=".env", extra="ignore")

    backend: str = "real"  # real | fake (fake = CPU test doubles, never use in production)
    data_dir: Path = Path("/data")
    log_level: str = "INFO"
    cors_origins: str = ""  # comma separated; empty = CORS disabled

    # --- auth / limits ---
    api_keys: str = ""  # "name:secret,name2:secret2" or bare secrets
    auth_disabled: bool = False
    rate_limit_per_min: int = 240
    max_ws_per_key: int = 16
    max_upload_mb: int = 300
    max_audio_seconds: int = 7200
    sync_max_seconds: int = 600  # longer files must use /v1/jobs
    max_tts_chars: int = 20000
    gate_timeout_seconds: float = 30.0  # wait for a GPU slot before returning 503

    # --- STT (faster-whisper) ---
    stt_model: str = "large-v3-turbo"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    stt_cpu_threads: int = 4
    stt_concurrency: int = 1
    stt_language: str = "en"
    stt_batch_size: int = 8  # file transcription; 0 = sequential

    # --- live STT ---
    vad: str = "silero"  # silero | energy
    vad_energy_threshold: float = 0.012
    live_endpoint_ms: int = 600
    live_partial_interval_ms: int = 1000
    live_max_utterance_s: int = 30
    live_idle_timeout_s: int = 120

    # --- TTS (Kokoro) ---
    tts_device: str = "cuda"
    tts_repo: str = "hexgrad/Kokoro-82M"
    tts_voice: str = "af_heart"
    tts_concurrency: int = 1
    extra_voices: str = ""  # comma separated additional Kokoro voice names

    # --- voice cloning (separate Chatterbox container) ---
    clone_url: str = ""  # e.g. http://clone:8100 ; empty = cloning disabled
    clone_token: str = ""

    # --- jobs ---
    job_workers: int = 1
    job_queue_max: int = 200
    retention_hours: int = 24

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.backend not in {"real", "fake"}:
            raise ValueError("SPEECH_BACKEND must be real or fake")
        if self.clone_url and not self.clone_token:
            raise ValueError("SPEECH_CLONE_TOKEN is required when SPEECH_CLONE_URL is set")
        return self

    def validate_for_start(self) -> None:
        if not self.auth_disabled and not self.api_keys.strip():
            raise RuntimeError("Set SPEECH_API_KEYS (or SPEECH_AUTH_DISABLED=true for local development only)")


@lru_cache
def get_settings() -> Settings:
    return Settings()
