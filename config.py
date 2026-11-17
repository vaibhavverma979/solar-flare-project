"""
config.py — Central configuration loader for the Aditya-L1 Flare System.

All tuneable parameters live in .env (see .env.example).
Import `settings` from this module anywhere in the codebase.

Example
-------
>>> from config import settings
>>> print(settings.data_raw_dir)
PosixPath('data/raw')
"""

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration loaded from the .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    data_catalogues_dir: Path = Path("data/catalogues")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///data/flare_system.db"
    timescaledb_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/aditya"
    redis_url: str = "redis://localhost:6379/0"

    # ── Ingestion ─────────────────────────────────────────────────────────────
    resample_cadence_s: int = 1  # seconds; 0 = use coarsest native cadence

    # ── Nowcasting (Phase 3) ──────────────────────────────────────────────────
    sigma_threshold: float = 5.0
    coincidence_window_s: int = 120  # ±2 minutes cross-instrument tolerance

    # ── Forecasting (Phase 4) ─────────────────────────────────────────────────
    forecast_horizon_min: int = 30
    sequence_window_min: int = 60

    # ── API (Phase 5) ─────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    stream_playback_speed: float = 1.0  # 1.0 = real-time; 10.0 = 10× faster
    openai_api_key: str | None = None

    # ── SUIT Vision (Phase 7 — Multi-Modal) ───────────────────────────────────
    suit_image_dir: Path = Path("data/raw/suit")
    suit_model_path: Path = Path("data/models/suit_cnn.pt")

    # ── Vector Database (Phase 7 — Multi-Modal) ──────────────────────────────
    vectordb_path: Path = Path("data/vectordb")
    vectordb_collection: str = "historical_flares"
    embedding_dim: int = 128

    # ── PatchTST Transformer (Phase 7 — Multi-Modal) ─────────────────────────
    patchtst_patch_len: int = 16
    patchtst_d_model: int = 64
    patchtst_n_heads: int = 4
    patchtst_n_layers: int = 3
    patchtst_context_len: int = 360  # 6-hour window at 1-min cadence

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    @field_validator("resample_cadence_s")
    @classmethod
    def cadence_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("resample_cadence_s must be >= 0")
        return v

    @field_validator("sigma_threshold")
    @classmethod
    def sigma_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("sigma_threshold must be > 0")
        return v

    def ensure_dirs(self) -> None:
        """Create all data directories if they do not already exist."""
        for d in (self.data_raw_dir, self.data_processed_dir, self.data_catalogues_dir):
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere
settings = Settings()
