"""
gpu-supervisor — configuration via environment variables.

Why: Centralises runtime tuning into a single Settings object so docker-compose
env vars (TOTAL_VRAM_GB, TIER*_KEEP_ALIVE_SECONDS, etc.) drive the supervisor
without code changes.
What: Pydantic Settings model parsed from environment / .env file.
Test: Set TOTAL_VRAM_GB=12, import settings, assert settings.total_vram_gb == 12.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── GPU budget ────────────────────────────────────────────────────────────
    # REQUIRED — set TOTAL_VRAM_GB to your GPU's usable VRAM in GB.
    # No default: a wrong default would silently mis-budget eviction on any
    # GPU other than the one the default was tuned for. main.py validates
    # this is > 0 at startup and exits with a clear error otherwise.
    total_vram_gb: float = 0.0

    # ── Per-tier keep-alive defaults ─────────────────────────────────────────
    # Tier 1: effectively infinite (never auto-evict)
    tier1_keep_alive_seconds: int = 99_999_999
    # Tier 2: idle-warm services — 30 min idle before unload
    tier2_keep_alive_seconds: int = 1800
    # Tier 3: on-demand services — 5 min after refcount=0
    tier3_keep_alive_seconds: int = 300

    # ── Timeouts for /lifecycle HTTP calls ───────────────────────────────────
    # Load: model weight copy to GPU can take 60–120 s for large models
    lifecycle_load_timeout_seconds: int = 120
    # Unload: GPU memory free is fast but VRAM sync can take a few seconds
    lifecycle_unload_timeout_seconds: int = 60

    # ── Background task interval ─────────────────────────────────────────────
    expiry_check_interval_seconds: int = 30

    # ── Tier 3 yield policy ───────────────────────────────────────────────────
    # Seconds to suggest as Retry-After when a Tier 3 claim is deferred because
    # a higher-priority service (Tier 1 or 2) is actively in use.
    tier3_yield_retry_seconds: int = 60

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Service port ─────────────────────────────────────────────────────────
    port: int = 8202

    # ── Optional API key authentication ──────────────────────────────────────
    # If empty, auth is disabled (open access). If set, all endpoints except
    # /health, /docs, and /openapi.json require X-API-Key header to match.
    api_key: str = ""

    def keep_alive_for_tier(self, tier: int) -> int:
        """Return the keep-alive seconds for the given tier (1, 2, or 3).

        Why: Single lookup point so callers don't duplicate the tier→seconds map.
        Test: Assert keep_alive_for_tier(2) == tier2_keep_alive_seconds.
        """
        mapping = {
            1: self.tier1_keep_alive_seconds,
            2: self.tier2_keep_alive_seconds,
            3: self.tier3_keep_alive_seconds,
        }
        return mapping[tier]


settings = Settings()
