"""
gpu-supervisor — configuration via environment variables.

Why: Centralises runtime tuning into a single Settings object so docker-compose
env vars (TOTAL_VRAM_GB, TIER*_KEEP_ALIVE_SECONDS, etc.) drive the supervisor
without code changes.
What: Pydantic Settings model parsed from environment / .env file.
Test: Set TOTAL_VRAM_GB=12, import settings, assert settings.total_vram_gb == 12.
"""

from __future__ import annotations

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("gpu-supervisor.config")


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

    # ── Per-device GPU budgets (multi-GPU deployments) ───────────────────────
    # Optional per-physical-GPU VRAM budgets. When > 0 they scope claim/eviction
    # accounting to that device so co-resident GPUs (e.g. a P4 and an RTX 3060)
    # cannot evict each other or OOM-conflict. When 0 (the default), the device
    # falls back to total_vram_gb, preserving single-GPU behaviour.
    # Env vars: GPU0_VRAM_GB, GPU1_VRAM_GB.
    gpu0_vram_gb: float = 0.0
    gpu1_vram_gb: float = 0.0

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

    # ── Soft reconciliation (nvidia-smi polling) ──────────────────────────────
    # How often to sample actual per-device VRAM with nvidia-smi to compare
    # against the supervisor's declared accounting. Env var: GPU_POLL_SECONDS.
    gpu_poll_seconds: int = 300
    # A device whose measured VRAM exceeds its declared sum by more than this
    # many MB is flagged "leak_suspected" in /status and logged at WARNING —
    # the likely cause is a leaked CUDA context the supervisor isn't accounting
    # for. Env var: LEAK_THRESHOLD_MB.
    leak_threshold_mb: int = 500

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

    def budget_for_device(self, device_id: str) -> float:
        """Return the VRAM budget (GB) for a physical GPU device.

        Why: Centralises the device_id → budget mapping so claim/register/eviction
        all agree on how much VRAM a device has, and so single-GPU deployments that
        only set TOTAL_VRAM_GB keep working without per-device config.
        What: Maps "0" → gpu0_vram_gb (or total_vram_gb if unset), "1" → gpu1_vram_gb
        (or total_vram_gb if unset), and any other id (incl. "default") → total_vram_gb.
        An unrecognised, non-"default" device_id logs a warning so misconfigured
        callers are surfaced rather than silently mis-budgeted.
        Test: With total=11.6, gpu0=7.4, gpu1=11.6: assert budget_for_device("0")==7.4,
        ("1")==11.6, ("default")==11.6; with gpu0 unset assert ("0")==total_vram_gb.
        """
        if device_id == "0":
            return self.gpu0_vram_gb or self.total_vram_gb
        if device_id == "1":
            return self.gpu1_vram_gb or self.total_vram_gb
        if device_id != "default":
            log.warning("Unknown device_id %r — falling back to total_vram_gb budget", device_id)
        return self.total_vram_gb


settings = Settings()
