"""
gpu-supervisor — Pydantic v2 request/response models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

# ── Registration ──────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    service_name: str
    base_url: str
    vram_gb_declared: float
    priority_tier: int
    keep_alive_seconds: Optional[int] = None
    # Physical GPU device this service loads onto. Defaults to "default" so
    # existing single-GPU callers that omit the field keep working unchanged.
    # Multi-GPU deployments pass "0"/"1" to map to GPU0_VRAM_GB/GPU1_VRAM_GB.
    device_id: str = "default"

    @field_validator("priority_tier")
    @classmethod
    def tier_must_be_valid(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError(f"priority_tier must be 1, 2, or 3; got {v}")
        return v

    @field_validator("vram_gb_declared")
    @classmethod
    def vram_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"vram_gb_declared must be > 0; got {v}")
        return v

    @field_validator("service_name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("service_name must not be empty")
        return v

    @field_validator("base_url")
    @classmethod
    def url_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("base_url must not be empty")
        # Normalise: strip trailing slash
        return v.rstrip("/")


class RegisterResponse(BaseModel):
    service_name: str
    status: str  # "registered" | "already_registered_updated"
    initial_state: str  # supervisor's view after first /lifecycle/status query


# ── Claim ─────────────────────────────────────────────────────────────────────


class ClaimResponse(BaseModel):
    service_name: str
    status: str  # "loaded" | "loading" | "evicted_to_load" | "failed"
    waited_seconds: float
    reference_count: int
    evicted: list[str]


# ── Status ────────────────────────────────────────────────────────────────────


class ServiceStatus(BaseModel):
    service_name: str
    base_url: str
    vram_gb_declared: float
    priority_tier: int
    device_id: str  # physical GPU device the service is budgeted against
    state: str  # loaded | unloaded | loading | unloading | unknown
    reference_count: int
    last_used: datetime
    keep_alive_seconds: int


class PerDeviceVRAM(BaseModel):
    """Per-physical-GPU VRAM accounting for the /status response.

    Why: Aggregate fields alone are misleading in multi-GPU mode — summing usage
    across devices against a single budget can report negative availability. A
    per-device breakdown lets monitoring see each GPU's true headroom.
    What: Holds total/used/available VRAM (GB) for one device_id.
    Test: For device "0" with 7.4 GB budget and 6.0 GB used, assert
    available_vram_gb == 1.4.
    """

    total_vram_gb: float
    used_vram_gb: float
    available_vram_gb: float


class DeviceReconciliation(BaseModel):
    """Measured-vs-declared VRAM comparison for one physical GPU.

    Why: Surfaces drift between nvidia-smi's measured usage and the supervisor's
    declared accounting so a leaked CUDA context (actual >> declared) is visible
    in /status, not just buried in logs.
    What: actual_used_mb (from nvidia-smi), declared_sum_mb (sum of loaded-service
    footprints on this device), delta_mb = actual - declared, and a status of
    "ok" or "leak_suspected" (delta_mb > leak_threshold_mb).
    Test: For actual=2870, declared=2867, threshold=500 assert status=="ok" and
    delta_mb==3; for actual=4000, declared=2867 assert status=="leak_suspected".
    """

    actual_used_mb: int
    declared_sum_mb: int
    delta_mb: int
    status: str  # "ok" | "leak_suspected"


class Reconciliation(BaseModel):
    """Soft-reconciliation block for the /status response.

    Why: Bundles the per-device comparison plus a human-readable warning list so
    a single field tells monitoring whether measured VRAM agrees with accounting.
    What: sampled_at is the nvidia-smi sample time (None if no sample yet);
    devices maps device_id → DeviceReconciliation; warnings lists leak messages.
    Test: With no nvidia-smi sample assert sampled_at is None and devices == {};
    with a leak present assert warnings is non-empty.
    """

    sampled_at: Optional[datetime] = None
    devices: dict[str, DeviceReconciliation] = {}
    warnings: list[str] = []


class SupervisorStatus(BaseModel):
    services: list[ServiceStatus]
    total_vram_gb: float
    used_vram_gb: float
    available_vram_gb: float
    # Per-device VRAM breakdown keyed by device_id. Empty when no services are
    # registered. Authoritative for multi-GPU availability; the aggregate fields
    # above are sums and may not reflect per-device headroom.
    per_device: dict[str, PerDeviceVRAM]
    # Soft reconciliation: nvidia-smi measured usage vs declared accounting.
    # Empty (sampled_at None, no devices) when nvidia-smi is unavailable or no
    # sample has completed yet.
    reconciliation: Reconciliation
    started_at: datetime
    last_eviction: Optional[datetime]
    eviction_count: int


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    registered_services: int
    loaded_services: int
    detail: Optional[str] = None
