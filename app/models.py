"""
gpu-supervisor — Pydantic v2 request/response models.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


# ── Registration ──────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    service_name: str
    base_url: str
    vram_gb_declared: float
    priority_tier: int
    keep_alive_seconds: Optional[int] = None

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
    evicted: List[str]


# ── Status ────────────────────────────────────────────────────────────────────


class ServiceStatus(BaseModel):
    service_name: str
    base_url: str
    vram_gb_declared: float
    priority_tier: int
    state: str  # loaded | unloaded | loading | unloading | unknown
    reference_count: int
    last_used: datetime
    keep_alive_seconds: int


class SupervisorStatus(BaseModel):
    services: List[ServiceStatus]
    total_vram_gb: float
    used_vram_gb: float
    available_vram_gb: float
    started_at: datetime
    last_eviction: Optional[datetime]
    eviction_count: int


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    registered_services: int
    loaded_services: int
    detail: Optional[str] = None
