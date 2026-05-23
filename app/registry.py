"""
gpu-supervisor — in-memory service registry.

The registry is the single source of truth for the supervisor's view of each
service.  It is NOT persisted to disk.  On supervisor restart the registry is
rebuilt by querying each service's /lifecycle/status endpoint.

Thread / coroutine safety: all mutations are guarded by asyncio.Lock.  The
lock is intentionally coarse (one lock for the entire registry) because
contention is expected to be extremely low — load/unload operations are rare
relative to read operations, and serialising them is desirable to avoid
double-load races.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional

log = logging.getLogger("gpu-supervisor")


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


@dataclass
class ServiceEntry:
    """Mutable runtime record for a single registered service."""

    service_name: str
    base_url: str
    vram_gb_declared: float
    priority_tier: int
    keep_alive_seconds: int

    # Mutable fields updated during operation
    state: str = "unknown"  # loaded | unloaded | loading | unloading | unknown
    reference_count: int = 0
    last_used: datetime = field(default_factory=_utcnow)

    def is_evictable(self) -> bool:
        """True if the service can be a candidate for eviction."""
        return (
            self.state == "loaded"
            and self.reference_count == 0
            and self.priority_tier > 1  # Tier 1 is never evicted
        )

    def is_idle_expired(self, now: datetime) -> bool:
        """True if keep-alive has elapsed and the service should be unloaded.

        Tier 1 services NEVER auto-expire regardless of keep_alive_seconds.
        This guard is explicit so that a per-service keep_alive_seconds override
        at /register time cannot accidentally expire a Tier 1 service.
        """
        if self.priority_tier == 1:
            return False  # Tier 1 services are never auto-expired
        if self.reference_count > 0:
            return False
        if self.state != "loaded":
            return False
        idle_seconds = (now - self.last_used).total_seconds()
        return idle_seconds > self.keep_alive_seconds


class ServiceRegistry:
    """Thread-safe in-memory registry of GPU services."""

    def __init__(self) -> None:
        self._services: dict[str, ServiceEntry] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ── Mutation helpers (must be called while holding the lock) ──────────────

    def _put(self, entry: ServiceEntry) -> None:
        self._services[entry.service_name] = entry

    # ── Public async API ──────────────────────────────────────────────────────

    async def register(
        self,
        service_name: str,
        base_url: str,
        vram_gb_declared: float,
        priority_tier: int,
        keep_alive_seconds: int,
        initial_state: str = "unknown",
    ) -> tuple[ServiceEntry, bool]:
        """
        Register or update a service entry.

        Returns:
            (entry, is_new) where is_new is True if this was a fresh registration.
        """
        async with self._lock:
            existing = self._services.get(service_name)
            if existing is not None:
                # Update fields that may have changed; preserve runtime state
                existing.base_url = base_url
                existing.vram_gb_declared = vram_gb_declared
                existing.priority_tier = priority_tier
                existing.keep_alive_seconds = keep_alive_seconds
                # Only update state if the new state is not "unknown" so that
                # a re-registration during a running load doesn't clobber state.
                if initial_state != "unknown":
                    existing.state = initial_state
                log.info(
                    "registry.update  service=%s base_url=%s vram=%.1fGB tier=%d state=%s",
                    service_name,
                    base_url,
                    vram_gb_declared,
                    priority_tier,
                    existing.state,
                )
                return existing, False

            entry = ServiceEntry(
                service_name=service_name,
                base_url=base_url,
                vram_gb_declared=vram_gb_declared,
                priority_tier=priority_tier,
                keep_alive_seconds=keep_alive_seconds,
                state=initial_state,
            )
            self._put(entry)
            log.info(
                "registry.new  service=%s base_url=%s vram=%.1fGB tier=%d state=%s",
                service_name,
                base_url,
                vram_gb_declared,
                priority_tier,
                initial_state,
            )
            return entry, True

    async def get(self, service_name: str) -> Optional[ServiceEntry]:
        """Return the entry for a service, or None if not registered."""
        async with self._lock:
            return self._services.get(service_name)

    async def get_all(self) -> list[ServiceEntry]:
        """Return a snapshot list of all entries (safe copy)."""
        async with self._lock:
            return list(self._services.values())

    async def increment_refcount(self, service_name: str) -> int:
        """Increment reference count; raise KeyError if not registered."""
        async with self._lock:
            entry = self._services.get(service_name)
            if entry is None:
                raise KeyError(f"Service not registered: {service_name!r}")
            entry.reference_count += 1
            entry.last_used = _utcnow()
            log.debug(
                "registry.refcount++  service=%s new_count=%d",
                service_name,
                entry.reference_count,
            )
            return entry.reference_count

    async def decrement_refcount(self, service_name: str) -> int:
        """Decrement reference count (clamped to 0); raise KeyError if not registered."""
        async with self._lock:
            entry = self._services.get(service_name)
            if entry is None:
                raise KeyError(f"Service not registered: {service_name!r}")
            entry.reference_count = max(0, entry.reference_count - 1)
            entry.last_used = _utcnow()
            log.debug(
                "registry.refcount--  service=%s new_count=%d",
                service_name,
                entry.reference_count,
            )
            return entry.reference_count

    async def set_state(self, service_name: str, state: str) -> None:
        """Update the state of a registered service."""
        async with self._lock:
            entry = self._services.get(service_name)
            if entry is None:
                raise KeyError(f"Service not registered: {service_name!r}")
            old_state = entry.state
            entry.state = state
            log.debug(
                "registry.state  service=%s %s -> %s",
                service_name,
                old_state,
                state,
            )

    async def used_vram_gb(self) -> float:
        """
        Sum of vram_gb_declared for all services that may be holding VRAM.

        Includes "unknown" state because a failed unload (/lifecycle/unload returned
        an error) leaves the GPU in an indeterminate state — the VRAM is most likely
        still allocated.  Counting it as used prevents over-commit after a failed
        unload.
        """
        async with self._lock:
            return sum(
                e.vram_gb_declared
                for e in self._services.values()
                if e.state in ("loaded", "loading", "unknown")
            )

    async def eviction_candidates(self) -> list[ServiceEntry]:
        """
        Return services eligible for eviction, sorted by eviction priority:
          1. Higher tier first (Tier 3 before Tier 2; Tier 1 excluded entirely)
          2. Within tier: oldest last_used first (LRU)
        """
        async with self._lock:
            candidates = [e for e in self._services.values() if e.is_evictable()]
        # Sort: highest tier first (3 > 2 > 1), then oldest last_used first
        candidates.sort(key=lambda e: (-e.priority_tier, e.last_used))
        return candidates

    async def idle_expired_services(self) -> list[ServiceEntry]:
        """Return services whose keep-alive has elapsed and should be unloaded."""
        now = _utcnow()
        async with self._lock:
            return [e for e in self._services.values() if e.is_idle_expired(now)]

    async def touch(self, service_name: str) -> None:
        """Update last_used timestamp without changing refcount."""
        async with self._lock:
            entry = self._services.get(service_name)
            if entry is not None:
                entry.last_used = _utcnow()
