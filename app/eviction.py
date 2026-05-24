"""
gpu-supervisor — eviction algorithm.

This module implements the eviction policy: given a set of loaded services
and a VRAM deficit, select and unload services in priority order until
enough VRAM has been freed, or raise NotEnoughVRAMError if it cannot be done.

Eviction order (descending priority of removal):
  1. Tier 3 services (on-demand), oldest last_used first
  2. Tier 2 services (idle-warm), oldest last_used first
  3. Tier 1 services are NEVER evicted (excluded from candidates)

Absolute protection: services with reference_count > 0 are never evicted,
regardless of tier.  This is enforced by ServiceEntry.is_evictable().

The eviction logic is async because unloading requires HTTP calls to the
service's /lifecycle/unload endpoint.
"""

from __future__ import annotations

import logging

from lifecycle_client import LifecycleClient, LifecycleError
from registry import ServiceRegistry

log = logging.getLogger("gpu-supervisor")


class NotEnoughVRAMError(Exception):
    """Raised when eviction cannot free enough VRAM for a requested load."""

    def __init__(self, needed: float, freed: float, candidates_exhausted: int) -> None:
        self.needed = needed
        self.freed = freed
        self.candidates_exhausted = candidates_exhausted
        super().__init__(
            f"Cannot free {needed:.2f} GB VRAM: only freed {freed:.2f} GB "
            f"after exhausting {candidates_exhausted} candidate(s)"
        )


async def evict_for_vram(
    vram_needed: float,
    registry: ServiceRegistry,
    client: LifecycleClient,
    device_id: str = "default",
) -> list[str]:
    """
    Evict services to free at least `vram_needed` GB of VRAM on one GPU device.

    Algorithm:
      1. Collect eviction candidates ON device_id: loaded, refcount==0, tier > 1
      2. Sort: Tier 3 first, then Tier 2; within tier, oldest last_used first
      3. Iterate: unload each candidate until freed >= vram_needed
      4. If freed < vram_needed after all candidates: raise NotEnoughVRAMError

    device_id scopes eviction to a single physical GPU so a claim on one device
    never frees VRAM on a different device (which would still OOM the target GPU).
    It defaults to "default" to preserve single-GPU callers.

    Returns:
        List of service names that were successfully evicted.

    Raises:
        NotEnoughVRAMError if insufficient VRAM can be freed.
    """
    candidates = await registry.eviction_candidates_for_device(device_id)

    if not candidates:
        raise NotEnoughVRAMError(
            needed=vram_needed,
            freed=0.0,
            candidates_exhausted=0,
        )

    log.info(
        "eviction.start  needed=%.2fGB candidates=%d",
        vram_needed,
        len(candidates),
    )

    freed: float = 0.0
    evicted: list[str] = []

    for entry in candidates:
        if freed >= vram_needed:
            break

        service_name = entry.service_name

        # Re-check evictability: another coroutine may have incremented refcount
        # between when we built the candidate list and now.
        fresh = await registry.get(service_name)
        if fresh is None or not fresh.is_evictable():
            log.info(
                "eviction.skip  service=%s reason=no_longer_evictable",
                service_name,
            )
            continue

        # Mark as unloading before the HTTP call to prevent double-eviction
        await registry.set_state(service_name, "unloading")

        try:
            await client.unload(service_name, entry.base_url)
            await registry.set_state(service_name, "unloaded")
            freed += entry.vram_gb_declared
            evicted.append(service_name)
            log.info(
                "eviction.success  service=%s freed=%.2fGB total_freed=%.2fGB",
                service_name,
                entry.vram_gb_declared,
                freed,
            )
        except LifecycleError as exc:
            # Log and continue — mark as an error state but don't crash
            log.warning(
                "eviction.failed  service=%s error=%s",
                service_name,
                exc,
            )
            await registry.set_state(service_name, "unknown")
            # Don't add this service's VRAM to freed total since unload failed

    if freed < vram_needed:
        raise NotEnoughVRAMError(
            needed=vram_needed,
            freed=freed,
            candidates_exhausted=len(candidates),
        )

    log.info(
        "eviction.complete  evicted=%s freed=%.2fGB",
        evicted,
        freed,
    )
    return evicted
