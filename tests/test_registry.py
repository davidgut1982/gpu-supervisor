"""
Unit tests for ServiceRegistry.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from registry import ServiceRegistry


def _make_entry_kwargs(
    name: str = "svc-a",
    base_url: str = "http://svc-a:8000",
    vram_gb: float = 2.0,
    tier: int = 2,
    keep_alive: int = 1800,
    state: str = "unloaded",
) -> dict:
    return dict(
        service_name=name,
        base_url=base_url,
        vram_gb_declared=vram_gb,
        priority_tier=tier,
        keep_alive_seconds=keep_alive,
        initial_state=state,
    )


@pytest.mark.asyncio
async def test_register_new_service():
    reg = ServiceRegistry()
    entry, is_new = await reg.register(**_make_entry_kwargs(name="svc-a"))
    assert is_new is True
    assert entry.service_name == "svc-a"
    assert entry.state == "unloaded"
    assert entry.reference_count == 0


@pytest.mark.asyncio
async def test_register_idempotent_update():
    reg = ServiceRegistry()
    _, is_new1 = await reg.register(**_make_entry_kwargs(name="svc-a", vram_gb=2.0))
    assert is_new1 is True

    _, is_new2 = await reg.register(**_make_entry_kwargs(name="svc-a", vram_gb=3.0))
    assert is_new2 is False

    entry = await reg.get("svc-a")
    assert entry is not None
    assert entry.vram_gb_declared == 3.0


@pytest.mark.asyncio
async def test_register_updates_base_url_and_tier():
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="svc-a", base_url="http://old:8000", tier=2))
    await reg.register(**_make_entry_kwargs(name="svc-a", base_url="http://new:8001", tier=3))

    entry = await reg.get("svc-a")
    assert entry.base_url == "http://new:8001"
    assert entry.priority_tier == 3


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none():
    reg = ServiceRegistry()
    result = await reg.get("does-not-exist")
    assert result is None


@pytest.mark.asyncio
async def test_increment_refcount():
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="svc-a"))

    new_count = await reg.increment_refcount("svc-a")
    assert new_count == 1

    new_count = await reg.increment_refcount("svc-a")
    assert new_count == 2


@pytest.mark.asyncio
async def test_decrement_refcount():
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="svc-a"))
    await reg.increment_refcount("svc-a")
    await reg.increment_refcount("svc-a")

    count = await reg.decrement_refcount("svc-a")
    assert count == 1

    count = await reg.decrement_refcount("svc-a")
    assert count == 0


@pytest.mark.asyncio
async def test_decrement_refcount_never_goes_negative():
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="svc-a"))

    # Decrement on zero should clamp to 0
    count = await reg.decrement_refcount("svc-a")
    assert count == 0

    # Second decrement still 0
    count = await reg.decrement_refcount("svc-a")
    assert count == 0


@pytest.mark.asyncio
async def test_increment_refcount_unregistered_raises():
    reg = ServiceRegistry()
    with pytest.raises(KeyError, match="does-not-exist"):
        await reg.increment_refcount("does-not-exist")


@pytest.mark.asyncio
async def test_decrement_refcount_unregistered_raises():
    reg = ServiceRegistry()
    with pytest.raises(KeyError, match="does-not-exist"):
        await reg.decrement_refcount("does-not-exist")


@pytest.mark.asyncio
async def test_set_state():
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="svc-a", state="unloaded"))
    await reg.set_state("svc-a", "loaded")

    entry = await reg.get("svc-a")
    assert entry.state == "loaded"


@pytest.mark.asyncio
async def test_set_state_unregistered_raises():
    reg = ServiceRegistry()
    with pytest.raises(KeyError):
        await reg.set_state("does-not-exist", "loaded")


@pytest.mark.asyncio
async def test_used_vram_gb_sums_loaded_only():
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="svc-a", vram_gb=2.0, state="loaded"))
    await reg.register(**_make_entry_kwargs(name="svc-b", vram_gb=3.0, state="unloaded"))
    await reg.register(**_make_entry_kwargs(name="svc-c", vram_gb=1.5, state="loading"))

    used = await reg.used_vram_gb()
    # loaded (2.0) + loading (1.5) = 3.5
    assert abs(used - 3.5) < 0.001


@pytest.mark.asyncio
async def test_get_all_returns_all_entries():
    reg = ServiceRegistry()
    for i in range(3):
        await reg.register(**_make_entry_kwargs(name=f"svc-{i}"))

    entries = await reg.get_all()
    assert len(entries) == 3
    names = {e.service_name for e in entries}
    assert names == {"svc-0", "svc-1", "svc-2"}


@pytest.mark.asyncio
async def test_eviction_candidates_tier3_before_tier2():
    reg = ServiceRegistry()
    # Tier 2 loaded, refcount=0
    await reg.register(**_make_entry_kwargs(name="tier2-svc", tier=2, vram_gb=2.0, state="loaded"))
    # Tier 3 loaded, refcount=0
    await reg.register(**_make_entry_kwargs(name="tier3-svc", tier=3, vram_gb=1.5, state="loaded"))

    candidates = await reg.eviction_candidates()
    assert len(candidates) == 2
    # Tier 3 should come first
    assert candidates[0].service_name == "tier3-svc"
    assert candidates[1].service_name == "tier2-svc"


@pytest.mark.asyncio
async def test_eviction_candidates_excludes_tier1():
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="tier1-svc", tier=1, state="loaded"))
    await reg.register(**_make_entry_kwargs(name="tier3-svc", tier=3, state="loaded"))

    candidates = await reg.eviction_candidates()
    names = {c.service_name for c in candidates}
    assert "tier1-svc" not in names
    assert "tier3-svc" in names


@pytest.mark.asyncio
async def test_eviction_candidates_excludes_nonzero_refcount():
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="busy-svc", tier=3, state="loaded"))
    await reg.increment_refcount("busy-svc")

    candidates = await reg.eviction_candidates()
    assert all(c.service_name != "busy-svc" for c in candidates)


@pytest.mark.asyncio
async def test_eviction_candidates_excludes_unloaded():
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="unloaded-svc", tier=3, state="unloaded"))

    candidates = await reg.eviction_candidates()
    assert len(candidates) == 0


@pytest.mark.asyncio
async def test_touch_updates_last_used():
    import time

    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="svc-a"))

    entry_before = await reg.get("svc-a")
    ts_before = entry_before.last_used

    await asyncio.sleep(0.01)  # Small delay to ensure timestamp difference
    await reg.touch("svc-a")

    entry_after = await reg.get("svc-a")
    assert entry_after.last_used > ts_before


# ── Regression: Issue 1 — VRAM accounting after failed unload ─────────────────


@pytest.mark.asyncio
async def test_used_vram_gb_counts_unknown_state():
    """
    Regression: services in 'unknown' state (failed unload) must still count
    toward used VRAM.  Before the fix, unknown was excluded and a subsequent
    claim could over-commit the GPU.
    """
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="svc-fail", vram_gb=4.0, state="loaded"))

    # Loaded state: VRAM must be counted
    used = await reg.used_vram_gb()
    assert abs(used - 4.0) < 0.001

    # Simulate failed unload — service transitions to "unknown"
    await reg.set_state("svc-fail", "unknown")

    # VRAM must STILL be counted (GPU still holds it)
    used_after = await reg.used_vram_gb()
    assert abs(used_after - 4.0) < 0.001, (
        f"used_vram_gb should still be 4.0 after failed unload (unknown state), got {used_after}"
    )


@pytest.mark.asyncio
async def test_used_vram_gb_excludes_cleanly_unloaded():
    """Cleanly unloaded services (state='unloaded') must NOT count toward VRAM."""
    reg = ServiceRegistry()
    await reg.register(**_make_entry_kwargs(name="svc-clean", vram_gb=3.0, state="loaded"))
    await reg.set_state("svc-clean", "unloaded")

    used = await reg.used_vram_gb()
    assert used == 0.0


# ── Regression: Issue 3 — Tier 1 expiry protection ───────────────────────────


@pytest.mark.asyncio
async def test_tier1_is_idle_expired_returns_false_regardless_of_keep_alive():
    """
    Regression: Tier 1 services must never return True from is_idle_expired(),
    even when keep_alive_seconds is overridden to a small value.

    Before the fix, only the magic number 99_999_999 prevented expiry; a small
    explicit keep_alive_seconds would have caused Tier 1 to be expired.
    """
    import asyncio as _asyncio
    from datetime import datetime, timezone

    reg = ServiceRegistry()
    # Register Tier 1 with a very short keep_alive (1 second) — should never expire
    await reg.register(
        service_name="tier1-permanent",
        base_url="http://tier1:8000",
        vram_gb_declared=2.0,
        priority_tier=1,
        keep_alive_seconds=1,  # Intentionally tiny
        initial_state="loaded",
    )

    # Wait longer than keep_alive_seconds
    await _asyncio.sleep(1.1)

    # is_idle_expired() must return False for Tier 1 despite elapsed time
    entry = await reg.get("tier1-permanent")
    assert entry is not None
    now = datetime.now(tz=timezone.utc)
    assert entry.is_idle_expired(now) is False, (
        "Tier 1 service should never be idle-expired regardless of keep_alive_seconds"
    )


@pytest.mark.asyncio
async def test_tier1_not_in_idle_expired_services():
    """
    Regression: Tier 1 services with small keep_alive_seconds must not appear
    in idle_expired_services() results.
    """
    import asyncio as _asyncio

    reg = ServiceRegistry()
    await reg.register(
        service_name="tier1-perm",
        base_url="http://tier1:8000",
        vram_gb_declared=2.0,
        priority_tier=1,
        keep_alive_seconds=1,
        initial_state="loaded",
    )
    # Also register a Tier 2 that SHOULD expire
    await reg.register(
        service_name="tier2-expires",
        base_url="http://tier2:8000",
        vram_gb_declared=1.0,
        priority_tier=2,
        keep_alive_seconds=1,
        initial_state="loaded",
    )

    await _asyncio.sleep(1.1)

    expired_names = [e.service_name for e in await reg.idle_expired_services()]
    assert "tier1-perm" not in expired_names, (
        "Tier 1 service must never appear in idle_expired_services()"
    )
    assert "tier2-expires" in expired_names, (
        "Tier 2 service with elapsed keep_alive should appear in idle_expired_services()"
    )
