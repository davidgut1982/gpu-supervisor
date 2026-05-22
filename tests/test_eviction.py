"""
Unit tests for the eviction algorithm and Tier 1 claim preemption.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from eviction import NotEnoughVRAMError, evict_for_vram
from lifecycle_client import LifecycleError
from registry import ServiceRegistry

APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR))


def _utcnow():
    return datetime.now(tz=timezone.utc)


async def _add_loaded_service(
    reg: ServiceRegistry,
    name: str,
    vram_gb: float,
    tier: int,
    last_used_offset_seconds: int = 0,
) -> None:
    """Helper: register a loaded service with optional last_used offset (older = more negative)."""
    entry, _ = await reg.register(
        service_name=name,
        base_url=f"http://{name}:8000",
        vram_gb_declared=vram_gb,
        priority_tier=tier,
        keep_alive_seconds=1800,
        initial_state="loaded",
    )
    if last_used_offset_seconds != 0:
        # Directly mutate last_used for test control
        entry.last_used = _utcnow() - timedelta(seconds=abs(last_used_offset_seconds))


def _make_mock_client(
    load_success: bool = True,
    unload_success: bool = True,
) -> MagicMock:
    client = MagicMock()
    client.load = AsyncMock(return_value={"status": "loaded", "vram_gb_actual": 1.0})
    if unload_success:
        client.unload = AsyncMock(return_value={"status": "unloaded"})
    else:
        client.unload = AsyncMock(side_effect=LifecycleError("mock unload failure"))
    return client


@pytest.mark.asyncio
async def test_evict_tier3_before_tier2():
    reg = ServiceRegistry()
    await _add_loaded_service(reg, "tier2-svc", vram_gb=2.0, tier=2)
    await _add_loaded_service(reg, "tier3-svc", vram_gb=1.5, tier=3)

    client = _make_mock_client()
    # Need 1.5 GB — only tier3-svc needs to be evicted
    evicted = await evict_for_vram(1.5, reg, client)

    assert evicted == ["tier3-svc"]
    # tier2-svc should still be loaded
    entry = await reg.get("tier2-svc")
    assert entry.state == "loaded"


@pytest.mark.asyncio
async def test_evict_lru_within_tier():
    reg = ServiceRegistry()
    # Two tier-3 services; older one (used 300s ago) should be evicted first
    await _add_loaded_service(reg, "tier3-old", vram_gb=1.0, tier=3, last_used_offset_seconds=300)
    await _add_loaded_service(reg, "tier3-new", vram_gb=1.0, tier=3, last_used_offset_seconds=60)

    client = _make_mock_client()
    # Need only 1 GB — should evict the older one
    evicted = await evict_for_vram(1.0, reg, client)

    assert evicted == ["tier3-old"]
    entry_new = await reg.get("tier3-new")
    assert entry_new.state == "loaded"


@pytest.mark.asyncio
async def test_tier1_never_evicted():
    reg = ServiceRegistry()
    # Only a tier-1 service available
    await _add_loaded_service(reg, "tier1-svc", vram_gb=3.0, tier=1)

    client = _make_mock_client()
    with pytest.raises(NotEnoughVRAMError):
        await evict_for_vram(3.0, reg, client)

    entry = await reg.get("tier1-svc")
    assert entry.state == "loaded"


@pytest.mark.asyncio
async def test_refcount_protects_from_eviction():
    reg = ServiceRegistry()
    await _add_loaded_service(reg, "busy-tier3", vram_gb=2.0, tier=3)
    # Give it a nonzero refcount
    await reg.increment_refcount("busy-tier3")

    client = _make_mock_client()
    with pytest.raises(NotEnoughVRAMError):
        await evict_for_vram(2.0, reg, client)

    # Service was not evicted
    entry = await reg.get("busy-tier3")
    assert entry.state == "loaded"
    assert entry.reference_count == 1


@pytest.mark.asyncio
async def test_not_enough_vram_error_when_no_candidates():
    reg = ServiceRegistry()
    # No loaded services at all
    client = _make_mock_client()

    with pytest.raises(NotEnoughVRAMError) as exc_info:
        await evict_for_vram(5.0, reg, client)

    assert exc_info.value.needed == 5.0
    assert exc_info.value.freed == 0.0


@pytest.mark.asyncio
async def test_not_enough_vram_error_when_candidates_insufficient():
    reg = ServiceRegistry()
    # Only 1 GB available via eviction, but need 3 GB
    await _add_loaded_service(reg, "tier3-small", vram_gb=1.0, tier=3)

    client = _make_mock_client()
    with pytest.raises(NotEnoughVRAMError) as exc_info:
        await evict_for_vram(3.0, reg, client)

    assert exc_info.value.freed < 3.0


@pytest.mark.asyncio
async def test_evict_multiple_services_until_freed():
    reg = ServiceRegistry()
    await _add_loaded_service(reg, "tier3-a", vram_gb=1.5, tier=3, last_used_offset_seconds=200)
    await _add_loaded_service(reg, "tier3-b", vram_gb=1.5, tier=3, last_used_offset_seconds=100)
    await _add_loaded_service(reg, "tier2-c", vram_gb=2.0, tier=2)

    client = _make_mock_client()
    # Need 2.5 GB — requires evicting both tier-3 services (1.5 + 1.5 = 3.0 >= 2.5)
    evicted = await evict_for_vram(2.5, reg, client)

    assert set(evicted) == {"tier3-a", "tier3-b"}
    entry_c = await reg.get("tier2-c")
    assert entry_c.state == "loaded"


@pytest.mark.asyncio
async def test_eviction_stops_early_once_freed_enough():
    reg = ServiceRegistry()
    await _add_loaded_service(reg, "tier3-big", vram_gb=5.0, tier=3, last_used_offset_seconds=300)
    await _add_loaded_service(reg, "tier2-extra", vram_gb=3.0, tier=2)

    client = _make_mock_client()
    # Need 4 GB — tier3-big (5.0 GB) alone is sufficient
    evicted = await evict_for_vram(4.0, reg, client)

    assert evicted == ["tier3-big"]
    # tier2-extra was not touched
    entry = await reg.get("tier2-extra")
    assert entry.state == "loaded"


@pytest.mark.asyncio
async def test_failed_unload_is_skipped_continues_to_next():
    reg = ServiceRegistry()
    # First candidate will fail to unload; second will succeed
    await _add_loaded_service(reg, "tier3-fail", vram_gb=2.0, tier=3, last_used_offset_seconds=200)
    await _add_loaded_service(reg, "tier3-ok", vram_gb=2.0, tier=3, last_used_offset_seconds=100)

    async def selective_unload(service_name: str, base_url: str) -> dict:
        if service_name == "tier3-fail":
            raise LifecycleError("simulated unload failure")
        return {"status": "unloaded"}

    client = MagicMock()
    client.unload = selective_unload

    evicted = await evict_for_vram(2.0, reg, client)
    assert evicted == ["tier3-ok"]


@pytest.mark.asyncio
async def test_eviction_returns_list_of_evicted_names():
    reg = ServiceRegistry()
    await _add_loaded_service(reg, "svc-x", vram_gb=2.0, tier=3)
    await _add_loaded_service(reg, "svc-y", vram_gb=2.0, tier=3)

    client = _make_mock_client()
    evicted = await evict_for_vram(3.5, reg, client)

    assert isinstance(evicted, list)
    assert len(evicted) == 2


# ── Tier 1 claim preemption tests (issue #156) ────────────────────────────────
#
# These tests exercise the /claim HTTP endpoint, not evict_for_vram directly,
# because the bug is in the claim handler's fast-path logic.
#
# Pattern: register services via POST /register, then POST /claim the Tier 1
# service, then inspect the registry state and response body.


@pytest.fixture
def preempt_client():
    """
    FastAPI TestClient with a fresh app instance and a mock LifecycleClient.

    All services register as "loaded" by default (mock_lc.status returns "loaded").
    Yields (test_client, mock_lifecycle_client).

    Mirrors the client_with_mocks fixture in test_api.py so the app lifespan
    runs properly and _registry / _client are initialised.
    """
    from fastapi.testclient import TestClient

    for mod in ["main", "registry", "eviction", "lifecycle_client", "config", "models"]:
        sys.modules.pop(mod, None)

    mock_lc = MagicMock()
    mock_lc.status = AsyncMock(return_value="loaded")
    mock_lc.load = AsyncMock(return_value={"status": "loaded", "vram_gb_actual": 1.0})
    mock_lc.unload = AsyncMock(return_value={"status": "unloaded"})

    with patch("lifecycle_client.LifecycleClient", return_value=mock_lc):
        import main as app_main
        app_main._client = mock_lc

        with TestClient(app_main.app) as tc:
            yield tc, mock_lc

    for mod in ["main", "registry", "eviction", "lifecycle_client", "config", "models"]:
        sys.modules.pop(mod, None)


def _register(tc, name: str, vram_gb: float, tier: int) -> None:
    resp = tc.post("/register", json={
        "service_name": name,
        "base_url": f"http://{name}:8000",
        "vram_gb_declared": vram_gb,
        "priority_tier": tier,
    })
    assert resp.status_code == 200, f"register failed for {name!r}: {resp.text}"


def test_tier1_claim_evicts_idle_tier2_before_granting(preempt_client):
    """
    Regression test for issue #156.

    A Tier 2 service (back-translator-lv) is loaded and idle (refcount=0).
    When a Tier 1 service (asr-transcription-lv) calls /claim, the supervisor
    must evict the idle Tier 2 service BEFORE granting the claim, so that
    Whisper does not cause a CUDA OOM by loading on top of the back-translator.

    Expected:
    - Tier 2 service state changes from "loaded" to "unloaded"
    - client.unload called for the Tier 2 service
    - /claim returns 200 for the Tier 1 service
    - Tier 2 service name appears in the "evicted" list of the response
    """
    tc, mock_lc = preempt_client

    # Both services register as "loaded" (mock_lc.status returns "loaded")
    _register(tc, "back-translator-lv", vram_gb=2.0, tier=2)
    _register(tc, "asr-transcription-lv", vram_gb=4.0, tier=1)

    # Verify Tier 2 starts loaded, refcount=0 (idle)
    resp_status = tc.get("/status")
    assert resp_status.status_code == 200
    svcs = {s["service_name"]: s for s in resp_status.json()["services"]}
    assert svcs["back-translator-lv"]["state"] == "loaded"
    assert svcs["back-translator-lv"]["reference_count"] == 0
    assert svcs["asr-transcription-lv"]["state"] == "loaded"

    # Reset unload mock call count for clarity
    mock_lc.unload.reset_mock()

    # Tier 1 claims — must trigger Tier 2 eviction
    resp = tc.post("/claim/asr-transcription-lv")
    assert resp.status_code == 200, f"Claim failed: {resp.text}"

    data = resp.json()
    assert "back-translator-lv" in data["evicted"], (
        f"Expected back-translator-lv in evicted list, got: {data['evicted']}"
    )

    # Tier 2 must be unloaded in registry
    resp_status2 = tc.get("/status")
    svcs2 = {s["service_name"]: s for s in resp_status2.json()["services"]}
    assert svcs2["back-translator-lv"]["state"] == "unloaded", (
        f"Expected back-translator-lv to be unloaded, got: {svcs2['back-translator-lv']['state']}"
    )

    # unload must have been called for Tier 2
    mock_lc.unload.assert_called_once()
    call_args = mock_lc.unload.call_args
    assert call_args[0][0] == "back-translator-lv"


def test_tier1_claim_does_not_evict_busy_tier2(preempt_client):
    """
    When a Tier 2 service has refcount > 0 (in-use), a Tier 1 claim must NOT
    evict it.  VRAM contention is preferable to interrupting an active claim.

    Expected:
    - Tier 2 service remains "loaded" with refcount=1
    - client.unload NOT called for the Tier 2 service
    - /claim still returns 200 for the Tier 1 service
    - "evicted" list is empty (no preemption occurred)
    """
    tc, mock_lc = preempt_client

    _register(tc, "sentence-embedder-lv", vram_gb=2.0, tier=2)
    _register(tc, "asr-transcription-lv", vram_gb=4.0, tier=1)

    # Claim the Tier 2 service so its refcount becomes 1 (busy)
    resp_t2 = tc.post("/claim/sentence-embedder-lv")
    assert resp_t2.status_code == 200

    # Verify Tier 2 is loaded and busy
    resp_status = tc.get("/status")
    svcs = {s["service_name"]: s for s in resp_status.json()["services"]}
    assert svcs["sentence-embedder-lv"]["state"] == "loaded"
    assert svcs["sentence-embedder-lv"]["reference_count"] == 1

    mock_lc.unload.reset_mock()

    # Tier 1 claims — must NOT evict the busy Tier 2
    resp = tc.post("/claim/asr-transcription-lv")
    assert resp.status_code == 200, f"Claim failed: {resp.text}"

    data = resp.json()
    assert "sentence-embedder-lv" not in data["evicted"], (
        f"Busy Tier 2 service must not be evicted, got evicted: {data['evicted']}"
    )

    # Tier 2 must still be loaded
    resp_status2 = tc.get("/status")
    svcs2 = {s["service_name"]: s for s in resp_status2.json()["services"]}
    assert svcs2["sentence-embedder-lv"]["state"] == "loaded", (
        f"Expected sentence-embedder-lv still loaded, got: {svcs2['sentence-embedder-lv']['state']}"
    )

    # unload must NOT have been called on the Tier 2 service
    mock_lc.unload.assert_not_called()


def test_tier2_claim_does_not_preempt_other_tier2(preempt_client):
    """
    Tier 2 claims must not trigger preemption of other Tier 2 services.
    This preserves the idle-warm behaviour: multiple Tier 2 services co-exist
    in VRAM and yield only to Tier 1 or through VRAM pressure eviction.

    Expected:
    - Tier 2 service B remains "loaded" after Tier 2 service A calls /claim
    - client.unload NOT called (no proactive preemption)
    - /claim returns 200
    """
    tc, mock_lc = preempt_client

    _register(tc, "back-translator-lv", vram_gb=2.0, tier=2)
    _register(tc, "sentence-embedder-lv", vram_gb=2.0, tier=2)

    # Both are loaded; verify initial state
    resp_status = tc.get("/status")
    svcs = {s["service_name"]: s for s in resp_status.json()["services"]}
    assert svcs["back-translator-lv"]["state"] == "loaded"
    assert svcs["sentence-embedder-lv"]["state"] == "loaded"

    mock_lc.unload.reset_mock()

    # Tier 2 A claims — must NOT evict Tier 2 B
    resp = tc.post("/claim/sentence-embedder-lv")
    assert resp.status_code == 200, f"Claim failed: {resp.text}"

    data = resp.json()
    assert "back-translator-lv" not in data["evicted"], (
        f"Tier 2 claim must not evict other Tier 2 services, got evicted: {data['evicted']}"
    )

    # Tier 2 B must still be loaded
    resp_status2 = tc.get("/status")
    svcs2 = {s["service_name"]: s for s in resp_status2.json()["services"]}
    assert svcs2["back-translator-lv"]["state"] == "loaded", (
        f"Expected back-translator-lv still loaded, got: {svcs2['back-translator-lv']['state']}"
    )

    # No unload calls
    mock_lc.unload.assert_not_called()
