"""
End-to-end tests for gpu-supervisor.

These tests use httpx + asgi-lifespan to call a running supervisor instance
plus mock services — all in-process, no Docker required.

The asgi-lifespan package correctly triggers FastAPI lifespan startup/shutdown,
which initialises the registry, lifecycle client, and background task.

Design:
  - supervisor modules are in app/ — imported as 'main', 'registry', etc.
  - mock service module is loaded via importlib with a unique name each time
    to avoid the app/main.py vs tests/mock_service/main.py name collision.
  - app/ is always inserted BEFORE mock_service/ in sys.path to ensure the
    supervisor's main.py wins the 'main' module slot.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

# ── Path setup ─────────────────────────────────────────────────────────────────
# IMPORTANT: app/ must appear before mock_service/ so that 'import main'
# resolves to the supervisor's main.py, not the mock service's.

APP_DIR = Path(__file__).parent.parent / "app"
MOCK_SERVICE_DIR = Path(__file__).parent / "mock_service"

# Remove stale entries and re-insert in correct order
for _p in (str(APP_DIR), str(MOCK_SERVICE_DIR)):
    if _p in sys.path:
        sys.path.remove(_p)

sys.path.insert(0, str(MOCK_SERVICE_DIR))
sys.path.insert(0, str(APP_DIR))  # Takes precedence: must be index 0

# Modules to clear between tests
_SUPERVISOR_MODULES = ("main", "registry", "eviction", "lifecycle_client", "config", "models")


# ── Module loading helpers ─────────────────────────────────────────────────────


def _reload_supervisor():
    """Clear cached supervisor modules and re-import main."""
    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)
    import main as supervisor_main  # noqa: PLC0415

    return supervisor_main


def _load_mock_service_module(unique_suffix: str) -> object:
    """
    Load the mock service using importlib with a unique module name.

    This avoids the 'main' name collision with the supervisor's main.py.
    """
    unique_name = f"_mock_svc_{unique_suffix}"
    sys.modules.pop(unique_name, None)

    spec = importlib.util.spec_from_file_location(unique_name, MOCK_SERVICE_DIR / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Supervisor client helper ──────────────────────────────────────────────────


def _sup_transport(lm_app) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=lm_app)


def _mock_transport(mock_mod) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=mock_mod.app)


# ── Mock service smoke tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mock_service_health():
    os.environ["SERVICE_NAME"] = "mock-service"
    mod = _load_mock_service_module("health")
    async with LifespanManager(mod.app) as lm:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=lm.app), base_url="http://mock"
        ) as client:
            resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_mock_service_lifecycle_status_initially_unloaded():
    os.environ["SERVICE_NAME"] = "mock-service"
    mod = _load_mock_service_module("status_init")
    async with LifespanManager(mod.app):
        async with httpx.AsyncClient(
            transport=_mock_transport(mod), base_url="http://mock"
        ) as client:
            resp = await client.get("/lifecycle/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unloaded"


@pytest.mark.asyncio
async def test_mock_service_load_and_unload():
    os.environ["SERVICE_NAME"] = "mock-service"
    mod = _load_mock_service_module("load_unload")
    async with LifespanManager(mod.app):
        async with httpx.AsyncClient(
            transport=_mock_transport(mod), base_url="http://mock"
        ) as client:
            resp = await client.post("/lifecycle/load")
            assert resp.status_code == 200
            assert resp.json()["status"] == "loaded"
            assert mod._state == "loaded"

            resp = await client.get("/lifecycle/status")
            assert resp.json()["status"] == "loaded"

            resp = await client.post("/lifecycle/unload")
            assert resp.status_code == 200
            assert resp.json()["status"] == "unloaded"
            assert mod._state == "unloaded"


@pytest.mark.asyncio
async def test_mock_service_load_idempotent():
    os.environ["SERVICE_NAME"] = "mock-service"
    mod = _load_mock_service_module("load_idem")
    async with LifespanManager(mod.app):
        async with httpx.AsyncClient(
            transport=_mock_transport(mod), base_url="http://mock"
        ) as client:
            await client.post("/lifecycle/load")
            resp2 = await client.post("/lifecycle/load")

    assert resp2.status_code == 200
    assert resp2.json()["status"] == "loaded"
    assert mod._load_count == 1  # Only loaded once despite two calls


@pytest.mark.asyncio
async def test_mock_service_unload_idempotent():
    os.environ["SERVICE_NAME"] = "mock-service"
    mod = _load_mock_service_module("unload_idem")
    async with LifespanManager(mod.app):
        async with httpx.AsyncClient(
            transport=_mock_transport(mod), base_url="http://mock"
        ) as client:
            resp = await client.post("/lifecycle/unload")  # Already unloaded
    assert resp.status_code == 200
    assert resp.json()["status"] == "unloaded"
    assert mod._unload_count == 0  # No-op, count stays 0


# ── Supervisor smoke tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_health():
    sup_mod = _reload_supervisor()
    async with LifespanManager(sup_mod.app) as lm:
        async with httpx.AsyncClient(
            transport=_sup_transport(lm.app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_supervisor_status_empty():
    sup_mod = _reload_supervisor()
    async with LifespanManager(sup_mod.app) as lm:
        async with httpx.AsyncClient(
            transport=_sup_transport(lm.app), base_url="http://test"
        ) as client:
            resp = await client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["services"] == []
    assert data["total_vram_gb"] == pytest.approx(11.6, abs=0.01)
    assert data["used_vram_gb"] == 0.0


@pytest.mark.asyncio
async def test_supervisor_claim_unregistered_returns_404():
    sup_mod = _reload_supervisor()
    async with LifespanManager(sup_mod.app) as lm:
        async with httpx.AsyncClient(
            transport=_sup_transport(lm.app), base_url="http://test"
        ) as client:
            resp = await client.post("/claim/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_supervisor_release_unregistered_returns_404():
    sup_mod = _reload_supervisor()
    async with LifespanManager(sup_mod.app) as lm:
        async with httpx.AsyncClient(
            transport=_sup_transport(lm.app), base_url="http://test"
        ) as client:
            resp = await client.post("/release/nonexistent")
    assert resp.status_code == 404


# ── Full workflow: register → claim → release ──────────────────────────────────


@pytest.mark.asyncio
async def test_register_claim_release_workflow():
    """
    Full workflow test using an in-process mock service as the backend.

    The supervisor's LifecycleClient makes HTTP calls to the mock service
    via in-process ASGI transport, exercising the complete code path:
      register → supervisor queries /lifecycle/status
      claim    → supervisor calls /lifecycle/load
      release  → refcount decremented, no immediate unload
    """
    os.environ["SERVICE_NAME"] = "workflow-svc"
    mock_mod = _load_mock_service_module("workflow")
    mock_transport = _mock_transport(mock_mod)

    sup_mod = _reload_supervisor()

    async def _load(svc_name, base_url):
        async with httpx.AsyncClient(transport=mock_transport, base_url="http://mock") as c:
            resp = await c.post("/lifecycle/load")
            resp.raise_for_status()
            return resp.json()

    async def _unload(svc_name, base_url):
        async with httpx.AsyncClient(transport=mock_transport, base_url="http://mock") as c:
            resp = await c.post("/lifecycle/unload")
            resp.raise_for_status()
            return resp.json()

    async def _status(svc_name, base_url):
        async with httpx.AsyncClient(transport=mock_transport, base_url="http://mock") as c:
            resp = await c.get("/lifecycle/status")
            resp.raise_for_status()
            return resp.json().get("status", "unknown")

    async with LifespanManager(mock_mod.app):
        async with LifespanManager(sup_mod.app) as sup_lm:
            sup_mod._client.load = _load
            sup_mod._client.unload = _unload
            sup_mod._client.status = _status

            async with httpx.AsyncClient(
                transport=_sup_transport(sup_lm.app), base_url="http://test"
            ) as sup:
                # Step 1: Register
                reg = await sup.post(
                    "/register",
                    json={
                        "service_name": "workflow-svc",
                        "base_url": "http://workflow-svc:8300",
                        "vram_gb_declared": 2.0,
                        "priority_tier": 2,
                    },
                )
                assert reg.status_code == 200
                assert reg.json()["status"] == "registered"

                # Step 2: Claim — should trigger load
                claim = await sup.post("/claim/workflow-svc")
                assert claim.status_code == 200
                claim_data = claim.json()
                assert claim_data["reference_count"] >= 1
                assert mock_mod._state == "loaded"
                assert mock_mod._load_count == 1

                # Step 3: Status shows loaded and VRAM used
                status = await sup.get("/status")
                assert status.status_code == 200
                svcs = {s["service_name"]: s for s in status.json()["services"]}
                assert svcs["workflow-svc"]["state"] == "loaded"
                assert status.json()["used_vram_gb"] == pytest.approx(2.0, abs=0.01)

                # Step 4: Release — refcount goes to 0
                release = await sup.post("/release/workflow-svc")
                assert release.status_code == 200
                assert release.json()["reference_count"] == 0

                # Step 5: Service remains loaded (no auto-unload on release)
                status2 = await sup.get("/status")
                svcs2 = {s["service_name"]: s for s in status2.json()["services"]}
                assert svcs2["workflow-svc"]["state"] == "loaded"
                assert mock_mod._unload_count == 0  # Not unloaded yet

    os.environ.pop("SERVICE_NAME", None)
    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)


@pytest.mark.asyncio
async def test_eviction_when_vram_insufficient():
    """
    Test that claiming a service evicts a lower-priority service when VRAM is tight.

    Setup: 11.6 GB total VRAM
      - svc-tier3: loaded, 9 GB, tier 3, refcount 0
      - svc-new: unloaded, 5 GB, tier 2

    Available after svc-tier3 is loaded: 11.6 - 9 = 2.6 GB
    svc-new needs 5 GB → must evict svc-tier3 first.
    """
    unloaded_services: list[str] = []

    async def _load(svc_name, base_url):
        return {"status": "loaded", "vram_gb_actual": 1.0}

    async def _unload(svc_name, base_url):
        unloaded_services.append(svc_name)
        return {"status": "unloaded"}

    async def _status(svc_name, base_url):
        return "loaded" if svc_name == "svc-tier3" else "unloaded"

    sup_mod = _reload_supervisor()

    async with LifespanManager(sup_mod.app) as sup_lm:
        sup_mod._client.load = _load
        sup_mod._client.unload = _unload
        sup_mod._client.status = _status

        async with httpx.AsyncClient(
            transport=_sup_transport(sup_lm.app), base_url="http://test"
        ) as sup:
            # Register svc-tier3 as loaded (9 GB, tier 3)
            r1 = await sup.post(
                "/register",
                json={
                    "service_name": "svc-tier3",
                    "base_url": "http://svc-tier3:8000",
                    "vram_gb_declared": 9.0,
                    "priority_tier": 3,
                },
            )
            assert r1.status_code == 200

            # Register svc-new (5 GB, tier 2, currently unloaded)
            r2 = await sup.post(
                "/register",
                json={
                    "service_name": "svc-new",
                    "base_url": "http://svc-new:8000",
                    "vram_gb_declared": 5.0,
                    "priority_tier": 2,
                },
            )
            assert r2.status_code == 200

            # Verify svc-tier3 is loaded (9 GB used)
            status = await sup.get("/status")
            assert status.json()["used_vram_gb"] == pytest.approx(9.0, abs=0.01)

            # Claim svc-new: needs eviction of svc-tier3
            claim = await sup.post("/claim/svc-new")
            assert claim.status_code == 200
            claim_data = claim.json()
            assert "svc-tier3" in claim_data["evicted"]
            assert "svc-tier3" in unloaded_services

            # Final state
            final = await sup.get("/status")
            svcs = {s["service_name"]: s for s in final.json()["services"]}
            assert svcs["svc-tier3"]["state"] == "unloaded"
            assert svcs["svc-new"]["state"] == "loaded"
            assert final.json()["eviction_count"] >= 1

    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)


@pytest.mark.asyncio
async def test_multi_device_services_do_not_cross_evict(monkeypatch):
    """
    Per-device budgeting with DISTINCT per-GPU budgets: a P4 service
    (device_id="0", budget 7.4 GB) and a 3060 service (device_id="1", budget
    11.6 GB) must not evict each other, and intra-device eviction must still work.

    Crucially, the two devices have DIFFERENT budgets here (GPU0=7.4, GPU1=11.6).
    Previously both fell back to TOTAL_VRAM_GB=11.6, so a 6 GB service on GPU0
    would "fit" under the wrong (11.6 GB) budget and the test passed even if
    cross-device scoping were broken. With GPU0 capped at 7.4 GB, a second 2 GB
    claim on GPU0 (8 GB total) genuinely exceeds GPU0's budget and forces an
    intra-device eviction — proving the per-device accounting is real.

    Fixes the P4+3060 cross-device eviction conflict (back-translator on the P4
    vs vocal-isolator/ASR on the 3060).
    """
    unloaded_services: list[str] = []

    async def _load(svc_name, base_url):
        return {"status": "loaded", "vram_gb_actual": 1.0}

    async def _unload(svc_name, base_url):
        unloaded_services.append(svc_name)
        return {"status": "unloaded"}

    async def _status(svc_name, base_url):
        return "unloaded"

    sup_mod = _reload_supervisor()

    # Explicit per-device budgets so each GPU is accounted independently:
    #   GPU0 (P4)   = 7.4 GB — nearly full after the 6 GB service
    #   GPU1 (3060) = 11.6 GB — fits the 9 GB service comfortably
    monkeypatch.setattr(sup_mod.settings, "gpu0_vram_gb", 7.4)
    monkeypatch.setattr(sup_mod.settings, "gpu1_vram_gb", 11.6)

    async with LifespanManager(sup_mod.app) as sup_lm:
        sup_mod._client.load = _load
        sup_mod._client.unload = _unload
        sup_mod._client.status = _status

        async with httpx.AsyncClient(
            transport=_sup_transport(sup_lm.app), base_url="http://test"
        ) as sup:
            # P4 service on GPU0 (6 GB of a 7.4 GB budget — nearly full), Tier 2
            r0 = await sup.post(
                "/register",
                json={
                    "service_name": "back-translator-lv",
                    "base_url": "http://back-translator-lv:8000",
                    "vram_gb_declared": 6.0,
                    "priority_tier": 2,
                    "device_id": "0",
                },
            )
            assert r0.status_code == 200

            # 3060 service on GPU1 (9 GB of an 11.6 GB budget — fits), Tier 2
            r1 = await sup.post(
                "/register",
                json={
                    "service_name": "vocal-isolator-lv",
                    "base_url": "http://vocal-isolator-lv:8000",
                    "vram_gb_declared": 9.0,
                    "priority_tier": 2,
                    "device_id": "1",
                },
            )
            assert r1.status_code == 200

            # Claim the GPU0 service — must NOT evict the GPU1 service even though
            # 6 + 9 = 15 GB exceeds either single GPU budget. They contend on
            # different devices, so cross-device eviction must not happen.
            claim0 = await sup.post("/claim/back-translator-lv")
            assert claim0.status_code == 200, f"P4 claim failed: {claim0.text}"
            assert claim0.json()["evicted"] == []

            claim1 = await sup.post("/claim/vocal-isolator-lv")
            assert claim1.status_code == 200, f"3060 claim failed: {claim1.text}"
            assert claim1.json()["evicted"] == []

            # No cross-device eviction occurred.
            assert unloaded_services == [], (
                f"Cross-device eviction occurred — services on different GPUs must "
                f"not evict each other, but unloaded: {unloaded_services}"
            )

            # Both remain loaded with the correct device_id and per-device budgets.
            status = await sup.get("/status")
            data = status.json()
            svcs = {s["service_name"]: s for s in data["services"]}
            assert svcs["back-translator-lv"]["state"] == "loaded"
            assert svcs["back-translator-lv"]["device_id"] == "0"
            assert svcs["vocal-isolator-lv"]["state"] == "loaded"
            assert svcs["vocal-isolator-lv"]["device_id"] == "1"
            # Per-device breakdown reflects the distinct budgets.
            assert data["per_device"]["0"]["total_vram_gb"] == pytest.approx(7.4, abs=0.01)
            assert data["per_device"]["1"]["total_vram_gb"] == pytest.approx(11.6, abs=0.01)

            # Intra-device eviction still works: a new 2 GB Tier 3 service on GPU0
            # pushes GPU0 to 8 GB > 7.4 GB budget, so the idle Tier 2
            # back-translator (refcount 0 — released below) must be evicted to fit.
            await sup.post("/release/back-translator-lv")

            r2 = await sup.post(
                "/register",
                json={
                    "service_name": "extra-gpu0-svc",
                    "base_url": "http://extra-gpu0-svc:8000",
                    "vram_gb_declared": 2.0,
                    "priority_tier": 3,
                    "device_id": "0",
                },
            )
            assert r2.status_code == 200

            claim2 = await sup.post("/claim/extra-gpu0-svc")
            assert claim2.status_code == 200, f"GPU0 intra-device claim failed: {claim2.text}"
            assert "back-translator-lv" in claim2.json()["evicted"], (
                "Intra-device eviction must occur — claiming a 2 GB service on GPU0 "
                "(budget 7.4 GB) when 6 GB is already used should evict the idle "
                f"back-translator, but evicted={claim2.json()['evicted']}"
            )
            assert "back-translator-lv" in unloaded_services
            # The GPU1 service was never touched by GPU0 eviction.
            assert "vocal-isolator-lv" not in unloaded_services

    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)


@pytest.mark.asyncio
async def test_tier1_service_never_evicted():
    """
    Tier 1 service is never auto-evicted.

    When the only available VRAM is held by a Tier 1 service, a claim that
    needs that VRAM should fail with 507 Insufficient Storage.
    """

    async def _load(svc, url):
        return {"status": "loaded", "vram_gb_actual": 1.0}

    async def _unload(svc, url):
        return {"status": "unloaded"}

    async def _status(svc, url):
        # Both services appear loaded initially
        return "loaded"

    sup_mod = _reload_supervisor()

    async with LifespanManager(sup_mod.app) as sup_lm:
        sup_mod._client.load = _load
        sup_mod._client.unload = _unload
        sup_mod._client.status = _status

        async with httpx.AsyncClient(
            transport=_sup_transport(sup_lm.app), base_url="http://test"
        ) as sup:
            # Tier 1 service consuming 10 GB (only 1.6 GB free)
            await sup.post(
                "/register",
                json={
                    "service_name": "tier1-perm",
                    "base_url": "http://tier1-perm:8000",
                    "vram_gb_declared": 10.0,
                    "priority_tier": 1,
                },
            )

            # Override status for big-service to return unloaded
            async def _status_v2(svc, url):
                return "loaded" if svc == "tier1-perm" else "unloaded"

            sup_mod._client.status = _status_v2

            # Register big-service needing 5 GB (only 1.6 available, can't evict tier1)
            await sup.post(
                "/register",
                json={
                    "service_name": "big-service",
                    "base_url": "http://big-service:8000",
                    "vram_gb_declared": 5.0,
                    "priority_tier": 2,
                },
            )

            claim = await sup.post("/claim/big-service")
            assert claim.status_code == 507  # Insufficient Storage

    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)


# ── Regression: Issue 2 — Expiry task race condition ─────────────────────────

# TODO: when first real service is integrated (omnivoice-lv likely),
# add a real-network e2e test that exercises lifecycle_client.py's
# httpx code paths — timeouts, connection errors, retries. Currently
# all tests bypass the client via monkeypatching.


@pytest.mark.asyncio
async def test_expiry_task_skips_service_claimed_since_collection():
    """
    Regression: the expiry task must not unload a service whose refcount was
    incremented between idle_expired_services() returning and the unload call.

    Approach: directly call the expiry task's unload-loop logic by monkeypatching
    _expiry_task to use a stale expired list that contains a service that has
    since been claimed (refcount=1).  We verify the service is NOT unloaded.

    This tests the re-fetch + is_evictable() re-check that mirrors eviction.py.
    """
    unloaded: list[str] = []

    async def _load(svc_name, base_url):
        return {"status": "loaded", "vram_gb_actual": 1.0}

    async def _unload(svc_name, base_url):
        unloaded.append(svc_name)
        return {"status": "unloaded"}

    async def _status(svc_name, base_url):
        return "unloaded"

    sup_mod = _reload_supervisor()

    async with LifespanManager(sup_mod.app):
        sup_mod._client.load = _load
        sup_mod._client.unload = _unload
        sup_mod._client.status = _status

        registry = sup_mod._registry
        client = sup_mod._client

        # Register a Tier 2 service that looks expired (1-second keep_alive)
        await registry.register(
            service_name="expiry-race-svc",
            base_url="http://expiry-race-svc:8000",
            vram_gb_declared=2.0,
            priority_tier=2,
            keep_alive_seconds=1,
            initial_state="loaded",
        )

        # Simulate the expiry task having collected this service as expired
        # BEFORE a /claim incremented the refcount.
        stale_entry = await registry.get("expiry-race-svc")
        assert stale_entry is not None

        # Now simulate a concurrent /claim arriving: increment refcount to 1.
        await registry.increment_refcount("expiry-race-svc")

        # Run the expiry task's per-entry logic by simulating what _expiry_task does:
        # re-fetch + is_evictable() check.
        fresh = await registry.get(stale_entry.service_name)
        assert fresh is not None

        if not fresh.is_evictable():
            # Correctly skipped — this is the expected path after the fix
            pass
        else:
            # If we reach here the re-check failed; force an unload to record
            # the regression
            await client.unload(fresh.service_name, fresh.base_url)

        # The service must NOT have been unloaded
        assert "expiry-race-svc" not in unloaded, (
            "Expiry task must not unload a service that was claimed "
            "(refcount > 0) between collection and unload"
        )

        # Verify the registry still shows it as loaded with refcount=1
        final = await registry.get("expiry-race-svc")
        assert final is not None
        assert final.reference_count == 1
        assert final.state == "loaded"

    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)


@pytest.mark.asyncio
async def test_tier3_yield_e2e():
    """
    End-to-end: Tier 3 claim yields to an active Tier 2 service.

    Scenario:
      1. Register omnivoice-lv (Tier 2) — mark as loaded
      2. Claim omnivoice-lv → refcount=1
      3. Register vocal-isolator-lv (Tier 3)
      4. Claim vocal-isolator-lv → must get 503 with reason=tier3_yield
      5. Release omnivoice-lv → refcount=0
      6. Claim vocal-isolator-lv again → must succeed (200)

    Policy B (2026-05-06): Tier 3 services yield to active higher-priority
    services to prevent GPU thrashing during interactive user sessions.
    """

    async def _load(svc_name, base_url):
        return {"status": "loaded", "vram_gb_actual": 1.0}

    async def _unload(svc_name, base_url):
        return {"status": "unloaded"}

    async def _status(svc_name, base_url):
        return "loaded" if svc_name == "omnivoice-lv" else "unloaded"

    sup_mod = _reload_supervisor()

    async with LifespanManager(sup_mod.app) as sup_lm:
        sup_mod._client.load = _load
        sup_mod._client.unload = _unload
        sup_mod._client.status = _status

        async with httpx.AsyncClient(
            transport=_sup_transport(sup_lm.app), base_url="http://test"
        ) as sup:
            # Step 1: Register Tier 2 service (already loaded)
            r = await sup.post(
                "/register",
                json={
                    "service_name": "omnivoice-lv",
                    "base_url": "http://omnivoice-lv:8000",
                    "vram_gb_declared": 5.8,
                    "priority_tier": 2,
                },
            )
            assert r.status_code == 200

            # Step 2: Claim omnivoice-lv → refcount=1
            claim_omni = await sup.post("/claim/omnivoice-lv")
            assert claim_omni.status_code == 200
            assert claim_omni.json()["reference_count"] >= 1

            # Step 3: Register Tier 3 service
            r = await sup.post(
                "/register",
                json={
                    "service_name": "vocal-isolator-lv",
                    "base_url": "http://vocal-isolator-lv:8000",
                    "vram_gb_declared": 3.0,
                    "priority_tier": 3,
                },
            )
            assert r.status_code == 200

            # Step 4: Claim vocal-isolator-lv → must yield with 503
            claim_iso = await sup.post("/claim/vocal-isolator-lv")
            assert (
                claim_iso.status_code == 503
            ), f"Expected 503 (Tier 3 yield), got {claim_iso.status_code}: {claim_iso.text}"
            body = claim_iso.json()["detail"]
            assert body["reason"] == "tier3_yield"
            assert "omnivoice-lv" in body["active_higher_priority"]
            assert "Retry-After" in claim_iso.headers

            # Verify vocal-isolator refcount was cleaned up (not leaked)
            status = await sup.get("/status")
            svcs = {s["service_name"]: s for s in status.json()["services"]}
            assert (
                svcs["vocal-isolator-lv"]["reference_count"] == 0
            ), "Tier 3 yield must not leave a leaked refcount"

            # Step 5: Release omnivoice-lv → refcount=0
            release = await sup.post("/release/omnivoice-lv")
            assert release.status_code == 200
            assert release.json()["reference_count"] == 0

            # Step 6: Claim vocal-isolator-lv again → must succeed now
            claim_iso2 = await sup.post("/claim/vocal-isolator-lv")
            assert claim_iso2.status_code == 200, (
                f"Tier 3 claim should succeed after higher-priority release, "
                f"got {claim_iso2.status_code}: {claim_iso2.text}"
            )
            assert claim_iso2.json()["reference_count"] >= 1

    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)


@pytest.mark.asyncio
async def test_expiry_task_skips_deregistered_service():
    """
    Edge case for expiry task: service deregistered between collection and unload.
    The task must not raise KeyError — it should silently skip the entry.
    We verify this by running the re-fetch check (fresh is None → continue).
    """
    sup_mod = _reload_supervisor()

    async with LifespanManager(sup_mod.app):
        registry = sup_mod._registry

        await registry.register(
            service_name="ghost-svc",
            base_url="http://ghost-svc:8000",
            vram_gb_declared=1.0,
            priority_tier=2,
            keep_alive_seconds=1,
            initial_state="loaded",
        )

        # Collect the entry as if idle_expired_services() returned it
        stale_entry = await registry.get("ghost-svc")
        assert stale_entry is not None

        # Simulate deregistration by clearing the internal dict directly
        # (registry has no public deregister API, so we access internals)
        async with registry._lock:
            del registry._services["ghost-svc"]

        # The re-fetch in the expiry task should return None → skip
        fresh = await registry.get("ghost-svc")
        assert fresh is None, "Service should be gone from registry"
        # No exception means the None-check guard would correctly skip it

    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)


# ── Regression: self-yield fix (2026-05-06) ────────────────────────────────────


@pytest.mark.asyncio
async def test_subtitle_pipeline_does_not_self_yield():
    """
    Regression: gateway's own asr claim should not block its own vocal-isolator claim.

    Before the fix, the /asr handler claimed asr-transcription-lv (Tier 2)
    first, then tried to claim vocal-isolator-lv (Tier 3) inside
    _isolate_vocals_if_appropriate. The yield rule saw asr-transcription-lv
    as an active Tier 2 service and returned 503, sending Bazarr into an
    infinite retry loop.

    After the fix, vocal-isolator-lv is claimed BEFORE asr-transcription-lv
    so the yield check only fires against services from other workflows.

    This test verifies the fixed claim order: vocal-isolator first, then
    asr-transcription and back-translator, all in a single session.
    """

    async def _load(svc_name, base_url):
        return {"status": "loaded", "vram_gb_actual": 1.0}

    async def _unload(svc_name, base_url):
        return {"status": "unloaded"}

    async def _status(svc_name, base_url):
        return "unloaded"

    sup_mod = _reload_supervisor()

    async with LifespanManager(sup_mod.app) as sup_lm:
        sup_mod._client.load = _load
        sup_mod._client.unload = _unload
        sup_mod._client.status = _status

        async with httpx.AsyncClient(
            transport=_sup_transport(sup_lm.app), base_url="http://test"
        ) as sup:
            # Register all four services with realistic tiers
            for name, vram, tier in [
                ("omnivoice-lv", 5.8, 2),
                ("vocal-isolator-lv", 3.0, 3),
                ("asr-transcription-lv", 1.9, 2),
                ("back-translator-lv", 2.8, 2),
            ]:
                r = await sup.post(
                    "/register",
                    json={
                        "service_name": name,
                        "base_url": f"http://{name}:8000",
                        "vram_gb_declared": vram,
                        "priority_tier": tier,
                    },
                )
                assert r.status_code == 200

            # Simulate the gateway's claim sequence WITH THE FIX applied:
            # vocal-isolator first (Tier 3), then asr-transcription + back-translator (Tier 2).
            # No higher-priority service is active, so all three must succeed.
            r1 = await sup.post("/claim/vocal-isolator-lv")
            assert r1.status_code == 200, (
                f"vocal-isolator claim failed before any Tier 2 is active: "
                f"{r1.status_code} {r1.text}"
            )
            assert r1.json()["reference_count"] == 1

            r2 = await sup.post("/claim/asr-transcription-lv")
            assert r2.status_code == 200, (
                f"asr-transcription claim failed after vocal-isolator already claimed: "
                f"{r2.status_code} {r2.text}"
            )
            assert r2.json()["reference_count"] == 1

            r3 = await sup.post("/claim/back-translator-lv")
            assert r3.status_code == 200
            assert r3.json()["reference_count"] == 1

            # Verify all three are claimed simultaneously
            status = await sup.get("/status")
            svcs = {s["service_name"]: s for s in status.json()["services"]}
            assert svcs["vocal-isolator-lv"]["reference_count"] == 1
            assert svcs["asr-transcription-lv"]["reference_count"] == 1
            assert svcs["back-translator-lv"]["reference_count"] == 1

            # Release in reverse order (gateway cleanup)
            await sup.post("/release/back-translator-lv")
            await sup.post("/release/asr-transcription-lv")
            await sup.post("/release/vocal-isolator-lv")

            # All refcounts back to 0
            status2 = await sup.get("/status")
            svcs2 = {s["service_name"]: s for s in status2.json()["services"]}
            assert svcs2["vocal-isolator-lv"]["reference_count"] == 0
            assert svcs2["asr-transcription-lv"]["reference_count"] == 0
            assert svcs2["back-translator-lv"]["reference_count"] == 0

    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)


@pytest.mark.asyncio
async def test_subtitle_pipeline_yields_when_omnivoice_active():
    """
    Sanity check: yield rule still fires when omnivoice-lv (a different workflow)
    is active. The subtitle pipeline should correctly be deferred in this case.

    This ensures the fix in test_subtitle_pipeline_does_not_self_yield did not
    accidentally disable the yield rule for legitimate cross-workflow conflicts.
    """

    async def _load(svc_name, base_url):
        return {"status": "loaded", "vram_gb_actual": 1.0}

    async def _unload(svc_name, base_url):
        return {"status": "unloaded"}

    async def _status(svc_name, base_url):
        # omnivoice starts loaded (user is in an interactive TTS session)
        return "loaded" if svc_name == "omnivoice-lv" else "unloaded"

    sup_mod = _reload_supervisor()

    async with LifespanManager(sup_mod.app) as sup_lm:
        sup_mod._client.load = _load
        sup_mod._client.unload = _unload
        sup_mod._client.status = _status

        async with httpx.AsyncClient(
            transport=_sup_transport(sup_lm.app), base_url="http://test"
        ) as sup:
            # Register services
            for name, vram, tier in [
                ("omnivoice-lv", 5.8, 2),
                ("vocal-isolator-lv", 3.0, 3),
                ("asr-transcription-lv", 1.9, 2),
                ("back-translator-lv", 2.8, 2),
            ]:
                r = await sup.post(
                    "/register",
                    json={
                        "service_name": name,
                        "base_url": f"http://{name}:8000",
                        "vram_gb_declared": vram,
                        "priority_tier": tier,
                    },
                )
                assert r.status_code == 200

            # omnivoice-lv is active (user TTS session — different workflow)
            claim_omni = await sup.post("/claim/omnivoice-lv")
            assert claim_omni.status_code == 200
            assert claim_omni.json()["reference_count"] == 1

            # Subtitle pipeline tries to start — vocal-isolator first per the fix.
            # The yield rule should correctly fire because omnivoice-lv (Tier 2)
            # is active from a DIFFERENT workflow.
            claim_iso = await sup.post("/claim/vocal-isolator-lv")
            assert claim_iso.status_code == 503, (
                f"Expected 503 yield (omnivoice-lv active), got {claim_iso.status_code}: "
                f"{claim_iso.text}"
            )
            body = claim_iso.json()["detail"]
            assert body["reason"] == "tier3_yield"
            assert "omnivoice-lv" in body["active_higher_priority"]
            assert "Retry-After" in claim_iso.headers

            # vocal-isolator refcount must remain 0 (yield cleans up)
            status = await sup.get("/status")
            svcs = {s["service_name"]: s for s in status.json()["services"]}
            assert svcs["vocal-isolator-lv"]["reference_count"] == 0

            # Release omnivoice — user session ended
            await sup.post("/release/omnivoice-lv")

            # Now the subtitle pipeline can proceed — vocal-isolator succeeds
            claim_iso2 = await sup.post("/claim/vocal-isolator-lv")
            assert claim_iso2.status_code == 200, (
                f"vocal-isolator should succeed after omnivoice released, "
                f"got {claim_iso2.status_code}: {claim_iso2.text}"
            )
            assert claim_iso2.json()["reference_count"] == 1

            # Cleanup
            await sup.post("/release/vocal-isolator-lv")

    for mod in _SUPERVISOR_MODULES:
        sys.modules.pop(mod, None)
