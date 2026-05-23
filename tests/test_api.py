"""
API integration tests using FastAPI TestClient + mocked lifecycle_client.

Tests exercise the HTTP API surface without making real service calls.
"""

from __future__ import annotations

import sys
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure app directory is on path
APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR))


def make_mock_client(
    status_return: str = "unloaded",
    load_success: bool = True,
    unload_success: bool = True,
) -> MagicMock:
    """Create a mock LifecycleClient with controllable behaviour."""
    client = MagicMock()
    client.status = AsyncMock(return_value=status_return)
    if load_success:
        client.load = AsyncMock(return_value={"status": "loaded", "vram_gb_actual": 1.0})
    else:
        from lifecycle_client import LifecycleError

        client.load = AsyncMock(side_effect=LifecycleError("mock load failure"))
    if unload_success:
        client.unload = AsyncMock(return_value={"status": "unloaded"})
    else:
        from lifecycle_client import LifecycleError

        client.unload = AsyncMock(side_effect=LifecycleError("mock unload failure"))
    return client


@pytest.fixture
def client_with_mocks():
    """
    Return a FastAPI TestClient with LifecycleClient mocked out.

    Re-imports main each time to get a fresh application state.
    Yields (test_client, mock_lifecycle_client).
    """
    from fastapi.testclient import TestClient

    # Remove cached main module to get fresh state each test
    for mod in ["main", "registry", "eviction", "lifecycle_client", "config", "models"]:
        sys.modules.pop(mod, None)

    mock_lc = make_mock_client()

    with patch("lifecycle_client.LifecycleClient", return_value=mock_lc):
        import main as app_main

        app_main._client = mock_lc

        with TestClient(app_main.app) as tc:
            yield tc, mock_lc

    # Cleanup
    for mod in ["main", "registry", "eviction", "lifecycle_client", "config", "models"]:
        sys.modules.pop(mod, None)


def register_service(
    tc,
    name: str = "test-svc",
    base_url: str = "http://test-svc:8000",
    vram_gb: float = 2.0,
    tier: int = 2,
) -> dict:
    """Helper: register a service via the API."""
    resp = tc.post(
        "/register",
        json={
            "service_name": name,
            "base_url": base_url,
            "vram_gb_declared": vram_gb,
            "priority_tier": tier,
        },
    )
    assert resp.status_code == 200, f"Register failed: {resp.text}"
    return resp.json()


# ── /health ────────────────────────────────────────────────────────────────────


def test_health_returns_200(client_with_mocks):
    tc, _ = client_with_mocks
    resp = tc.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "registered_services" in data
    assert "loaded_services" in data


# ── /register ─────────────────────────────────────────────────────────────────


def test_register_new_service(client_with_mocks):
    tc, _ = client_with_mocks
    data = register_service(tc)
    assert data["status"] == "registered"
    assert data["service_name"] == "test-svc"


def test_register_second_time_returns_updated(client_with_mocks):
    tc, _ = client_with_mocks
    register_service(tc, name="svc-a", vram_gb=2.0)
    resp = tc.post(
        "/register",
        json={
            "service_name": "svc-a",
            "base_url": "http://svc-a:8000",
            "vram_gb_declared": 3.0,
            "priority_tier": 2,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "already_registered_updated"


def test_register_invalid_tier_422(client_with_mocks):
    tc, _ = client_with_mocks
    resp = tc.post(
        "/register",
        json={
            "service_name": "svc-x",
            "base_url": "http://svc-x:8000",
            "vram_gb_declared": 1.0,
            "priority_tier": 99,  # invalid
        },
    )
    assert resp.status_code == 422


def test_register_zero_vram_422(client_with_mocks):
    tc, _ = client_with_mocks
    resp = tc.post(
        "/register",
        json={
            "service_name": "svc-x",
            "base_url": "http://svc-x:8000",
            "vram_gb_declared": 0.0,  # invalid
            "priority_tier": 2,
        },
    )
    assert resp.status_code == 422


def test_register_vram_exceeds_total_422(client_with_mocks):
    tc, _ = client_with_mocks
    resp = tc.post(
        "/register",
        json={
            "service_name": "svc-huge",
            "base_url": "http://svc-huge:8000",
            "vram_gb_declared": 100.0,  # exceeds 11.6 GB total
            "priority_tier": 2,
        },
    )
    assert resp.status_code == 422


def test_register_tier1_with_keep_alive_override_422(client_with_mocks):
    """
    Regression: Tier 1 services must not accept a keep_alive_seconds override.
    A user could pass keep_alive_seconds=300 for a Tier 1 service in good faith,
    which before the fix would silently allow the service to be expired.
    """
    tc, _ = client_with_mocks
    resp = tc.post(
        "/register",
        json={
            "service_name": "tier1-svc",
            "base_url": "http://tier1-svc:8000",
            "vram_gb_declared": 2.0,
            "priority_tier": 1,
            "keep_alive_seconds": 300,  # must be rejected for Tier 1
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "Tier 1" in detail, f"Error message should mention 'Tier 1', got: {detail!r}"


def test_register_tier1_without_keep_alive_override_succeeds(client_with_mocks):
    """Tier 1 registration without keep_alive_seconds override must succeed."""
    tc, _ = client_with_mocks
    resp = tc.post(
        "/register",
        json={
            "service_name": "tier1-no-override",
            "base_url": "http://tier1-no-override:8000",
            "vram_gb_declared": 2.0,
            "priority_tier": 1,
            # No keep_alive_seconds — should use tier default
        },
    )
    assert resp.status_code == 200


# ── /claim ────────────────────────────────────────────────────────────────────


def test_claim_unregistered_returns_404(client_with_mocks):
    tc, _ = client_with_mocks
    resp = tc.post("/claim/nonexistent-service")
    assert resp.status_code == 404


def test_claim_loaded_service_returns_immediately(client_with_mocks):
    tc, mock_lc = client_with_mocks

    # Register as already loaded
    mock_lc.status = AsyncMock(return_value="loaded")
    register_service(tc, name="already-loaded")

    resp = tc.post("/claim/already-loaded")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "loaded"
    assert data["reference_count"] >= 1
    assert data["evicted"] == []
    # Load should NOT have been called since service was already loaded
    mock_lc.load.assert_not_called()


def test_claim_unloaded_service_triggers_load(client_with_mocks):
    tc, mock_lc = client_with_mocks

    mock_lc.status = AsyncMock(return_value="unloaded")
    register_service(tc, name="svc-to-load", vram_gb=1.0, tier=2)

    resp = tc.post("/claim/svc-to-load")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("loaded", "evicted_to_load")
    assert data["reference_count"] >= 1
    mock_lc.load.assert_called_once()


def test_claim_increments_refcount(client_with_mocks):
    tc, mock_lc = client_with_mocks

    mock_lc.status = AsyncMock(return_value="unloaded")
    register_service(tc, name="refcount-svc", vram_gb=1.0)

    resp1 = tc.post("/claim/refcount-svc")
    assert resp1.status_code == 200
    count1 = resp1.json()["reference_count"]

    # Second claim: service is now loaded (mock returns loaded)
    mock_lc.status = AsyncMock(return_value="loaded")
    resp2 = tc.post("/claim/refcount-svc")
    assert resp2.status_code == 200
    count2 = resp2.json()["reference_count"]

    assert count2 > count1


def test_claim_load_failure_returns_502(client_with_mocks):
    tc, mock_lc = client_with_mocks

    from lifecycle_client import LifecycleError

    mock_lc.status = AsyncMock(return_value="unloaded")
    mock_lc.load = AsyncMock(side_effect=LifecycleError("GPU OOM"))

    register_service(tc, name="failing-svc", vram_gb=1.0)

    resp = tc.post("/claim/failing-svc")
    assert resp.status_code == 502
    assert "failing-svc" in resp.json()["detail"]


# ── /release ──────────────────────────────────────────────────────────────────


def test_release_unregistered_returns_404(client_with_mocks):
    tc, _ = client_with_mocks
    resp = tc.post("/release/nonexistent-service")
    assert resp.status_code == 404


def test_release_decrements_refcount(client_with_mocks):
    tc, mock_lc = client_with_mocks

    mock_lc.status = AsyncMock(return_value="unloaded")
    register_service(tc, name="release-svc", vram_gb=1.0)

    # Claim to get refcount=1
    tc.post("/claim/release-svc")

    # Release
    resp = tc.post("/release/release-svc")
    assert resp.status_code == 200
    assert resp.json()["reference_count"] == 0


def test_release_does_not_unload_immediately(client_with_mocks):
    tc, mock_lc = client_with_mocks

    mock_lc.status = AsyncMock(return_value="unloaded")
    register_service(tc, name="release-noauto-svc", vram_gb=1.0)
    tc.post("/claim/release-noauto-svc")
    tc.post("/release/release-noauto-svc")

    # Unload should only have been called if eviction happened, not on release
    mock_lc.unload.assert_not_called()


# ── /status ───────────────────────────────────────────────────────────────────


def test_status_returns_supervisor_state(client_with_mocks):
    tc, _ = client_with_mocks
    resp = tc.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "services" in data
    assert "total_vram_gb" in data
    assert "used_vram_gb" in data
    assert "available_vram_gb" in data
    assert "started_at" in data
    assert "eviction_count" in data


def test_status_shows_registered_services(client_with_mocks):
    tc, mock_lc = client_with_mocks
    mock_lc.status = AsyncMock(return_value="unloaded")

    register_service(tc, name="svc-1", vram_gb=2.0)
    register_service(tc, name="svc-2", vram_gb=3.0)

    resp = tc.get("/status")
    data = resp.json()
    names = [s["service_name"] for s in data["services"]]
    assert "svc-1" in names
    assert "svc-2" in names


def test_status_vram_accounting(client_with_mocks):
    tc, mock_lc = client_with_mocks

    # Register as loaded so VRAM is counted
    mock_lc.status = AsyncMock(return_value="loaded")
    register_service(tc, name="loaded-svc", vram_gb=3.0)

    resp = tc.get("/status")
    data = resp.json()
    assert data["used_vram_gb"] == pytest.approx(3.0, abs=0.01)
    assert data["available_vram_gb"] == pytest.approx(11.6 - 3.0, abs=0.01)


def test_status_empty_initially(client_with_mocks):
    tc, _ = client_with_mocks
    resp = tc.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["services"] == []
    assert data["used_vram_gb"] == 0.0


# ── Policy A: Tier 1 keep-alive and idle-expiry ───────────────────────────────


def test_tier1_service_never_idle_expired(client_with_mocks):
    """
    Tier 1 service must be excluded from idle_expired_services() even when its
    keep_alive period has nominally elapsed.

    Regression guard for Policy A (2026-05-06): fluency-gate-roberta-lv was
    promoted to Tier 1 so it never gets auto-evicted mid-session.

    The ServiceEntry.is_idle_expired() method has an explicit Tier 1 guard
    (see registry.py). This test verifies the supervisor's registration path
    honours that contract by checking that a Tier 1 service with a very short
    notional keep_alive (set via the registry directly) never appears expired.
    """
    import sys
    from pathlib import Path

    # Access the registry model directly to set a past last_used timestamp.
    app_dir = Path(__file__).parent.parent / "app"
    sys.path.insert(0, str(app_dir))
    from datetime import datetime, timedelta

    from registry import ServiceEntry

    # Construct a Tier 1 service entry whose last_used is 1 hour ago
    entry = ServiceEntry(
        service_name="fluency-gate-roberta-lv",
        base_url="http://fluency-gate-roberta-lv:8005",
        vram_gb_declared=1.3,
        priority_tier=1,
        keep_alive_seconds=60,  # would normally expire after 60 s
        state="loaded",
        reference_count=0,
        last_used=datetime.now(tz=UTC) - timedelta(hours=1),
    )

    # Tier 1 must never report as idle-expired regardless of keep_alive_seconds
    # or how long ago last_used was.
    now = datetime.now(tz=UTC)
    assert not entry.is_idle_expired(now), (
        "Tier 1 service reported as idle_expired — this would allow auto-eviction "
        "of fluency-gate-roberta-lv mid-session (Policy A violation)"
    )

    # Sanity: same entry at Tier 2 WOULD be expired
    entry.priority_tier = 2
    assert entry.is_idle_expired(
        now
    ), "Tier 2 service with elapsed keep_alive should be idle_expired"


# ── Policy B: Tier 3 yield tests ─────────────────────────────────────────────


def test_tier3_claim_yields_to_active_tier2(client_with_mocks):
    """
    When a Tier 2 service has refcount > 0, a Tier 3 claim must return 503
    with reason=tier3_yield and must NOT trigger eviction.

    Policy B (2026-05-06): Bazarr's vocal-isolator pipeline waits politely
    while the user is actively using interactive services.
    """
    tc, mock_lc = client_with_mocks

    # Register and claim a Tier 2 service (omnivoice-lv), refcount → 1
    mock_lc.status = AsyncMock(return_value="loaded")
    register_service(tc, name="omnivoice-lv", vram_gb=5.8, tier=2)
    claim_resp = tc.post("/claim/omnivoice-lv")
    assert claim_resp.status_code == 200

    # Register a Tier 3 service
    mock_lc.status = AsyncMock(return_value="unloaded")
    register_service(tc, name="vocal-isolator-lv", vram_gb=3.0, tier=3)

    # Claim Tier 3 — must yield with 503
    response = tc.post("/claim/vocal-isolator-lv")
    assert (
        response.status_code == 503
    ), f"Expected 503 (tier3_yield), got {response.status_code}: {response.text}"
    body = response.json()["detail"]
    assert body["reason"] == "tier3_yield", f"Expected reason=tier3_yield, got: {body}"
    assert "omnivoice-lv" in body["active_higher_priority"]
    assert "Retry-After" in response.headers

    # Verify refcount on vocal-isolator was NOT incremented (yield cleaned up)
    status = tc.get("/status")
    svcs = {s["service_name"]: s for s in status.json()["services"]}
    assert (
        svcs["vocal-isolator-lv"]["reference_count"] == 0
    ), "Tier 3 yield must not leave a stale refcount on vocal-isolator-lv"

    # Load must NOT have been called (no eviction/loading for a yielded claim)
    mock_lc.load.assert_not_called()


def test_tier3_claim_proceeds_when_no_active_higher_priority(client_with_mocks):
    """
    When no Tier 1 or Tier 2 service has refcount > 0, a Tier 3 claim
    must proceed normally (load the service, not yield with 503).
    """
    tc, mock_lc = client_with_mocks

    # Register a Tier 2 service but do NOT claim it (refcount stays 0)
    mock_lc.status = AsyncMock(return_value="loaded")
    register_service(tc, name="omnivoice-lv", vram_gb=5.8, tier=2)

    # Register and claim a Tier 3 service — no higher-priority service is active
    mock_lc.status = AsyncMock(return_value="unloaded")
    register_service(tc, name="vocal-isolator-lv", vram_gb=3.0, tier=3)

    response = tc.post("/claim/vocal-isolator-lv")
    assert (
        response.status_code == 200
    ), f"Tier 3 should succeed when no higher-priority service is active: {response.text}"
    assert response.json()["reference_count"] >= 1


def test_tier3_yield_retry_after_header_present(client_with_mocks):
    """Verify Retry-After header is included in the 503 yield response."""
    tc, mock_lc = client_with_mocks

    mock_lc.status = AsyncMock(return_value="loaded")
    register_service(tc, name="fluency-gate-roberta-lv", vram_gb=1.3, tier=1)
    tc.post("/claim/fluency-gate-roberta-lv")  # refcount → 1

    mock_lc.status = AsyncMock(return_value="unloaded")
    register_service(tc, name="vocal-isolator-lv", vram_gb=3.0, tier=3)

    response = tc.post("/claim/vocal-isolator-lv")
    assert response.status_code == 503
    assert "Retry-After" in response.headers
    # Retry-After should be a positive integer
    retry_after = int(response.headers["Retry-After"])
    assert retry_after > 0
