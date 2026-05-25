"""
gpu-supervisor — GPU VRAM lifecycle supervisor.

Endpoints:
  POST /register              — Service self-registers with VRAM footprint and tier
  POST /claim/{service_name}  — Caller declares intent to use a service; supervisor loads it
  POST /release/{service_name}— Caller signals it is done; refcount decremented
  GET  /status                — Full registry state for monitoring/debugging
  GET  /health                — Service health check

Design notes:
  - Pure CPU process; no GPU code. The supervisor tracks VRAM by accounting, not measurement.
  - Registry is in-memory only. Source of truth is the GPU itself.
  - Services self-register; no pre-configuration required at startup.
  - Tier 1 services: never auto-evicted.  Tier 2: idle-warm.  Tier 3: on-demand.
  - Reference count > 0 is absolute protection from eviction (even Tier 3).

Internal port: 8202 (configurable via PORT env var)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Optional

from config import settings
from eviction import NotEnoughVRAMError, evict_for_vram
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from gpu_metrics import DeviceMetrics, GpuMetricsCollector
from lifecycle_client import LifecycleClient, LifecycleError
from models import (
    ClaimResponse,
    DeviceReconciliation,
    HealthResponse,
    PerDeviceVRAM,
    Reconciliation,
    RegisterRequest,
    RegisterResponse,
    ServiceStatus,
    SupervisorStatus,
)
from registry import ServiceRegistry

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("gpu-supervisor")


# ── Application state ──────────────────────────────────────────────────────────

_registry: Optional[ServiceRegistry] = None
_client: Optional[LifecycleClient] = None
_gpu_metrics: Optional[GpuMetricsCollector] = None
_started_at: Optional[datetime] = None
_last_eviction: Optional[datetime] = None
_eviction_count: int = 0
_background_task_healthy: bool = True

# Serialise load/unload operations to prevent concurrent VRAM budget races.
# Per-device locks so loads onto distinct GPUs (e.g. P4 vs 3060) can proceed
# concurrently — a global lock would needlessly serialise non-contending devices.
_load_locks: dict[str, asyncio.Lock] = {}


def _get_load_lock(device_id: str) -> asyncio.Lock:
    """Return (creating on first use) the load lock for a physical GPU device.

    Why: Serialise load/eviction only among claims targeting the same device so
    that VRAM budget races are prevented without blocking loads onto other GPUs.
    What: Lazily creates and caches one asyncio.Lock per device_id. Safe without
    its own lock because the event loop is single-threaded (no await between the
    membership check and assignment).
    Test: Assert _get_load_lock("0") is _get_load_lock("0") and is not
    _get_load_lock("1").
    """
    if device_id not in _load_locks:
        _load_locks[device_id] = asyncio.Lock()
    return _load_locks[device_id]


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


# ── Soft reconciliation (nvidia-smi vs declared accounting) ──────────────────


def _declared_sum_mb_by_index(entries: list, nvidia_index: str) -> int:
    """Sum declared VRAM (MB) of loaded services mapped to one nvidia-smi index.

    Why: Reconciliation compares measured VRAM (keyed by nvidia-smi index "0"/"1")
    against the supervisor's declared footprints (keyed by device_id "0"/"1"/"default").
    Single-GPU services register as "default"; physically they live on GPU index 0,
    so they must count toward index "0"'s declared sum to compare meaningfully.
    What: Sums vram_gb_declared * 1024 for entries in state "loaded" whose device_id
    equals nvidia_index, plus "default" entries when nvidia_index == "0". Only "loaded"
    counts (matches the reconciliation constraint: in-flight/unknown states are excluded
    from the declared baseline).
    Test: One loaded "0" service at 2.8 GB and one "default" at 0.0 → index "0" sum is
    ~2867 MB; a loaded "1" service is excluded from index "0".
    """
    total_gb = 0.0
    for e in entries:
        if e.state != "loaded":
            continue
        if e.device_id == nvidia_index or (nvidia_index == "0" and e.device_id == "default"):
            total_gb += e.vram_gb_declared
    return round(total_gb * 1024)


def _build_reconciliation(
    metrics: dict[str, DeviceMetrics],
    entries: list,
) -> Reconciliation:
    """Compare measured per-device VRAM against declared sums into a Reconciliation.

    Why: Single place that turns an nvidia-smi snapshot + registry state into the
    /status reconciliation block and the warning list, so the endpoint and the
    background warning logger stay consistent.
    What: For each measured device computes delta = actual_used - declared_sum,
    flags "leak_suspected" when delta > leak_threshold_mb, and emits a human-readable
    warning per leaking device. Returns an empty Reconciliation when no sample exists.
    Test: metrics {"0": used 4000} + loaded "0" declaring 2.8 GB, threshold 500 →
    devices["0"].status == "leak_suspected" and warnings has one entry.
    """
    if not metrics:
        return Reconciliation()

    devices: dict[str, DeviceReconciliation] = {}
    warnings: list[str] = []
    sampled_at: Optional[datetime] = None

    for index, dm in metrics.items():
        sampled_at = dm.sampled_at
        declared = _declared_sum_mb_by_index(entries, index)
        delta = dm.used_vram_mb - declared
        leaking = delta > settings.leak_threshold_mb
        status_str = "leak_suspected" if leaking else "ok"
        devices[index] = DeviceReconciliation(
            actual_used_mb=dm.used_vram_mb,
            declared_sum_mb=declared,
            delta_mb=delta,
            status=status_str,
        )
        if leaking:
            warnings.append(
                f"GPU device {index} ({dm.name}): actual={dm.used_vram_mb}MB "
                f"declared_sum={declared}MB delta=+{delta}MB — possible leaked CUDA context"
            )

    return Reconciliation(sampled_at=sampled_at, devices=devices, warnings=warnings)


async def _log_reconciliation_warnings(metrics: dict[str, DeviceMetrics]) -> None:
    """on_sample callback: log a WARNING for each device whose VRAM exceeds declared.

    Why: Surfaces suspected leaks in the logs every poll (not just when /status is
    hit), giving an operator a passive alarm without an external monitor.
    What: Builds reconciliation against the current registry and logs each warning
    string at WARNING level. No-op when the registry isn't initialised or no leaks.
    Test: Patch registry to one loaded service under-declaring vs metrics, invoke this,
    assert a WARNING containing "possible leaked CUDA context" is emitted.
    """
    if _registry is None:
        return
    entries = await _registry.get_all()
    recon = _build_reconciliation(metrics, entries)
    for warning in recon.warnings:
        log.warning("reconciliation.leak_suspected  %s", warning)


# ── Background keep-alive expiry task ─────────────────────────────────────────


async def _expiry_task(registry: ServiceRegistry, client: LifecycleClient) -> None:
    """
    Background task: runs every `expiry_check_interval_seconds` seconds.

    For each loaded service with refcount==0 whose keep-alive has elapsed,
    call /lifecycle/unload and mark it as unloaded.  Tier 1 services have
    keep_alive_seconds=999999999 so they are never expired.
    """
    global _last_eviction, _eviction_count, _background_task_healthy

    while True:
        try:
            await asyncio.sleep(settings.expiry_check_interval_seconds)
            expired = await registry.idle_expired_services()
            for entry in expired:
                # Re-fetch the entry and re-check evictability before unloading.
                # A concurrent /claim may have incremented refcount from 0 → 1
                # between when idle_expired_services() ran and now.  Mirroring
                # the eviction module's pattern prevents unloading a claimed service.
                fresh = await registry.get(entry.service_name)
                if fresh is None:
                    log.info(
                        "expiry.skip  service=%s reason=deregistered_since_check",
                        entry.service_name,
                    )
                    continue
                if not fresh.is_evictable():
                    log.info(
                        "expiry.skip  service=%s reason=claimed_since_check refcount=%d",
                        fresh.service_name,
                        fresh.reference_count,
                    )
                    continue

                log.info(
                    "expiry.unload  service=%s idle_timeout=%ds",
                    fresh.service_name,
                    fresh.keep_alive_seconds,
                )
                await registry.set_state(fresh.service_name, "unloading")
                try:
                    await client.unload(fresh.service_name, fresh.base_url)
                    await registry.set_state(fresh.service_name, "unloaded")
                    _last_eviction = _utcnow()
                    _eviction_count += 1
                    log.info(
                        "expiry.unloaded  service=%s reason=idle_timeout",
                        fresh.service_name,
                    )
                except LifecycleError as exc:
                    log.warning(
                        "expiry.unload_failed  service=%s error=%s",
                        fresh.service_name,
                        exc,
                    )
                    await registry.set_state(fresh.service_name, "unknown")

        except asyncio.CancelledError:
            log.info("expiry task cancelled — shutting down")
            return
        except Exception as exc:
            # Log but don't crash the task; set health flag
            log.exception("expiry task unexpected error: %s", exc)
            _background_task_healthy = False


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _registry, _client, _gpu_metrics, _started_at, _background_task_healthy

    # Fail fast if NO VRAM budget is configured.  A wrong/zero VRAM budget would
    # silently mis-budget every eviction decision, so refuse to start instead of
    # operating with a misleading default.  Multi-GPU deployments may configure
    # only per-device budgets (GPU0_VRAM_GB / GPU1_VRAM_GB) without TOTAL_VRAM_GB,
    # so accept startup if ANY budget is > 0.
    if max(settings.total_vram_gb, settings.gpu0_vram_gb, settings.gpu1_vram_gb) <= 0:
        log.error(
            "No GPU VRAM budget configured (TOTAL_VRAM_GB=%.2f, GPU0_VRAM_GB=%.2f, "
            "GPU1_VRAM_GB=%.2f). Set at least one to your GPU's usable VRAM in GB "
            "(e.g. TOTAL_VRAM_GB=11.6 for an RTX 3060 12 GB).",
            settings.total_vram_gb,
            settings.gpu0_vram_gb,
            settings.gpu1_vram_gb,
        )
        sys.exit(1)

    log.info("gpu-supervisor starting up on port %d …", settings.port)
    log.info(
        "config  total_vram=%.1fGB gpu0_vram=%.1fGB gpu1_vram=%.1fGB "
        "tier2_keep_alive=%ds tier3_keep_alive=%ds auth=%s",
        settings.total_vram_gb,
        settings.gpu0_vram_gb,
        settings.gpu1_vram_gb,
        settings.tier2_keep_alive_seconds,
        settings.tier3_keep_alive_seconds,
        "enabled" if settings.api_key else "disabled",
    )

    _registry = ServiceRegistry()
    _client = LifecycleClient(
        load_timeout=float(settings.lifecycle_load_timeout_seconds),
        unload_timeout=float(settings.lifecycle_unload_timeout_seconds),
    )
    _started_at = _utcnow()
    _background_task_healthy = True

    task = asyncio.create_task(_expiry_task(_registry, _client))

    # Soft reconciliation: poll nvidia-smi to compare measured VRAM against the
    # declared accounting. Degrades to a no-op (empty reconciliation) if nvidia-smi
    # is unavailable; start() never raises. Stored on app.state for testability.
    _gpu_metrics = GpuMetricsCollector(
        poll_interval_seconds=settings.gpu_poll_seconds,
        on_sample=_log_reconciliation_warnings,
    )
    app.state.gpu_metrics = _gpu_metrics
    await _gpu_metrics.start()

    log.info("gpu-supervisor ready.")

    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if _gpu_metrics is not None:
            await _gpu_metrics.stop()
        log.info("gpu-supervisor shutdown complete.")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="gpu-supervisor",
    description=(
        "GPU VRAM lifecycle supervisor.\n\n"
        "Tracks VRAM usage per service, evicts idle/low-priority services to "
        "make room for incoming requests, and coordinates claim/release semantics "
        "for workflow-scoped GPU access.\n\n"
        "**Endpoints:**\n\n"
        "- `POST /register` — Service self-registration\n"
        "- `POST /claim/{service_name}` — Acquire a service (loads if needed)\n"
        "- `POST /release/{service_name}` — Release a service\n"
        "- `GET /status` — Full registry state\n"
        "- `GET /health` — Health check"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Optional API key authentication ────────────────────────────────────────────
#
# Why: Allow the supervisor to be exposed beyond a trusted Docker network without
# changing call sites. When API_KEY is unset/empty, auth is disabled and all
# requests pass through unchanged (preserves existing zero-config behaviour).
# What: Compares X-API-Key header against settings.api_key for every request
# except /health, /docs, and /openapi.json (so liveness probes and the OpenAPI
# explorer remain reachable without credentials).
# Test: Set API_KEY=secret, GET /status without header → 401; with correct
# header → 200. Unset API_KEY, GET /status without header → 200.
_AUTH_EXEMPT_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if settings.api_key and request.url.path not in _AUTH_EXEMPT_PATHS:
        if request.headers.get("X-API-Key", "") != settings.api_key:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ── Exception handler ──────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. See service logs for details."},
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _require_registry() -> ServiceRegistry:
    if _registry is None:
        raise HTTPException(status_code=503, detail="Supervisor not initialised.")
    return _registry


def _require_client() -> LifecycleClient:
    if _client is None:
        raise HTTPException(status_code=503, detail="Lifecycle client not initialised.")
    return _client


# ── POST /register ─────────────────────────────────────────────────────────────


@app.post(
    "/register",
    response_model=RegisterResponse,
    summary="Register a service with the supervisor",
    tags=["lifecycle"],
)
async def register(request: RegisterRequest) -> RegisterResponse:
    """
    Service self-registration endpoint.

    The service declares its VRAM footprint and priority tier.  The supervisor
    queries the service's /lifecycle/status to get its current state and stores
    the entry in the in-memory registry.

    Calling /register again for an existing service updates the entry (idempotent).
    If the service is unreachable at registration time, initial_state is recorded
    as "unknown" — the supervisor proceeds without error.
    """
    registry = _require_registry()
    client = _require_client()

    # Validate VRAM declaration does not exceed the budget of the target device.
    # Per-device budgeting: a service on GPU0 is checked against GPU0's budget so
    # a small GPU (e.g. P4 ~7.4 GB) rejects models that only fit the larger GPU.
    device_budget = settings.budget_for_device(request.device_id)
    if request.vram_gb_declared > device_budget:
        raise HTTPException(
            status_code=422,
            detail=(
                f"vram_gb_declared ({request.vram_gb_declared:.1f} GB) exceeds "
                f"device {request.device_id!r} budget ({device_budget:.1f} GB)"
            ),
        )

    # Tier 1 services are never auto-expired; reject keep_alive_seconds overrides
    # to prevent accidental misconfiguration (e.g. keep_alive_seconds=300 on a
    # permanent model service that must stay loaded).
    if request.priority_tier == 1 and request.keep_alive_seconds is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Tier 1 services cannot override keep_alive_seconds. "
                "Tier 1 services are never auto-expired. "
                "Remove the keep_alive_seconds field or set priority_tier to 2 or 3."
            ),
        )

    # Determine keep-alive: use per-service override if provided, else tier default
    keep_alive = (
        request.keep_alive_seconds
        if request.keep_alive_seconds is not None
        else settings.keep_alive_for_tier(request.priority_tier)
    )

    # Query current service state (best-effort; unreachable = "unknown")
    initial_state = await client.status(request.service_name, request.base_url) or "unknown"

    entry, is_new = await registry.register(
        service_name=request.service_name,
        base_url=request.base_url,
        vram_gb_declared=request.vram_gb_declared,
        priority_tier=request.priority_tier,
        keep_alive_seconds=keep_alive,
        initial_state=initial_state,
        device_id=request.device_id,
    )

    status_str = "registered" if is_new else "already_registered_updated"
    log.info(
        "register.ok  service=%s status=%s state=%s",
        request.service_name,
        status_str,
        entry.state,
    )

    return RegisterResponse(
        service_name=request.service_name,
        status=status_str,
        initial_state=entry.state,
    )


# ── POST /claim/{service_name} ─────────────────────────────────────────────────


@app.post(
    "/claim/{service_name}",
    response_model=ClaimResponse,
    summary="Claim a service (loads it if needed)",
    tags=["lifecycle"],
)
async def claim(service_name: str) -> ClaimResponse:
    """
    Declare intent to use a service.  The supervisor ensures it is loaded.

    Algorithm:
      1. If not registered: 404
      2. Increment reference count and update last_used
      3. If already loaded: return immediately
      4. If unloaded:
         a. Check available VRAM
         b. If insufficient: run eviction algorithm
         c. Call /lifecycle/load, wait for completion
         d. On success: return loaded
         e. On failure: decrement refcount, return 502

    The load_lock serialises concurrent claims to prevent VRAM budget races.
    """
    global _last_eviction, _eviction_count

    registry = _require_registry()
    client = _require_client()

    entry = await registry.get(service_name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Service not registered: {service_name!r}. " f"Call POST /register first.",
        )

    # Increment refcount before load attempt so that concurrent claims for the
    # same service see refcount > 0 and don't double-load.
    await registry.increment_refcount(service_name)

    t0 = time.monotonic()
    evicted: list[str] = []

    # Policy B (2026-05-06): Tier 3 yields to higher-priority active services.
    # When a Tier 3 service tries to claim while any Tier 1 or Tier 2 service
    # has refcount > 0, reject with 503 rather than evicting or loading.
    # This prevents thrashing when the user is actively using interactive
    # services (e.g. tts-service during synthesis, asr-service during transcription).
    if entry.priority_tier == 3:
        # Only yield to higher-priority services sharing the claimant's device.
        # A busy Tier 2 on a different GPU (e.g. a 3060 service) must not block a
        # Tier 3 claim targeting another GPU (e.g. a P4) — they don't contend.
        higher_priority_active = [
            e
            for e in (await registry.get_all())
            if e.priority_tier < 3
            and e.reference_count > 0
            and e.device_id == entry.device_id
            and e.service_name != service_name
        ]
        if higher_priority_active:
            # Decrement the refcount we just incremented — caller didn't get the claim
            await registry.decrement_refcount(service_name)
            names = [e.service_name for e in higher_priority_active]
            log.info(
                "claim deferred: %s (Tier 3) yielded to active higher-priority services: %s",
                service_name,
                names,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "reason": "tier3_yield",
                    "service": service_name,
                    "active_higher_priority": names,
                    "retry_after_seconds": settings.tier3_yield_retry_seconds,
                },
                headers={"Retry-After": str(settings.tier3_yield_retry_seconds)},
            )

    # Policy (2026-05-08): Tier 1 claims preempt idle Tier 2 services.
    # Before granting a Tier 1 claim (even via the fast path), evict any Tier 2
    # service that is loaded with refcount == 0.  This prevents CUDA OOM when
    # Whisper loads on top of a warm back-translator or sentence-embedder.
    #
    # If the Tier 2 service is in-use (refcount > 0), log a warning and proceed —
    # VRAM contention is preferable to interrupting an active claim.
    if entry.priority_tier == 1:
        all_services = await registry.get_all()
        # Scope preemption to the claimant's own device: a Tier 2 service on a
        # different physical GPU cannot OOM-conflict with this load, so it must
        # not be evicted (would free VRAM on the wrong device).
        idle_tier2 = [
            s
            for s in all_services
            if s.priority_tier == 2
            and s.state == "loaded"
            and s.reference_count == 0
            and s.device_id == entry.device_id
            and s.service_name != service_name
        ]
        busy_tier2 = [
            s
            for s in all_services
            if s.priority_tier == 2
            and s.state == "loaded"
            and s.reference_count > 0
            and s.device_id == entry.device_id
            and s.service_name != service_name
        ]
        for s in busy_tier2:
            log.warning(
                "claim.tier1_preempt_skipped  claimant=%s tier2_service=%s refcount=%d "
                "(in-use Tier 2 not evicted; VRAM contention possible)",
                service_name,
                s.service_name,
                s.reference_count,
            )
        if idle_tier2:
            async with _get_load_lock(entry.device_id):
                for s in idle_tier2:
                    # Re-check under the lock — a concurrent claim may have taken it
                    fresh = await registry.get(s.service_name)
                    if fresh is None or not fresh.is_evictable():
                        log.info(
                            "claim.tier1_preempt_skip  claimant=%s tier2_service=%s "
                            "reason=no_longer_evictable",
                            service_name,
                            s.service_name,
                        )
                        continue
                    log.info(
                        "claim.tier1_preempt  claimant=%s tier2_service=%s "
                        "vram=%.2fGB (Tier 2, idle)",
                        service_name,
                        s.service_name,
                        s.vram_gb_declared,
                    )
                    await registry.set_state(s.service_name, "unloading")
                    try:
                        await client.unload(s.service_name, s.base_url)
                        await registry.set_state(s.service_name, "unloaded")
                        evicted.append(s.service_name)
                        _last_eviction = _utcnow()
                        _eviction_count += 1
                    except LifecycleError as exc:
                        log.warning(
                            "claim.tier1_preempt_failed  claimant=%s tier2_service=%s error=%s",
                            service_name,
                            s.service_name,
                            exc,
                        )
                        await registry.set_state(s.service_name, "unknown")

    if entry.state == "loaded":
        waited = time.monotonic() - t0
        status_str = "evicted_to_load" if evicted else "loaded"
        return ClaimResponse(
            service_name=service_name,
            status=status_str,
            waited_seconds=round(waited, 3),
            reference_count=entry.reference_count,
            evicted=evicted,
        )

    # Service needs loading — serialise this section per device
    async with _get_load_lock(entry.device_id):
        # Re-read entry in case another coroutine loaded it while we waited
        entry = await registry.get(service_name)
        if entry is None:
            await registry.decrement_refcount(service_name)
            raise HTTPException(status_code=404, detail=f"Service {service_name!r} disappeared.")

        if entry.state == "loaded":
            waited = time.monotonic() - t0
            status_str = "evicted_to_load" if evicted else "loaded"
            return ClaimResponse(
                service_name=service_name,
                status=status_str,
                waited_seconds=round(waited, 3),
                reference_count=entry.reference_count,
                evicted=evicted,
            )

        # Check VRAM budget — scoped to the service's physical GPU device so that
        # accounting and eviction only consider VRAM on the device being loaded to.
        device_id = entry.device_id
        used_gb = await registry.used_vram_gb_for_device(device_id)
        available_gb = settings.budget_for_device(device_id) - used_gb
        vram_needed = entry.vram_gb_declared - available_gb

        if vram_needed > 0:
            log.info(
                "claim.eviction_needed  service=%s device=%s vram_needed=%.2fGB available=%.2fGB",
                service_name,
                device_id,
                entry.vram_gb_declared,
                available_gb,
            )
            try:
                evicted = await evict_for_vram(vram_needed, registry, client, device_id)
                _last_eviction = _utcnow()
                _eviction_count += len(evicted)
            except NotEnoughVRAMError as exc:
                await registry.decrement_refcount(service_name)
                log.warning(
                    "claim.insufficient_vram  service=%s detail=%s",
                    service_name,
                    exc,
                )
                raise HTTPException(
                    status_code=507,
                    detail=(
                        f"Insufficient VRAM to load {service_name!r}: "
                        f"needed {exc.needed:.2f} GB but only freed {exc.freed:.2f} GB. "
                        f"Exhausted {exc.candidates_exhausted} eviction candidate(s)."
                    ),
                ) from None

        # Load the service
        await registry.set_state(service_name, "loading")
        try:
            await client.load(service_name, entry.base_url)
            await registry.set_state(service_name, "loaded")
        except LifecycleError as exc:
            await registry.decrement_refcount(service_name)
            await registry.set_state(service_name, "unloaded")
            log.error(
                "claim.load_failed  service=%s error=%s",
                service_name,
                exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to load {service_name!r}: {exc}",
            ) from None

    waited = time.monotonic() - t0
    status_str = "evicted_to_load" if evicted else "loaded"
    log.info(
        "claim.ok  service=%s status=%s waited=%.2fs evicted=%s",
        service_name,
        status_str,
        waited,
        evicted,
    )

    # Re-read for accurate refcount after all operations
    entry = await registry.get(service_name)
    ref_count = entry.reference_count if entry else 1

    return ClaimResponse(
        service_name=service_name,
        status=status_str,
        waited_seconds=round(waited, 3),
        reference_count=ref_count,
        evicted=evicted,
    )


# ── POST /release/{service_name} ───────────────────────────────────────────────


@app.post(
    "/release/{service_name}",
    summary="Release a service (decrements reference count)",
    tags=["lifecycle"],
    status_code=200,
)
async def release(service_name: str) -> dict:
    """
    Signal that the caller is done using a service.

    Decrements the reference count (clamped to 0 — never goes negative).
    Does NOT immediately unload — the background keep-alive task handles that.

    Returns 200 with the updated reference count.
    Returns 404 if the service is not registered.
    """
    registry = _require_registry()

    entry = await registry.get(service_name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Service not registered: {service_name!r}",
        )

    new_count = await registry.decrement_refcount(service_name)
    log.info(
        "release.ok  service=%s new_refcount=%d",
        service_name,
        new_count,
    )

    return {"service_name": service_name, "reference_count": new_count}


# ── GET /status ────────────────────────────────────────────────────────────────


@app.get(
    "/status",
    response_model=SupervisorStatus,
    summary="Full registry state",
    tags=["monitoring"],
)
async def status() -> SupervisorStatus:
    """
    Return the full supervisor state: all registered services, VRAM accounting,
    eviction statistics, and uptime.
    """
    registry = _require_registry()

    entries = await registry.get_all()

    # "unknown" counts as used: a failed unload leaves VRAM most likely still
    # allocated. This mirrors registry.used_vram_gb_for_device() so the reported
    # usage matches the accounting eviction decisions are based on.
    _vram_holding = ("loaded", "loading", "unknown")
    used_gb = sum(e.vram_gb_declared for e in entries if e.state in _vram_holding)

    # Build per-device breakdown so multi-GPU deployments get correct per-GPU
    # headroom. The aggregate fields below sum across devices against the sum of
    # device budgets, avoiding the negative available_vram_gb that a single shared
    # budget produced when both GPUs were in use.
    device_ids = {e.device_id for e in entries}
    per_device: dict[str, PerDeviceVRAM] = {}
    for dev_id in sorted(device_ids):
        dev_budget = settings.budget_for_device(dev_id)
        dev_used = sum(
            e.vram_gb_declared
            for e in entries
            if e.device_id == dev_id and e.state in _vram_holding
        )
        per_device[dev_id] = PerDeviceVRAM(
            total_vram_gb=dev_budget,
            used_vram_gb=round(dev_used, 2),
            available_vram_gb=round(dev_budget - dev_used, 2),
        )

    # Aggregate budget is the sum of all distinct device budgets so it agrees with
    # the per-device totals. Falls back to total_vram_gb when nothing is registered.
    total_budget = (
        sum(d.total_vram_gb for d in per_device.values()) if per_device else settings.total_vram_gb
    )

    services = [
        ServiceStatus(
            service_name=e.service_name,
            base_url=e.base_url,
            vram_gb_declared=e.vram_gb_declared,
            priority_tier=e.priority_tier,
            device_id=e.device_id,
            state=e.state,
            reference_count=e.reference_count,
            last_used=e.last_used,
            keep_alive_seconds=e.keep_alive_seconds,
        )
        for e in entries
    ]

    # Soft reconciliation: latest nvidia-smi sample vs declared accounting.
    # Empty (sampled_at None) when nvidia-smi is unavailable or before first poll.
    metrics = _gpu_metrics.latest() if _gpu_metrics is not None else {}
    reconciliation = _build_reconciliation(metrics, entries)

    return SupervisorStatus(
        services=services,
        total_vram_gb=round(total_budget, 3),
        used_vram_gb=round(used_gb, 3),
        available_vram_gb=round(total_budget - used_gb, 3),
        per_device=per_device,
        reconciliation=reconciliation,
        started_at=_started_at or _utcnow(),
        last_eviction=_last_eviction,
        eviction_count=_eviction_count,
    )


# ── GET /health ────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    summary="Health check",
    tags=["health"],
)
async def health() -> JSONResponse:
    """
    Return service health.

    Returns 200 {"status": "ok", ...} if the supervisor is running normally.
    Returns 503 with details if the background task has crashed or the
    registry is not initialised.
    """
    if _registry is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "detail": "Registry not initialised",
                "registered_services": 0,
                "loaded_services": 0,
            },
        )

    if not _background_task_healthy:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "detail": "Background expiry task has encountered an error",
                "registered_services": 0,
                "loaded_services": 0,
            },
        )

    entries = await _registry.get_all()
    registered = len(entries)
    loaded = sum(1 for e in entries if e.state == "loaded")

    return JSONResponse(
        status_code=200,
        content=HealthResponse(
            status="ok",
            registered_services=registered,
            loaded_services=loaded,
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
