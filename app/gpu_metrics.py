"""
gpu-supervisor — soft reconciliation via periodic nvidia-smi polling.

Why: The supervisor tracks VRAM by accounting (sum of declared footprints), never
by measurement. A leaked CUDA context or a service that under/over-reports its
footprint silently drifts the accounting from reality. This module measures actual
per-device VRAM with nvidia-smi so /status and the logs can surface that drift
("soft" reconciliation — observe and warn, never act).
What: A background poller that shells out to nvidia-smi every poll_interval seconds,
parses the CSV, and caches per-device DeviceMetrics. latest() returns the most recent
sample (empty until the first poll, or permanently empty if nvidia-smi is unavailable).
Test: Monkeypatch _run_nvidia_smi to return canned CSV, call _poll_once, assert
latest()["0"].used_vram_mb matches the parsed value; with nvidia-smi missing assert
latest() == {} and polling disables itself without raising.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

log = logging.getLogger("gpu-supervisor.gpu_metrics")

# nvidia-smi query: index,name,total,used,free as plain integers (MB), no header.
_NVIDIA_SMI_ARGS = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
]

# Bound the subprocess so a hung driver can't wedge the poll loop forever.
_NVIDIA_SMI_TIMEOUT_SECONDS = 15


@dataclass
class DeviceMetrics:
    """Measured VRAM for one physical GPU at a point in time.

    Why: Gives reconciliation a measured ground-truth to compare the supervisor's
    declared accounting against, per device.
    What: Holds the nvidia-smi index, board name, and total/used/free VRAM in MB
    plus the sample timestamp.
    Test: Construct with used=2870, assert free == total - used holds for caller-
    supplied values (no invariant enforced here — values mirror nvidia-smi).
    """

    device_id: str  # nvidia-smi index as a string, e.g. "0", "1"
    name: str  # board name, e.g. "Tesla P4"
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int
    sampled_at: datetime


class GpuMetricsCollector:
    """Background poller caching the latest per-device VRAM measurement.

    Why: Centralises nvidia-smi access behind a non-blocking interface so the
    /status endpoint can read a cached snapshot without ever shelling out on the
    request path, and so a missing/broken nvidia-smi degrades gracefully instead
    of crashing the supervisor.
    What: start() launches an asyncio task that samples every poll_interval seconds
    via run_in_executor (subprocess off the event loop); latest() returns a copy of
    the most recent snapshot keyed by device_id.
    Test: Instantiate, monkeypatch _run_nvidia_smi to canned CSV, await _poll_once(),
    assert latest() is populated; set it to raise FileNotFoundError, assert _poll_once
    disables polling and latest() stays empty.
    """

    def __init__(
        self,
        poll_interval_seconds: int = 300,
        on_sample: Callable[[dict[str, DeviceMetrics]], Awaitable[None]] | None = None,
    ) -> None:
        self._poll_interval = poll_interval_seconds
        # Injected hook invoked after every successful sample. Kept as a callback
        # (dependency inversion) so the collector never imports the registry/config
        # — the supervisor owns the declared-sum comparison and warning logging.
        self._on_sample = on_sample
        self._latest: dict[str, DeviceMetrics] = {}
        self._task: asyncio.Task | None = None
        # None = not yet probed; True/False = nvidia-smi usable or not. Once False
        # the loop stops sampling so we don't spam failures every interval.
        self._available: bool | None = None

    async def start(self) -> None:
        """Probe nvidia-smi once, then launch the background polling task.

        Why: Surfacing "no nvidia-smi" at startup (rather than on the first /status
        read) makes a misconfigured deployment obvious in the boot logs, and lets us
        skip launching a loop that would only ever fail.
        What: Runs one synchronous poll; if it succeeds, schedules the periodic loop.
        If nvidia-smi is unavailable, logs a single warning and returns without a task.
        Test: With nvidia-smi present assert a task is created and latest() populated;
        with it absent assert no task and a single startup warning.
        """
        await self._poll_once()
        if self._available:
            self._task = asyncio.create_task(self._poll_loop())
            log.info(
                "gpu_metrics polling started (interval=%ds, devices=%d)",
                self._poll_interval,
                len(self._latest),
            )
        else:
            log.warning(
                "gpu_metrics disabled — nvidia-smi unavailable; reconciliation will be empty"
            )

    async def stop(self) -> None:
        """Cancel the polling task on shutdown.

        Why: Clean lifespan teardown — leave no orphaned task awaiting sleep.
        What: Cancels the task (if any) and awaits its CancelledError.
        Test: After start(), call stop(), assert the task is done.
        """
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def latest(self) -> dict[str, DeviceMetrics]:
        """Return the most recent per-device snapshot (empty before first poll).

        Why: /status reads this synchronously; it must be safe to call before any
        sample exists and must not expose the internal cache for mutation.
        What: Returns a shallow copy keyed by device_id; empty dict if no sample yet
        or if nvidia-smi is unavailable.
        Test: Call before start() → {}; after a successful poll → keys "0"/"1".
        """
        return dict(self._latest)

    # ── Internal ────────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Sample on a fixed interval until cancelled.

        Why: Keeps the cached snapshot fresh without blocking the event loop.
        What: Sleeps poll_interval, polls, repeats; logs and continues on transient
        errors so one bad sample doesn't kill the loop.
        Test: Patch _poll_once to record calls and sleep to no-op once, assert it loops.
        """
        while True:
            try:
                await asyncio.sleep(self._poll_interval)
                await self._poll_once()
            except asyncio.CancelledError:
                log.info("gpu_metrics poll loop cancelled — shutting down")
                return
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("gpu_metrics poll loop unexpected error: %s", exc)

    async def _poll_once(self) -> None:
        """Run nvidia-smi off the event loop, parse, and update the cache.

        Why: subprocess is blocking; running it in the default executor keeps the
        FastAPI event loop responsive. Any failure permanently disables polling so
        a missing driver doesn't generate noise every interval.
        What: Invokes _run_nvidia_smi via run_in_executor, parses CSV into the cache,
        and emits a WARNING-eligible leak summary handled by callers reading latest().
        Test: Patch _run_nvidia_smi to canned CSV, await this, assert _latest updated
        and _available True; patch it to raise, assert _available False and cache empty.
        """
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, self._run_nvidia_smi)
        except FileNotFoundError:
            self._available = False
            return
        except (subprocess.SubprocessError, OSError) as exc:
            self._available = False
            log.warning("gpu_metrics nvidia-smi invocation failed: %s", exc)
            return

        parsed = self._parse_csv(raw)
        if parsed:
            self._latest = parsed
            self._available = True
            if self._on_sample is not None:
                await self._on_sample(parsed)
        elif self._available is None:
            # First probe returned nothing parseable (e.g. no GPUs): disable quietly.
            self._available = False

    @staticmethod
    def _run_nvidia_smi() -> str:
        """Invoke nvidia-smi and return its stdout.

        Why: Isolated as a tiny seam so tests can monkeypatch the subprocess call
        without patching subprocess globally.
        What: Runs the CSV query with a timeout; raises on non-zero exit.
        Test: On a host with nvidia-smi, returns CSV lines; monkeypatched in unit tests.
        """
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, trusted binary
            _NVIDIA_SMI_ARGS,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
            check=True,
        )
        return result.stdout

    @staticmethod
    def _parse_csv(raw: str) -> dict[str, DeviceMetrics]:
        """Parse nvidia-smi CSV (noheader,nounits) into DeviceMetrics keyed by index.

        Why: Tolerate blank lines and malformed rows so one odd line from a driver
        quirk doesn't blank the whole snapshot.
        What: Splits each non-empty line on commas into index,name,total,used,free
        (MB integers); skips rows that don't parse and logs them at debug.
        Test: Feed "0, Tesla P4, 7611, 2870, 4741", assert result["0"].used_vram_mb
        == 2870 and name == "Tesla P4"; feed a garbage line, assert it is skipped.
        """
        sampled_at = datetime.now(tz=UTC)
        devices: dict[str, DeviceMetrics] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                log.debug("gpu_metrics skipping unparseable nvidia-smi row: %r", line)
                continue
            index, name, total, used, free = parts
            try:
                devices[index] = DeviceMetrics(
                    device_id=index,
                    name=name,
                    total_vram_mb=int(total),
                    used_vram_mb=int(used),
                    free_vram_mb=int(free),
                    sampled_at=sampled_at,
                )
            except ValueError:
                log.debug("gpu_metrics skipping non-numeric nvidia-smi row: %r", line)
                continue
        return devices
