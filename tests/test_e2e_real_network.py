"""
Real-network end-to-end tests for gpu-supervisor's LifecycleClient.

These tests spin up a real uvicorn subprocess running mock_service/main.py and
exercise the actual LifecycleClient HTTP code paths — no monkeypatching.

This tests:
  - Real TCP connection, request, and response parsing
  - Timeout enforcement (configurable via client constructor)
  - Connection error handling (server killed mid-run)

How to run:
    # All tests (includes real_network):
    pytest tests/test_e2e_real_network.py -v

    # Skip in fast CI:
    pytest -m "not real_network"

    # Run only real_network:
    pytest -m real_network

Requirements:
    - uvicorn available in PATH (installed via requirements-test.txt)
    - Port 18099 available on localhost
    - mock_service/main.py in tests/mock_service/

Mark: real_network — excluded from default CI via -m "not real_network".
"""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent.parent / "app"
MOCK_SERVICE_DIR = Path(__file__).parent / "mock_service"

for _p in (str(APP_DIR), str(MOCK_SERVICE_DIR)):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(MOCK_SERVICE_DIR))
sys.path.insert(0, str(APP_DIR))

# ── Constants ──────────────────────────────────────────────────────────────────

STARTUP_TIMEOUT_S = 15.0   # Max seconds to wait for uvicorn to become ready
POLL_INTERVAL_S = 0.2


def _free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# ── Subprocess helpers ─────────────────────────────────────────────────────────


def _start_mock_server(
    port: int | None = None,
    env_overrides: dict | None = None,
) -> tuple[subprocess.Popen, int]:
    """
    Start mock_service as a subprocess uvicorn process.

    Allocates a free port if none provided. Returns (proc, port).
    The caller must terminate the proc when done.
    """
    if port is None:
        port = _free_port()

    env = os.environ.copy()
    env["SERVICE_NAME"] = "real-network-mock"
    env["VRAM_GB"] = "1.5"
    env["INITIAL_STATE"] = "unloaded"
    env["LOAD_DELAY_SECONDS"] = "0.2"
    env["UNLOAD_DELAY_SECONDS"] = "0.1"
    env["PORT"] = str(port)
    if env_overrides:
        env.update(env_overrides)

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=str(MOCK_SERVICE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc, port


def _wait_for_server(base_url: str, timeout: float = STARTUP_TIMEOUT_S) -> None:
    """
    Poll /health until the server responds 200 or timeout is reached.

    Raises RuntimeError if the server doesn't come up in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)
    raise RuntimeError(
        f"Mock server at {base_url} did not become ready within {timeout}s"
    )


def _stop_server(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Terminate a subprocess gracefully, then force-kill if needed."""
    if proc.poll() is not None:
        return  # Already exited
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_server():
    """
    Start a real uvicorn mock_service process on a free port. Yield base_url. Clean up after.
    """
    proc, port = _start_mock_server()
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(base_url)
        yield base_url
    finally:
        _stop_server(proc)


# ── Tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.real_network
@pytest.mark.asyncio
async def test_lifecycle_client_load_real_http(mock_server):
    """
    LifecycleClient.load() makes a real HTTP POST to /lifecycle/load and
    correctly parses the {"status": "loaded"} response.
    """
    from lifecycle_client import LifecycleClient

    client = LifecycleClient(load_timeout=10.0)
    result = await client.load("real-network-mock", mock_server)

    assert result["status"] == "loaded"
    assert "vram_gb_actual" in result


@pytest.mark.real_network
@pytest.mark.asyncio
async def test_lifecycle_client_unload_real_http(mock_server):
    """
    LifecycleClient.unload() makes a real HTTP POST to /lifecycle/unload and
    correctly parses the {"status": "unloaded"} response.

    First loads the service so unload has something to do.
    """
    from lifecycle_client import LifecycleClient

    client = LifecycleClient(load_timeout=10.0, unload_timeout=10.0)
    await client.load("real-network-mock", mock_server)
    result = await client.unload("real-network-mock", mock_server)

    assert result["status"] == "unloaded"


@pytest.mark.real_network
@pytest.mark.asyncio
async def test_lifecycle_client_status_real_http(mock_server):
    """
    LifecycleClient.status() returns "unloaded" for a freshly started service.
    """
    from lifecycle_client import LifecycleClient

    client = LifecycleClient()
    state = await client.status("real-network-mock", mock_server)

    assert state == "unloaded"


@pytest.mark.real_network
@pytest.mark.asyncio
async def test_lifecycle_client_load_idempotent_real_http(mock_server):
    """
    Calling load() twice succeeds both times (idempotent on the mock side).
    """
    from lifecycle_client import LifecycleClient

    client = LifecycleClient(load_timeout=10.0)
    r1 = await client.load("real-network-mock", mock_server)
    r2 = await client.load("real-network-mock", mock_server)

    assert r1["status"] == "loaded"
    assert r2["status"] == "loaded"


@pytest.mark.real_network
@pytest.mark.asyncio
async def test_lifecycle_client_load_unload_load_cycle_real_http(mock_server):
    """
    Full load → unload → load cycle works correctly over real HTTP.
    """
    from lifecycle_client import LifecycleClient

    client = LifecycleClient(load_timeout=10.0, unload_timeout=10.0)

    r1 = await client.load("real-network-mock", mock_server)
    assert r1["status"] == "loaded"

    mid_state = await client.status("real-network-mock", mock_server)
    assert mid_state == "loaded"

    r2 = await client.unload("real-network-mock", mock_server)
    assert r2["status"] == "unloaded"

    post_state = await client.status("real-network-mock", mock_server)
    assert post_state == "unloaded"

    r3 = await client.load("real-network-mock", mock_server)
    assert r3["status"] == "loaded"


@pytest.mark.real_network
@pytest.mark.asyncio
async def test_lifecycle_client_connection_error_raises_lifecycle_error():
    """
    LifecycleClient raises LifecycleError (not raw httpx exception) when the
    server is not reachable at all.

    Uses a port that was free at fixture time and is now closed — guaranteed
    to be refusing connections.
    """
    from lifecycle_client import LifecycleClient, LifecycleError

    # Allocate a port and immediately release it — nothing will be listening
    dead_port = _free_port()
    client = LifecycleClient(load_timeout=3.0)
    with pytest.raises(LifecycleError, match="network error"):
        await client.load("dead-service", f"http://127.0.0.1:{dead_port}")


@pytest.mark.real_network
@pytest.mark.asyncio
async def test_lifecycle_client_timeout_raises_lifecycle_error():
    """
    LifecycleClient raises LifecycleError when /lifecycle/load takes longer
    than the configured timeout.

    Uses a mock configured to delay 5s but client timeout set to 1s.
    """
    proc, port = _start_mock_server(env_overrides={"LOAD_DELAY_SECONDS": "5.0"})
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(base_url)

        from lifecycle_client import LifecycleClient, LifecycleError

        # Client timeout (1s) < server delay (5s) → should raise LifecycleError
        client = LifecycleClient(load_timeout=1.0)
        with pytest.raises(LifecycleError, match="timed out"):
            await client.load("slow-service", base_url)
    finally:
        _stop_server(proc)


@pytest.mark.real_network
@pytest.mark.asyncio
async def test_lifecycle_client_server_error_raises_lifecycle_error():
    """
    LifecycleClient raises LifecycleError when the server returns a 5xx error.
    """
    proc, port = _start_mock_server(env_overrides={"LOAD_SHOULD_FAIL": "true"})
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(base_url)

        from lifecycle_client import LifecycleClient, LifecycleError

        client = LifecycleClient(load_timeout=5.0)
        with pytest.raises(LifecycleError, match="HTTP 500"):
            await client.load("fail-service", base_url)
    finally:
        _stop_server(proc)


@pytest.mark.real_network
@pytest.mark.asyncio
async def test_lifecycle_client_status_returns_none_on_connection_error():
    """
    LifecycleClient.status() returns None (not raises) when the server is
    unreachable — this is the expected behavior for registration-time probing.
    """
    from lifecycle_client import LifecycleClient

    dead_port = _free_port()
    client = LifecycleClient()
    state = await client.status("dead-service", f"http://127.0.0.1:{dead_port}")

    assert state is None, (
        "status() should return None (not raise) when service is unreachable; "
        f"got {state!r}"
    )
