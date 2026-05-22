"""
gpu-supervisor — shared pytest fixtures.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Add the app directory to sys.path so tests can import app modules directly.
APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR))


# ── Event loop fixture ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


# ── Registry fixture ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def registry():
    """Fresh ServiceRegistry for each test."""
    from registry import ServiceRegistry
    return ServiceRegistry()


# ── LifecycleClient mock fixture ──────────────────────────────────────────────

@pytest.fixture
def mock_client():
    """
    Mock LifecycleClient that returns success for load/unload/status.
    Individual tests can override return values or side_effects as needed.
    """
    client = MagicMock()
    client.load = AsyncMock(return_value={"status": "loaded", "vram_gb_actual": 1.0})
    client.unload = AsyncMock(return_value={"status": "unloaded"})
    client.status = AsyncMock(return_value="unloaded")
    return client


# ── FastAPI TestClient fixture ────────────────────────────────────────────────

@pytest.fixture
def test_app():
    """
    Create a FastAPI TestClient with a fresh application state.

    We patch the lifecycle client to avoid real HTTP calls during API tests.
    """
    from unittest.mock import patch, AsyncMock, MagicMock

    # Patch LifecycleClient before importing main to avoid real HTTP calls
    mock_client = MagicMock()
    mock_client.load = AsyncMock(return_value={"status": "loaded", "vram_gb_actual": 1.0})
    mock_client.unload = AsyncMock(return_value={"status": "unloaded"})
    mock_client.status = AsyncMock(return_value="unloaded")

    return mock_client
