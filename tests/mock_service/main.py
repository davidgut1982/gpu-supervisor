"""
mock-service — minimal FastAPI service implementing the supervisor service contract.

Implements:
  POST /lifecycle/load    — simulates model load (delays configurable via env)
  POST /lifecycle/unload  — simulates model unload
  GET  /lifecycle/status  — returns current state
  GET  /health            — returns {"status": "ok"}

State is in-memory (single process, single worker).

Configuration via environment variables:
  SERVICE_NAME         — name reported in responses (default: "mock-service")
  VRAM_GB              — declared VRAM footprint (default: 1.0)
  INITIAL_STATE        — "loaded" | "unloaded" (default: "unloaded")
  LOAD_DELAY_SECONDS   — simulated load time (default: 0.1)
  UNLOAD_DELAY_SECONDS — simulated unload time (default: 0.05)
  LOAD_SHOULD_FAIL     — if "true", /lifecycle/load returns 500 (default: "false")
  UNLOAD_SHOULD_FAIL   — if "true", /lifecycle/unload returns 500 (default: "false")
  PORT                 — listening port (default: 8300)
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("mock-service")

# ── Configuration ──────────────────────────────────────────────────────────────

SERVICE_NAME = os.environ.get("SERVICE_NAME", "mock-service")
VRAM_GB = float(os.environ.get("VRAM_GB", "1.0"))
INITIAL_STATE = os.environ.get("INITIAL_STATE", "unloaded")
LOAD_DELAY = float(os.environ.get("LOAD_DELAY_SECONDS", "0.1"))
UNLOAD_DELAY = float(os.environ.get("UNLOAD_DELAY_SECONDS", "0.05"))
LOAD_SHOULD_FAIL = os.environ.get("LOAD_SHOULD_FAIL", "false").lower() == "true"
UNLOAD_SHOULD_FAIL = os.environ.get("UNLOAD_SHOULD_FAIL", "false").lower() == "true"
PORT = int(os.environ.get("PORT", "8300"))

# ── State ──────────────────────────────────────────────────────────────────────

_state: str = INITIAL_STATE
_load_count: int = 0
_unload_count: int = 0


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    log.info(
        "mock-service starting: name=%s vram=%.1fGB initial_state=%s port=%d",
        SERVICE_NAME,
        VRAM_GB,
        INITIAL_STATE,
        PORT,
    )
    yield
    log.info("mock-service shutdown.")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title=f"mock-service ({SERVICE_NAME})",
    description="Mock service implementing the GPU supervisor service contract.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health", summary="Health check")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service_name": SERVICE_NAME, "state": _state})


@app.get("/lifecycle/status", summary="Get model load state")
async def lifecycle_status() -> JSONResponse:
    return JSONResponse({
        "status": _state,
        "vram_gb_actual": VRAM_GB if _state == "loaded" else 0.0,
        "service_name": SERVICE_NAME,
    })


@app.post("/lifecycle/load", summary="Load model to GPU (simulated)")
async def lifecycle_load() -> JSONResponse:
    global _state, _load_count

    if LOAD_SHOULD_FAIL:
        log.warning("lifecycle.load CONFIGURED TO FAIL for service=%s", SERVICE_NAME)
        raise HTTPException(status_code=500, detail=f"{SERVICE_NAME}: load configured to fail")

    if _state == "loaded":
        log.info("lifecycle.load NOOP (already loaded) service=%s", SERVICE_NAME)
        return JSONResponse({
            "status": "loaded",
            "vram_gb_actual": VRAM_GB,
            "note": "idempotent — already loaded",
        })

    log.info("lifecycle.load START service=%s delay=%.2fs", SERVICE_NAME, LOAD_DELAY)
    await asyncio.sleep(LOAD_DELAY)
    _state = "loaded"
    _load_count += 1
    log.info("lifecycle.load DONE service=%s load_count=%d", SERVICE_NAME, _load_count)

    return JSONResponse({"status": "loaded", "vram_gb_actual": VRAM_GB})


@app.post("/lifecycle/unload", summary="Unload model from GPU (simulated)")
async def lifecycle_unload() -> JSONResponse:
    global _state, _unload_count

    if UNLOAD_SHOULD_FAIL:
        log.warning("lifecycle.unload CONFIGURED TO FAIL for service=%s", SERVICE_NAME)
        raise HTTPException(status_code=500, detail=f"{SERVICE_NAME}: unload configured to fail")

    if _state == "unloaded":
        log.info("lifecycle.unload NOOP (already unloaded) service=%s", SERVICE_NAME)
        return JSONResponse({"status": "unloaded", "note": "idempotent — already unloaded"})

    log.info("lifecycle.unload START service=%s delay=%.2fs", SERVICE_NAME, UNLOAD_DELAY)
    await asyncio.sleep(UNLOAD_DELAY)
    _state = "unloaded"
    _unload_count += 1
    log.info("lifecycle.unload DONE service=%s unload_count=%d", SERVICE_NAME, _unload_count)

    return JSONResponse({"status": "unloaded"})


@app.get("/debug/state", summary="Debug: current internal state")
async def debug_state() -> JSONResponse:
    return JSONResponse({
        "service_name": SERVICE_NAME,
        "state": _state,
        "vram_gb": VRAM_GB,
        "load_count": _load_count,
        "unload_count": _unload_count,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
