# gpu-supervisor

GPU VRAM lifecycle supervisor — a priority-based load/evict coordinator for
running multiple GPU models on constrained hardware.

If you have ever tried to run two ML services on one consumer GPU and watched
`CUDA out of memory` ruin your day, this is for you.

---

## The problem

A single GPU has finite VRAM. The combined working set of every model you'd
like to run usually exceeds it:

- A TTS model, an ASR model, and a translation model together need 10+ GB,
  but your card has 8 or 12.
- You don't need all of them loaded at once — but you also don't want to
  pay model-load latency on every request.
- Concurrent requests racing for VRAM can blow each other up mid-inference.

`gpu-supervisor` is a small CPU-only HTTP service that solves this:

- Each managed service **declares** its VRAM footprint and a priority tier.
- Callers wrap inference calls with `claim` / `release` to acquire the service.
- The supervisor **loads** the service if not already loaded, and **evicts**
  lower-priority idle services to free VRAM when needed.
- A reference count protects in-use services from eviction.
- A background task unloads services that have been idle past their keep-alive.

The supervisor itself never touches the GPU. It tracks VRAM by accounting,
not measurement, and relies on each managed service to honestly load/unload
on demand.

---

## Architecture at a glance

```
            ┌─────────────────┐
            │   your caller   │
            │ (web app, etc.) │
            └────────┬────────┘
                     │ 1. POST /claim/tts
                     ▼
            ┌─────────────────┐         3. POST /lifecycle/load
            │  gpu-supervisor │────────────────────────────────┐
            │   (this repo)   │                                ▼
            └────────┬────────┘                       ┌─────────────────┐
                     │ 2. returns 200                 │   tts-service   │
                     ▼                                │ (GPU container) │
            ┌─────────────────┐  4. POST /infer       └─────────────────┘
            │   your caller   │──────────────────────────────► (same)
            └─────────────────┘
                     │ 5. POST /release/tts
                     ▼
            ┌─────────────────┐
            │  gpu-supervisor │ (decrements refcount; may evict later)
            └─────────────────┘
```

The supervisor is **not in the inference path** — once a service is claimed,
callers talk to it directly. The supervisor only mediates load/unload.

---

## The lifecycle contract

Any service managed by the supervisor must expose three endpoints:

| Method | Path                | Purpose                                   |
|--------|---------------------|-------------------------------------------|
| POST   | `/lifecycle/load`   | Load weights to GPU; block until ready    |
| POST   | `/lifecycle/unload` | Free VRAM; block until VRAM is released   |
| GET    | `/lifecycle/status` | Report current state without modifying it |

Both `load` and `unload` must be **idempotent**: calling `load` on an
already-loaded service is a success no-op.

See [Example service skeleton](#example-service-skeleton) below.

---

## Quickstart

### 1. Run the supervisor

```bash
git clone https://github.com/davidgut1982/gpu-supervisor.git
cd gpu-supervisor

# Required: tell the supervisor how much VRAM your GPU has.
export TOTAL_VRAM_GB=11.6   # e.g. RTX 3060 12 GB ≈ 11.6 usable

docker compose up -d
```

Verify it's up:

```bash
curl http://localhost:8202/health
# {"status":"ok","registered_services":0,"loaded_services":0,...}
```

### 2. Register a service

A managed service registers itself at startup:

```bash
curl -X POST http://localhost:8202/register \
  -H 'Content-Type: application/json' \
  -d '{
    "service_name": "my-tts",
    "base_url": "http://my-tts:8000",
    "vram_gb_declared": 5.8,
    "priority_tier": 2
  }'
```

### 3. Claim and release around inference calls

```bash
# Claim — loads the service if not already loaded
curl -X POST http://localhost:8202/claim/my-tts

# … call the service directly for inference …
curl -X POST http://my-tts:8000/synthesize -d '...'

# Release — decrements reference count
curl -X POST http://localhost:8202/release/my-tts
```

---

## API reference

### POST /register

Self-registration. Idempotent — calling again updates the entry.

```json
{
  "service_name": "my-tts",
  "base_url": "http://my-tts:8000",
  "vram_gb_declared": 5.8,
  "priority_tier": 2,
  "keep_alive_seconds": null
}
```

Response:

```json
{
  "service_name": "my-tts",
  "status": "registered",
  "initial_state": "unloaded"
}
```

### POST /claim/{service_name}

Acquire a service. Loads it if not loaded, evicting lower-priority idle
services if VRAM is tight.

Response:

```json
{
  "service_name": "my-tts",
  "status": "loaded",
  "waited_seconds": 0.12,
  "reference_count": 1,
  "evicted": []
}
```

Status codes:

| Code | Meaning                                                       |
|------|---------------------------------------------------------------|
| 200  | Loaded and ready                                              |
| 404  | Service not registered                                        |
| 502  | The service's `/lifecycle/load` call failed                   |
| 503  | Tier 3 yielded to active higher-priority service              |
| 507  | Insufficient VRAM even after exhausting eviction candidates   |

### POST /release/{service_name}

Decrement reference count (clamped to 0). Does **not** unload immediately —
the background task handles that after `keep_alive_seconds` of idleness.

Response:

```json
{"service_name": "my-tts", "reference_count": 0}
```

### GET /status

Full registry state, VRAM accounting, and eviction stats.

### GET /health

`200` when healthy, `503` if the background task has crashed.

---

## Priority tiers

| Tier | Name        | Default keep-alive | Evicted when…                                        |
|------|-------------|--------------------|------------------------------------------------------|
| 1    | Always warm | ∞                  | Never                                                 |
| 2    | Idle warm   | 30 min             | Tier 1 claim or VRAM pressure (and idle)             |
| 3    | On-demand   | 5 min              | Any VRAM pressure (and idle); yields to active T1/T2 |

Two absolute rules:

1. **`reference_count > 0` blocks eviction.** A service mid-inference is never
   preempted, regardless of tier.
2. **Tier 1 is never auto-evicted.** Even with `keep_alive_seconds` set.

Eviction order when VRAM is short:

1. Tier 3 first, oldest `last_used` within tier (LRU)
2. Then Tier 2, same ordering
3. Stops as soon as enough VRAM is freed

---

## Configuration

All settings are environment variables. Only `TOTAL_VRAM_GB` is required.

| Variable                            | Required | Default     | Description                                              |
|-------------------------------------|----------|-------------|----------------------------------------------------------|
| `TOTAL_VRAM_GB`                     | **yes**  | —           | Usable VRAM in GB. Service exits if unset or ≤ 0         |
| `API_KEY`                           | no       | `""`        | If set, require `X-API-Key` header on all endpoints      |
| `PORT`                              | no       | `8202`      | HTTP port                                                |
| `LOG_LEVEL`                         | no       | `INFO`      | Python logging level                                     |
| `TIER1_KEEP_ALIVE_SECONDS`          | no       | `99999999`  | Effectively infinite                                     |
| `TIER2_KEEP_ALIVE_SECONDS`          | no       | `1800`      | 30 min                                                   |
| `TIER3_KEEP_ALIVE_SECONDS`          | no       | `300`       | 5 min                                                    |
| `LIFECYCLE_LOAD_TIMEOUT_SECONDS`    | no       | `120`       | Max time for `/lifecycle/load`                           |
| `LIFECYCLE_UNLOAD_TIMEOUT_SECONDS`  | no       | `60`        | Max time for `/lifecycle/unload`                         |
| `EXPIRY_CHECK_INTERVAL_SECONDS`     | no       | `30`        | Background expiry task interval                          |
| `TIER3_YIELD_RETRY_SECONDS`         | no       | `60`        | `Retry-After` value when a Tier 3 claim yields           |

### Optional API key auth

Set `API_KEY` in your environment (or in `docker-compose.yml`). When set, every
request must include `X-API-Key: <value>`, except `/health`, `/docs`, and
`/openapi.json` which remain open (so liveness probes and the OpenAPI explorer
still work).

```bash
export API_KEY="$(openssl rand -hex 32)"
docker compose up -d

curl http://localhost:8202/status                        # 401 Unauthorized
curl -H "X-API-Key: $API_KEY" http://localhost:8202/status  # 200
```

Leave `API_KEY` unset for trusted-network deployments.

---

## Example service skeleton

Here is a minimal FastAPI service that implements the lifecycle contract
and self-registers with the supervisor on startup.

```python
import gc
import os
from contextlib import asynccontextmanager

import httpx
import torch
from fastapi import FastAPI

SUPERVISOR_URL = os.environ.get("GPU_SUPERVISOR_URL", "http://gpu-supervisor:8202")
SERVICE_NAME = "my-tts"
BASE_URL = f"http://{SERVICE_NAME}:8000"
VRAM_GB = 5.8
PRIORITY_TIER = 2

_model = None


def _load_model_to_gpu():
    # Replace with your real model load
    return torch.nn.Linear(1024, 1024).cuda()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register with the supervisor (non-fatal if it's not up yet)
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            await c.post(
                f"{SUPERVISOR_URL}/register",
                json={
                    "service_name": SERVICE_NAME,
                    "base_url": BASE_URL,
                    "vram_gb_declared": VRAM_GB,
                    "priority_tier": PRIORITY_TIER,
                },
            )
    except Exception:
        pass  # supervisor will catch up via /lifecycle/status on next /register
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/lifecycle/load")
async def lifecycle_load() -> dict:
    global _model
    if _model is None:
        _model = _load_model_to_gpu()
    return {"status": "loaded", "vram_gb_actual": VRAM_GB}


@app.post("/lifecycle/unload")
async def lifecycle_unload() -> dict:
    global _model
    if _model is not None:
        del _model
        _model = None
        gc.collect()
        torch.cuda.empty_cache()
    return {"status": "unloaded"}


@app.get("/lifecycle/status")
async def lifecycle_status() -> dict:
    is_loaded = _model is not None
    return {
        "status": "loaded" if is_loaded else "unloaded",
        "vram_gb_actual": VRAM_GB if is_loaded else 0.0,
    }
```

### Caller-side wrapper

```python
SUPERVISOR = "http://gpu-supervisor:8202"

async def call_tts(text: str) -> bytes:
    async with httpx.AsyncClient() as c:
        claim = await c.post(f"{SUPERVISOR}/claim/my-tts", timeout=150.0)
        claim.raise_for_status()
        try:
            resp = await c.post(
                "http://my-tts:8000/synthesize",
                json={"text": text},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.content
        finally:
            # Always release — even on error
            await c.post(f"{SUPERVISOR}/release/my-tts")
```

---

## Startup ordering

The registry is **in-memory only**. There is no on-disk persistence, by design.

This has one operational consequence: **if the supervisor restarts, every
service must re-register.** Two patterns handle this:

1. **Self-registration on every startup** (recommended). Each service calls
   `/register` from its own startup hook. If the supervisor restarts later,
   services do not automatically re-register — see pattern 2.

2. **Periodic re-registration**. Have each service re-`/register` every few
   minutes (idempotent). This makes recovery from a supervisor restart
   automatic, at the cost of a small amount of background traffic.

For a small fixed set of services, either is fine. For larger deployments,
prefer pattern 2.

---

## Implementation notes

- **`torch.cuda.empty_cache()` alone is not sufficient.** You must also drop
  every reference to the model object and call `gc.collect()` before
  `empty_cache()`, or VRAM will not actually be released.
- **`/lifecycle/load` must block.** The supervisor only marks the service as
  loaded after the HTTP call returns 200.
- **Failed unloads are recorded as `state="unknown"`** and the VRAM is still
  counted as used. This prevents over-commit after a half-failed unload.
- **A single `asyncio.Lock` serialises load/unload transitions.** This is
  intentional — it eliminates the VRAM double-booking race at the cost of
  serialising concurrent claims that need to load.

---

## Running tests

```bash
pip install -r app/requirements.txt
pip install -r tests/requirements-test.txt
python -m pytest
```

---

## License

MIT. See [LICENSE](LICENSE) for details.
