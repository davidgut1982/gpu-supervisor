# gpu-supervisor
#
# Lightweight CPU-only supervisor for GPU VRAM lifecycle management.
# Tracks VRAM usage per service, evicts idle services, and coordinates
# load/unload via each service's /lifecycle endpoints.
#
# Base image: python:3.11-slim (no GPU stack needed — pure CPU process)
#
# Internal port: 8202
# Host port:     8202 (set in docker-compose.yml)
#
# Build:
#   docker compose build
#
# Run (after build):
#   docker compose up -d

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# curl: required for Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install before copying app source — dependency layer changes rarely,
# so Docker cache reuse is maximised this way.
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY app/ .

EXPOSE 8202

# ── Runtime ───────────────────────────────────────────────────────────────────
# --workers 1: the in-memory registry uses asyncio.Lock for safety within a
# single process. Multiple workers would have separate registry instances and
# incorrect VRAM accounting. Use 1 worker + async concurrency instead.
CMD ["python", "main.py"]
