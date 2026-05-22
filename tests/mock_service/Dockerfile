# mock-service — tiny FastAPI service for gpu-supervisor end-to-end tests.
#
# Implements the supervisor service contract:
#   POST /lifecycle/load
#   POST /lifecycle/unload
#   GET  /lifecycle/status
#   GET  /health
#
# Configuration via environment variables (see main.py).

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .

EXPOSE 8300

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8300", "--workers", "1"]
