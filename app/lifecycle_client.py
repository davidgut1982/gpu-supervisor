"""
gpu-supervisor — HTTP client for calling services' /lifecycle endpoints.

Each managed service must expose:
  POST /lifecycle/load    -> {"status": "loaded", "vram_gb_actual": float}
  POST /lifecycle/unload  -> {"status": "unloaded"}
  GET  /lifecycle/status  -> {"status": "loaded"|"unloaded", "vram_gb_actual": float}

All calls are expected to BLOCK until the operation is complete.  Timeouts
are enforced here on the supervisor side as a safety net.

The client is intentionally stateless — it holds no references to the
registry and performs no state mutations.  State updates are the caller's
responsibility.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger("gpu-supervisor")


class LifecycleError(Exception):
    """Raised when a /lifecycle call returns an unexpected status or times out."""


class LifecycleClient:
    """Thin httpx wrapper for service /lifecycle endpoints."""

    def __init__(
        self,
        load_timeout: float = 120.0,
        unload_timeout: float = 60.0,
        status_timeout: float = 10.0,
    ) -> None:
        self._load_timeout = load_timeout
        self._unload_timeout = unload_timeout
        self._status_timeout = status_timeout

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _post(
        self,
        url: str,
        timeout: float,
        service_name: str,
        operation: str,
    ) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise LifecycleError(
                f"{service_name} /lifecycle/{operation} timed out after {timeout}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LifecycleError(
                f"{service_name} /lifecycle/{operation} returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise LifecycleError(
                f"{service_name} /lifecycle/{operation} network error: {exc}"
            ) from exc

    async def _get(
        self,
        url: str,
        timeout: float,
        service_name: str,
        operation: str,
    ) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise LifecycleError(
                f"{service_name} /lifecycle/{operation} timed out after {timeout}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LifecycleError(
                f"{service_name} /lifecycle/{operation} returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise LifecycleError(
                f"{service_name} /lifecycle/{operation} network error: {exc}"
            ) from exc

    # ── Public API ────────────────────────────────────────────────────────────

    async def load(self, service_name: str, base_url: str) -> dict:
        """
        POST {base_url}/lifecycle/load

        Blocks until the service reports it is loaded and ready.
        Returns the JSON response body.
        Raises LifecycleError on timeout or non-2xx response.
        """
        url = f"{base_url}/lifecycle/load"
        log.info("lifecycle.load  service=%s url=%s", service_name, url)
        result = await self._post(url, self._load_timeout, service_name, "load")
        log.info(
            "lifecycle.load  service=%s result=%s",
            service_name,
            result,
        )
        return result

    async def unload(self, service_name: str, base_url: str) -> dict:
        """
        POST {base_url}/lifecycle/unload

        Blocks until the service has freed VRAM.
        Returns the JSON response body.
        Raises LifecycleError on timeout or non-2xx response.
        """
        url = f"{base_url}/lifecycle/unload"
        log.info("lifecycle.unload  service=%s url=%s", service_name, url)
        result = await self._post(url, self._unload_timeout, service_name, "unload")
        log.info(
            "lifecycle.unload  service=%s result=%s",
            service_name,
            result,
        )
        return result

    async def status(
        self, service_name: str, base_url: str
    ) -> Optional[str]:
        """
        GET {base_url}/lifecycle/status

        Returns the service's reported state ("loaded" or "unloaded"),
        or None if the service is unreachable.

        Does NOT raise on network errors — a service that is down at
        registration time is not a fatal error; its state is recorded as
        "unknown" and the supervisor proceeds.
        """
        url = f"{base_url}/lifecycle/status"
        try:
            result = await self._get(url, self._status_timeout, service_name, "status")
            state = result.get("status", "unknown")
            log.info(
                "lifecycle.status  service=%s state=%s",
                service_name,
                state,
            )
            return state
        except LifecycleError as exc:
            log.warning(
                "lifecycle.status  service=%s unreachable: %s",
                service_name,
                exc,
            )
            return None
