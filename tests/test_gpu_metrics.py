"""
Unit tests for soft reconciliation: GpuMetricsCollector + reconciliation builders.

These exercise nvidia-smi parsing, graceful degradation when nvidia-smi is absent,
and the measured-vs-declared comparison that powers /status reconciliation.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Ensure app directory is on path (mirrors other test modules).
APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from gpu_metrics import DeviceMetrics, GpuMetricsCollector  # noqa: E402

# Two-GPU sample mimicking the latvian host (P4 on index 0, RTX 3060 on index 1).
_SAMPLE_CSV = "0, Tesla P4, 7611, 2870, 4741\n1, NVIDIA GeForce RTX 3060, 12288, 9800, 2488\n"


# ── CSV parsing ──────────────────────────────────────────────────────────────


def test_parse_csv_two_devices():
    parsed = GpuMetricsCollector._parse_csv(_SAMPLE_CSV)
    assert set(parsed) == {"0", "1"}
    assert parsed["0"].name == "Tesla P4"
    assert parsed["0"].total_vram_mb == 7611
    assert parsed["0"].used_vram_mb == 2870
    assert parsed["0"].free_vram_mb == 4741
    assert parsed["1"].used_vram_mb == 9800


def test_parse_csv_skips_blank_and_malformed_rows():
    raw = "\n0, Tesla P4, 7611, 2870, 4741\ngarbage line\n1, X, not_a_number, 1, 1\n"
    parsed = GpuMetricsCollector._parse_csv(raw)
    assert set(parsed) == {"0"}  # malformed and non-numeric rows skipped


def test_parse_csv_empty_returns_empty():
    assert GpuMetricsCollector._parse_csv("") == {}


# ── latest() safety ──────────────────────────────────────────────────────────


def test_latest_empty_before_poll():
    collector = GpuMetricsCollector(poll_interval_seconds=300)
    assert collector.latest() == {}


def test_latest_returns_copy():
    collector = GpuMetricsCollector()
    collector._latest = GpuMetricsCollector._parse_csv(_SAMPLE_CSV)
    snapshot = collector.latest()
    snapshot.clear()
    assert collector.latest() != {}  # mutating the copy must not affect the cache


# ── Polling / graceful degradation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_once_populates_cache(monkeypatch):
    collector = GpuMetricsCollector()
    monkeypatch.setattr(collector, "_run_nvidia_smi", lambda: _SAMPLE_CSV)
    await collector._poll_once()
    assert collector._available is True
    assert collector.latest()["1"].used_vram_mb == 9800


@pytest.mark.asyncio
async def test_poll_once_disables_when_nvidia_smi_missing(monkeypatch):
    collector = GpuMetricsCollector()

    def _raise():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(collector, "_run_nvidia_smi", _raise)
    await collector._poll_once()
    assert collector._available is False
    assert collector.latest() == {}


@pytest.mark.asyncio
async def test_start_no_task_when_unavailable(monkeypatch):
    collector = GpuMetricsCollector()

    def _raise():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(collector, "_run_nvidia_smi", _raise)
    await collector.start()
    assert collector._task is None  # no background loop launched
    await collector.stop()  # safe no-op


@pytest.mark.asyncio
async def test_on_sample_invoked_after_successful_poll(monkeypatch):
    seen: list[dict] = []

    async def _hook(metrics):
        seen.append(metrics)

    collector = GpuMetricsCollector(on_sample=_hook)
    monkeypatch.setattr(collector, "_run_nvidia_smi", lambda: _SAMPLE_CSV)
    await collector._poll_once()
    assert len(seen) == 1
    assert set(seen[0]) == {"0", "1"}


# ── Reconciliation builder ──────────────────────────────────────────────────


def _dm(index: str, used: int, name: str = "GPU") -> DeviceMetrics:
    return DeviceMetrics(
        device_id=index,
        name=name,
        total_vram_mb=12288,
        used_vram_mb=used,
        free_vram_mb=12288 - used,
        sampled_at=datetime(2026, 5, 25, 10, 0, tzinfo=UTC),
    )


class _Entry:
    """Minimal stand-in for ServiceEntry (only fields reconciliation reads)."""

    def __init__(self, device_id: str, vram_gb: float, state: str = "loaded"):
        self.device_id = device_id
        self.vram_gb_declared = vram_gb
        self.state = state


def test_declared_sum_default_maps_to_index_zero():
    import main

    entries = [_Entry("default", 2.8)]
    # 2.8 GB * 1024 = 2867 MB; "default" counts toward physical index "0".
    assert main._declared_sum_mb_by_index(entries, "0") == round(2.8 * 1024)
    assert main._declared_sum_mb_by_index(entries, "1") == 0


def test_declared_sum_only_counts_loaded():
    import main

    entries = [_Entry("0", 2.8, state="loading"), _Entry("0", 1.0, state="loaded")]
    assert main._declared_sum_mb_by_index(entries, "0") == round(1.0 * 1024)


def test_reconciliation_ok_within_threshold():
    import main

    metrics = {"0": _dm("0", used=2870)}
    entries = [_Entry("0", 2.8)]  # 2867 MB declared, delta = 3
    recon = main._build_reconciliation(metrics, entries)
    assert recon.devices["0"].delta_mb == 2870 - round(2.8 * 1024)
    assert recon.devices["0"].status == "ok"
    assert recon.warnings == []


def test_reconciliation_negative_delta_is_ok():
    import main

    metrics = {"1": _dm("1", used=9800)}
    entries = [_Entry("1", 10.45)]  # ~10700 MB declared, delta negative
    recon = main._build_reconciliation(metrics, entries)
    assert recon.devices["1"].delta_mb < 0
    assert recon.devices["1"].status == "ok"


def test_reconciliation_flags_leak(monkeypatch):
    import main

    metrics = {"0": _dm("0", used=4000, name="Tesla P4")}
    entries = [_Entry("0", 2.8)]  # declared 2867, delta ~1133 > 500
    recon = main._build_reconciliation(metrics, entries)
    assert recon.devices["0"].status == "leak_suspected"
    assert len(recon.warnings) == 1
    assert "possible leaked CUDA context" in recon.warnings[0]
    assert "Tesla P4" in recon.warnings[0]


def test_reconciliation_empty_when_no_metrics():
    import main

    recon = main._build_reconciliation({}, [_Entry("0", 2.8)])
    assert recon.sampled_at is None
    assert recon.devices == {}
    assert recon.warnings == []
