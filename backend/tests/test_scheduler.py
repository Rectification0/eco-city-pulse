"""Scheduled ingestion (task 2.9, design §6.3 "Scheduled — Cron").

No database and no clock-watching: the loop is driven by patching its one unit
of work, so these run in milliseconds rather than waiting out an interval.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import IngestionMode, Settings
from services import scheduler


def _settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        ingestion_startup_delay_seconds=0,
        ingestion_interval_minutes=1,
        **overrides,  # type: ignore[arg-type]
    )


@pytest.mark.anyio
async def test_the_scheduler_does_not_start_in_demo_mode() -> None:
    """DR-1: a default installation makes no outbound request at all."""
    assert scheduler.start(_settings(ingestion_mode=IngestionMode.DEMO)) is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mode", [IngestionMode.MANUAL, IngestionMode.UPLOAD, IngestionMode.DEMO]
)
async def test_only_scheduled_mode_starts_the_loop(mode: IngestionMode) -> None:
    """Manual and Upload are request-driven; neither implies a background poll."""
    assert scheduler.start(_settings(ingestion_mode=mode)) is None


@pytest.mark.anyio
async def test_scheduled_mode_runs_ingestion(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    done = asyncio.Event()

    def _fake_run(settings: Settings) -> list[object]:
        calls.append(1)
        done.set()
        return []

    monkeypatch.setattr(scheduler, "_run_once", _fake_run)

    task = scheduler.start(_settings(ingestion_mode=IngestionMode.SCHEDULED))
    try:
        await asyncio.wait_for(done.wait(), timeout=5)
    finally:
        await scheduler.stop(task)

    assert calls


@pytest.mark.anyio
async def test_a_failing_run_does_not_kill_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad run must not silently end scheduling for the process's life --
    that failure mode is invisible until someone notices stale data."""
    attempted = asyncio.Event()

    def _explode(settings: Settings) -> list[object]:
        attempted.set()
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(scheduler, "_run_once", _explode)

    task = scheduler.start(_settings(ingestion_mode=IngestionMode.SCHEDULED))
    try:
        await asyncio.wait_for(attempted.wait(), timeout=5)
        # Yield once so the loop can propagate the exception if it is going to.
        await asyncio.sleep(0)

        assert task is not None
        assert not task.done(), "a failing run ended the scheduler"
    finally:
        await scheduler.stop(task)


@pytest.mark.anyio
async def test_stopping_is_idempotent_and_safe() -> None:
    await scheduler.stop(None)


@pytest.mark.anyio
async def test_stop_awaits_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown must not race: a half-applied run on restart is worse than a
    slightly slower stop."""
    monkeypatch.setattr(scheduler, "_run_once", lambda settings: [])

    task = scheduler.start(_settings(ingestion_mode=IngestionMode.SCHEDULED))
    await scheduler.stop(task)

    assert task is not None and task.done()
