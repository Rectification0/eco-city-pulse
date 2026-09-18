"""Scheduled ingestion (task 2.9, design §6.3 "Scheduled — Cron").

An in-process asyncio task rather than a cron daemon or a scheduler library.
Three reasons:

* The deployment is a three-container compose stack (specs §12). Adding a
  fourth container for cron, or a second process inside this one, buys nothing
  the event loop cannot already do.
* It needs no extra dependency.
* An external cron can still drive ingestion by calling ``POST /data/ingest``,
  which is the same code path. Scheduling stays a deployment choice rather than
  something baked into the application.

The task only runs when ``INGESTION_MODE=scheduled``. In demo mode it never
starts, so a default installation makes no outbound requests at all (DR-1).

Database work happens in a worker thread: the ORM session is synchronous, and
running it on the event loop would block every request for the duration of an
ingestion run.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone

from core.config import IngestionMode, Settings
from db.session import session_scope
from services import ingestion_service

logger = logging.getLogger(__name__)


def _run_once(settings: Settings) -> list[ingestion_service.IngestionOutcome]:
    """One scheduled pass. Synchronous; called via ``asyncio.to_thread``."""
    with session_scope(settings) as session:
        return ingestion_service.run_live(
            session, settings, mode=IngestionMode.SCHEDULED
        )


async def _loop(settings: Settings) -> None:
    interval = max(1, settings.ingestion_interval_minutes) * 60

    if settings.ingestion_startup_delay_seconds > 0:
        await asyncio.sleep(settings.ingestion_startup_delay_seconds)

    while True:
        started = datetime.now(timezone.utc)
        try:
            outcomes = await asyncio.to_thread(_run_once, settings)
            logger.info(
                "Scheduled ingestion finished: %s",
                ", ".join(f"{o.source_name}={o.status.value}" for o in outcomes),
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the loop must outlive any one failure
            # ingest_source already absorbs upstream failures; anything reaching
            # here is a bug or a database problem. Logged and retried rather
            # than allowed to kill the scheduler for the process's lifetime.
            logger.exception("Scheduled ingestion raised; continuing.")

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        await asyncio.sleep(max(1.0, interval - elapsed))


def start(settings: Settings) -> asyncio.Task[None] | None:
    """Start the loop if the configuration asks for it. Returns the task."""
    if settings.ingestion_mode is not IngestionMode.SCHEDULED:
        logger.debug(
            "Scheduler not started: ingestion mode is %s.", settings.ingestion_mode.value
        )
        return None

    logger.info(
        "Starting scheduled ingestion every %s minute(s).",
        settings.ingestion_interval_minutes,
    )
    return asyncio.create_task(_loop(settings), name="scheduled-ingestion")


async def stop(task: asyncio.Task[None] | None) -> None:
    """Cancel the loop and wait for it, so shutdown is not racy."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


__all__ = ["start", "stop"]
