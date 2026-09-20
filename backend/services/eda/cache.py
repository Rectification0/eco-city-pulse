"""Cache in front of the profile computation (task 4.6, design §11).

design §11: "keys include the dataset version so results never go stale
silently". The emphasis is the point -- a cache that can serve a profile of
data that has since changed is worse than no cache, because the number looks
authoritative and is wrong.

**The dataset version** is a fingerprint taken from the same slice the profile
will be computed over: how many rows, the highest id, the newest timestamp, and
how many rows are flagged. It costs one aggregate query and it moves whenever
the data does -- an ingestion run raises the count and the max id, a quality
run changes the flag count. A cached entry is therefore unreachable once its
data has changed, rather than merely unlikely to be served.

**Scope.** An in-process LRU with a TTL, not Redis: the deployment is three
containers (specs §12) and adding a fourth for a cache in front of a
second-scale computation would be poor value. The consequence is honest and
worth stating -- with several uvicorn workers each holds its own cache, so the
hit rate falls but correctness does not, because the key still pins the data.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Observation

DEFAULT_TTL_SECONDS = 600
DEFAULT_MAX_ENTRIES = 32


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    """A fingerprint of the slice a result was computed from.

    Returned to the caller as well as used as a key: a client holding two
    profiles can tell whether they describe the same data, which a timestamp
    alone would not settle.
    """

    rows: int
    max_id: int | None
    latest_timestamp: datetime | None
    flagged_rows: int

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "rows": self.rows,
                "max_id": self.max_id,
                "latest": self.latest_timestamp.isoformat()
                if self.latest_timestamp
                else None,
                "flagged": self.flagged_rows,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "rows": self.rows,
            "max_id": self.max_id,
            "latest_timestamp": self.latest_timestamp.isoformat()
            if self.latest_timestamp
            else None,
            "flagged_rows": self.flagged_rows,
        }


def dataset_version(
    session: Session,
    *,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> DatasetVersion:
    """Fingerprint the slice without loading it.

    Four aggregates in one query. ``max_id`` catches inserts, ``rows`` catches
    inserts and deletes together with it, ``latest_timestamp`` catches a
    backfill that lands earlier than the newest row, and ``flagged_rows``
    catches a quality run that changed nothing but the judgement.
    """
    statement = select(
        func.count(Observation.id),
        func.max(Observation.id),
        func.max(Observation.timestamp),
        func.count(Observation.id).filter(Observation.is_anomaly.is_(True)),
    )

    # `is not None` for the same reason as in ``load_observations``: the
    # fingerprint has to cover exactly the slice that was loaded. If it covered
    # every source while the frame held one scope's worth, two scopes would
    # share a cache key and each would be served the other's answer.
    if source_ids is not None:
        statement = statement.where(Observation.source_id.in_(list(source_ids)))
    if start is not None:
        statement = statement.where(Observation.timestamp >= start)
    if end is not None:
        statement = statement.where(Observation.timestamp <= end)

    rows, max_id, latest, flagged = session.execute(statement).one()

    return DatasetVersion(
        rows=int(rows or 0),
        max_id=int(max_id) if max_id is not None else None,
        latest_timestamp=latest,
        flagged_rows=int(flagged or 0),
    )


@dataclass(slots=True)
class _Entry:
    value: Any
    stored_at: float


class ProfileCache:
    """Thread-safe LRU with a TTL. Small on purpose."""

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def key(self, version: DatasetVersion, **params: Any) -> str:
        """The dataset fingerprint plus the request that shaped the result.

        Both halves matter: the same data profiled over different columns is a
        different answer, and the same request against changed data must not
        reuse the old one.
        """
        payload = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"{version.fingerprint}:{digest}"

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if time.monotonic() - entry.stored_at > self.ttl_seconds:
                # Expiry is a backstop for a version that somehow did not move;
                # the fingerprint is what actually guarantees freshness.
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry.value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._entries[key] = _Entry(value=value, stored_at=time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }


# Module-level instance, shared by the routes.
PROFILE_CACHE = ProfileCache()


__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_TTL_SECONDS",
    "PROFILE_CACHE",
    "DatasetVersion",
    "ProfileCache",
    "dataset_version",
]
