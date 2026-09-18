"""Loading observations into pandas.

The boundary where the project stops being a record stream and becomes a
matrix. Ingestion (Phase 2) deliberately avoids pandas; from here on -- data
quality, EDA, features, modelling -- a DataFrame is the right shape, and every
one of those phases wants the same frame, so it is built in one place.

``station`` is the grouping key everything downstream uses. Statistics that mix
stations are almost always wrong: a forward fill must not carry one sensor's
value into another's gap, and an outlier bound computed across a city's whole
pollution gradient flags the dirtiest district rather than any real anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.exceptions import InsufficientDataError
from db.models import Observation

# The measured columns, in the order they appear in ``observations``. Named
# once so the quality engine, the EDA profile and the feature builder cannot
# drift apart about what "the feature set" means.
MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "pm25",
    "pm10",
    "temp",
    "humidity",
    "traffic_score",
)

# Columns that identify a row rather than measure anything.
IDENTITY_COLUMNS: tuple[str, ...] = ("id", "source_id", "timestamp", "lat", "lon")

STATION_COLUMN = "station"


@dataclass(frozen=True, slots=True)
class DatasetWindow:
    """What a loaded frame covers. Travels with reports so a number is never
    quoted without the slice of data it came from."""

    rows: int
    stations: int
    start: datetime | None
    end: datetime | None
    source_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "stations": self.stations,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "source_ids": list(self.source_ids),
        }


def station_key(lat: float, lon: float) -> str:
    """A stable label for one location.

    Coordinates are already rounded to 5 dp by the harmonizer, so this is a
    faithful identifier rather than a fuzzy match.
    """
    return f"{lat:.5f},{lon:.5f}"


def load_observations(
    session: Session,
    *,
    source_ids: tuple[int, ...] | list[int] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read ``observations`` into a time-sorted frame.

    Sorted by ``(station, timestamp)`` because every later step -- gap filling,
    lag features, rolling windows -- assumes a per-station series in
    chronological order, and sorting once here means none of them has to
    remember to do it.
    """
    statement = select(Observation).order_by(Observation.timestamp)

    if source_ids:
        statement = statement.where(Observation.source_id.in_(list(source_ids)))
    if start is not None:
        statement = statement.where(Observation.timestamp >= start)
    if end is not None:
        statement = statement.where(Observation.timestamp <= end)
    if limit is not None:
        statement = statement.limit(limit)

    records = [
        {
            "id": row.id,
            "source_id": row.source_id,
            "timestamp": row.timestamp,
            "lat": row.lat,
            "lon": row.lon,
            "pm25": row.pm25,
            "pm10": row.pm10,
            "temp": row.temp,
            "humidity": row.humidity,
            "traffic_score": row.traffic_score,
            "is_anomaly": row.is_anomaly,
        }
        for row in session.scalars(statement)
    ]

    frame = pd.DataFrame(records, columns=[*IDENTITY_COLUMNS, *MEASUREMENT_COLUMNS, "is_anomaly"])
    return prepare_frame(frame)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and add the station key. Safe on an empty frame."""
    frame = frame.copy()

    if frame.empty:
        frame[STATION_COLUMN] = pd.Series(dtype="object")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        for column in MEASUREMENT_COLUMNS:
            frame[column] = pd.Series(dtype="float64")
        return frame

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    # float64 throughout: SQL NULL arrives as None, and an object column would
    # make every later isna()/mean() call quietly do the wrong thing.
    for column in MEASUREMENT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    frame[STATION_COLUMN] = [
        station_key(lat, lon) for lat, lon in zip(frame["lat"], frame["lon"], strict=True)
    ]
    # Cast via a null check rather than fillna: fillna on an object column is
    # a deprecated implicit downcast in pandas 2.x.
    frame["is_anomaly"] = frame["is_anomaly"].notna() & frame["is_anomaly"].astype(bool)

    return frame.sort_values([STATION_COLUMN, "timestamp"]).reset_index(drop=True)


def describe_window(frame: pd.DataFrame, source_ids: tuple[int, ...] = ()) -> DatasetWindow:
    if frame.empty:
        return DatasetWindow(rows=0, stations=0, start=None, end=None, source_ids=source_ids)

    return DatasetWindow(
        rows=len(frame),
        stations=int(frame[STATION_COLUMN].nunique()),
        start=frame["timestamp"].min().to_pydatetime(),
        end=frame["timestamp"].max().to_pydatetime(),
        source_ids=source_ids or tuple(sorted(frame["source_id"].unique().tolist())),
    )


def require_rows(frame: pd.DataFrame, *, minimum: int, what: str) -> None:
    """Fail with a domain error rather than letting numpy raise on an empty
    array three call frames later."""
    if len(frame) < minimum:
        raise InsufficientDataError(
            f"{what} needs at least {minimum} observations; {len(frame)} available.",
            details={"required": minimum, "available": len(frame)},
        )


__all__ = [
    "IDENTITY_COLUMNS",
    "MEASUREMENT_COLUMNS",
    "STATION_COLUMN",
    "DatasetWindow",
    "describe_window",
    "load_observations",
    "prepare_frame",
    "require_rows",
    "station_key",
]
