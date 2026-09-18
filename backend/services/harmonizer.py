"""Harmonization: one shape, one clock, one grid (DR-2 … DR-4, design §6.2).

Every source passes through here before it can reach ``observations``. Three
transformations, each answering a specific way that multi-source data goes
wrong:

* **Timestamps → UTC** (DR-2). Sources report in local time, in epoch seconds,
  in ISO strings with and without offsets. Storing any of that as-is means lag
  features computed across sources are silently misaligned by hours.
* **Coordinates → decimal degrees** (DR-3). Feeds report degrees-minutes-
  seconds, hemisphere suffixes, or plain floats. One convention, validated.
* **Resample → hourly mean** (DR-4). Sources sample on their own cadence; a
  shared grid is what makes a join between air quality and weather meaningful
  at all.

Deliberately **no pandas**. Ingestion handles a stream of records, not a
matrix, and pure Python keeps the write path free of the scientific stack that
Phases 3-4 rightly pull in for analysis.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from core.exceptions import HarmonizationError

# The measurement columns of ``observations``. Named once here so the resampler
# and the write path cannot drift apart.
MEASUREMENT_FIELDS: tuple[str, ...] = (
    "pm25",
    "pm10",
    "temp",
    "humidity",
    "traffic_score",
)

# ~1.1 m. Live feeds jitter the last digits of a station's coordinates between
# calls; without rounding, one physical sensor would resample into several
# "locations" and the hourly grid would fragment.
COORDINATE_PRECISION = 5

_AXIS_HEMISPHERES = {"lat": ("N", "S"), "lon": ("E", "W")}
_AXIS_LIMITS = {"lat": 90.0, "lon": 180.0}

# Accepts 28.61 | 28.61N | -77.21 | 28°36'36"N | 28 36 36 N | 28:36:36N
_COORDINATE = re.compile(
    r"""
    ^\s*
    (?P<sign>[+-])?\s*
    (?P<deg>\d+(?:\.\d+)?)\s*(?:°|d|deg|:)?\s*
    (?:
        (?P<min>\d+(?:\.\d+)?)\s*(?:'|′|m|min|:)?\s*
        (?:(?P<sec>\d+(?:\.\d+)?)\s*(?:"|″|s|sec)?\s*)?
    )?
    (?P<hemisphere>[NSEW])?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class SourceReading:
    """One measurement event, after schema validation and harmonization.

    The common currency between adapters and the write path: an adapter's only
    job is to turn whatever its provider returns into these.
    """

    timestamp: datetime
    lat: float
    lon: float
    pm25: float | None = None
    pm10: float | None = None
    temp: float | None = None
    humidity: float | None = None
    traffic_score: float | None = None

    def measurements(self) -> dict[str, float | None]:
        return {field: getattr(self, field) for field in MEASUREMENT_FIELDS}

    def is_empty(self) -> bool:
        """True when the reading carries no measurement at all.

        Worth checking before a write: a row of nothing but a timestamp and a
        coordinate adds a phantom observation the quality engine would later
        report as 100% missing.
        """
        return all(value is None for value in self.measurements().values())


# --- DR-2: timestamps -------------------------------------------------------


def to_utc(
    value: datetime | str | int | float,
    *,
    assume_timezone: timezone = timezone.utc,
    field: str = "timestamp",
) -> datetime:
    """Coerce any supported timestamp representation to an aware UTC datetime.

    A naive value is interpreted in ``assume_timezone``. That assumption is
    always the caller's explicit choice -- the upload endpoint takes it as a
    parameter and records it in the run log -- because guessing silently is how
    a dataset acquires a fixed offset error nobody can later explain.
    """
    if isinstance(value, bool):  # bool is an int; never a timestamp
        raise HarmonizationError(f"{field} must be a timestamp, not a boolean.")

    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            raise HarmonizationError(f"{field} is not a finite epoch value.")
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise HarmonizationError(f"{field} is not a valid epoch: {value!r}") from exc

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise HarmonizationError(f"{field} is empty.")
        # "Z" is valid ISO 8601 but fromisoformat only accepts it from 3.11 on
        # some paths; normalising it costs nothing and removes the doubt.
        candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        try:
            value = datetime.fromisoformat(candidate)
        except ValueError:
            # A bare epoch that arrived as a string, which CSV makes common.
            try:
                return datetime.fromtimestamp(float(text), tz=timezone.utc)
            except (ValueError, OverflowError, OSError) as exc:
                raise HarmonizationError(
                    f"{field} is not an ISO 8601 timestamp or epoch: {value!r}"
                ) from exc

    if not isinstance(value, datetime):
        raise HarmonizationError(f"{field} has unsupported type {type(value).__name__}.")

    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        value = value.replace(tzinfo=assume_timezone)

    return value.astimezone(timezone.utc)


def floor_to_hour(moment: datetime) -> datetime:
    """DR-4: the hourly grid every source is resampled onto.

    Converts to UTC *before* flooring. Flooring first would truncate in
    whatever zone the value happens to carry, so a reading at 17:30+05:30 would
    land on 17:00 local -- 11:30 UTC -- which is not on the hourly grid at all
    and would silently shard one hour into two buckets.

    Naive input is rejected rather than assumed: ``astimezone`` on a naive
    datetime quietly interprets it in the *server's* local zone, which is how a
    dataset ends up with an offset that depends on where it was deployed.
    """
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise HarmonizationError(
            "Cannot place a naive timestamp on the hourly grid; convert it first (DR-2)."
        )
    return moment.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


# --- DR-3: coordinates ------------------------------------------------------


def to_decimal_degrees(value: float | int | str, *, axis: str) -> float:
    """Coerce a coordinate to decimal degrees, validated for its axis.

    ``axis`` is ``"lat"`` or ``"lon"``. It is required rather than inferred
    because the range check and the hemisphere check both depend on it, and a
    transposed pair is the single most common coordinate bug there is.
    """
    if axis not in _AXIS_LIMITS:
        raise ValueError(f"axis must be 'lat' or 'lon', got {axis!r}")

    if isinstance(value, bool):
        raise HarmonizationError(f"{axis} must be a coordinate, not a boolean.")

    if isinstance(value, (int, float)):
        degrees = float(value)
    elif isinstance(value, str):
        degrees = _parse_coordinate_string(value, axis=axis)
    else:
        raise HarmonizationError(f"{axis} has unsupported type {type(value).__name__}.")

    if math.isnan(degrees) or math.isinf(degrees):
        raise HarmonizationError(f"{axis} is not a finite number.")

    limit = _AXIS_LIMITS[axis]
    if not -limit <= degrees <= limit:
        raise HarmonizationError(
            f"{axis}={degrees} is outside [-{limit:g}, {limit:g}]; "
            "the value is probably in radians or the pair is transposed."
        )

    return round(degrees, COORDINATE_PRECISION)


def _parse_coordinate_string(raw: str, *, axis: str) -> float:
    match = _COORDINATE.match(raw)
    if match is None:
        raise HarmonizationError(f"{axis} is not a recognisable coordinate: {raw!r}")

    parts = match.groupdict()
    degrees = float(parts["deg"])
    minutes = float(parts["min"] or 0.0)
    seconds = float(parts["sec"] or 0.0)

    if minutes >= 60 or seconds >= 60:
        raise HarmonizationError(f"{axis} has out-of-range minutes/seconds: {raw!r}")

    value = degrees + minutes / 60 + seconds / 3600

    hemisphere = (parts["hemisphere"] or "").upper()
    if hemisphere:
        expected = _AXIS_HEMISPHERES[axis]
        if hemisphere not in expected:
            raise HarmonizationError(
                f"{axis} carries hemisphere {hemisphere!r}; expected one of {expected}. "
                "Latitude and longitude are probably swapped."
            )
        if hemisphere in ("S", "W"):
            value = -value

    if parts["sign"] == "-":
        if hemisphere in ("S", "W"):
            raise HarmonizationError(
                f"{axis} is negative and also marked {hemisphere!r}: {raw!r}"
            )
        value = -value

    return value


def harmonize_coordinates(lat: float | int | str, lon: float | int | str) -> tuple[float, float]:
    """Both axes together, so a transposition is caught by the range check."""
    return to_decimal_degrees(lat, axis="lat"), to_decimal_degrees(lon, axis="lon")


# --- DR-4: hourly resample --------------------------------------------------


def resample_hourly(readings: Iterable[SourceReading]) -> list[SourceReading]:
    """Collapse readings onto the hourly grid, averaging within each hour.

    Grouped by ``(hour, lat, lon)``: averaging across locations would smear a
    city's spatial gradient into a single meaningless number. Each field is
    averaged over the values that are actually present, so one missing
    temperature in an hour does not discard that hour's PM2.5.

    Output is sorted by ``(timestamp, lat, lon)`` -- deterministic ordering
    makes the write path's batching reproducible and the tests readable.
    """
    buckets: dict[tuple[datetime, float, float], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for reading in readings:
        key = (floor_to_hour(reading.timestamp), reading.lat, reading.lon)
        bucket = buckets[key]
        for field, value in reading.measurements().items():
            if value is not None:
                bucket[field].append(value)

    resampled: list[SourceReading] = []
    for (hour, lat, lon), bucket in buckets.items():
        averages = {
            field: (round(sum(values) / len(values), 4) if (values := bucket.get(field)) else None)
            for field in MEASUREMENT_FIELDS
        }
        resampled.append(SourceReading(timestamp=hour, lat=lat, lon=lon, **averages))

    resampled.sort(key=lambda reading: (reading.timestamp, reading.lat, reading.lon))
    return resampled


def harmonize(
    readings: Iterable[SourceReading], *, drop_empty: bool = True
) -> list[SourceReading]:
    """Apply the grid step and drop readings that carry no measurement.

    Timestamps and coordinates are harmonized by the adapters as they build
    each ``SourceReading`` -- that is where the provider-specific shape is
    known. This is the part that has to see the whole stream at once.
    """
    resampled = resample_hourly(readings)
    return [r for r in resampled if not (drop_empty and r.is_empty())]


def iter_normalized(
    readings: Iterable[SourceReading],
    *,
    assume_timezone: timezone = timezone.utc,
) -> Iterator[SourceReading]:
    """Re-apply DR-2/DR-3 to readings built outside an adapter.

    Used by the upload path, where the values come from a file rather than from
    code that already guaranteed their shape.
    """
    for reading in readings:
        lat, lon = harmonize_coordinates(reading.lat, reading.lon)
        yield replace(
            reading,
            timestamp=to_utc(reading.timestamp, assume_timezone=assume_timezone),
            lat=lat,
            lon=lon,
        )


__all__ = [
    "COORDINATE_PRECISION",
    "MEASUREMENT_FIELDS",
    "SourceReading",
    "floor_to_hour",
    "harmonize",
    "harmonize_coordinates",
    "iter_normalized",
    "resample_hourly",
    "to_decimal_degrees",
    "to_utc",
]
