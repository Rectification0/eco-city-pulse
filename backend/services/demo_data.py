"""Demo-mode dataset generator (task 1.9, DR-1, AC-2).

The platform must demonstrate the whole pipeline with every live API disabled,
so the demo bundle cannot be downloaded at seed time -- it has to be produced
locally. This module synthesises an hourly history for one station per city
district using only the standard library: no network, no pandas, no numpy.

The series are not noise. They carry the structure the later phases exist to
find, and each property below is deliberate:

* **seasonal + diurnal + weekly** components, so STL decomposition (4.5) has a
  trend and a seasonality to separate;
* **autocorrelated residuals** (AR(1)), so lag features (5.2) actually predict
  and a naive lag-1 baseline (7.5) is a genuinely hard target to beat;
* **cross-correlation** -- PM2.5 rises with traffic and falls with temperature
  -- so the correlation matrix (4.2) and PCA (6.2) have signal to report;
* **missing values of two different mechanisms** (see ``_apply_missingness``),
  so the MCAR/MAR characterisation (3.1) is a real judgement, not a formality;
* **occasional extreme spikes**, left with ``is_anomaly`` unset, so the outlier
  detectors of Phase 3 have something to detect.

Values are a function of the timestamp, not of when the generator runs: the
AR chain is always replayed from ``DEMO_EPOCH``. Re-seeding therefore produces
identical rows for any hour it produced before.

Nothing here claims to be measured data. It is synthetic, shaped to resemble a
North Indian metro's pollution regime; it is labelled as such everywhere it is
persisted.
"""

from __future__ import annotations

import math
import random
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from core.config import Settings
from services import geo_service

# Anchor for the autocorrelated residual chain. Fixed so that a given hour gets
# the same value no matter which window a run asks for.
DEMO_EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)

DEFAULT_SEED = 20260918
DEFAULT_DAYS = 120

# India Standard Time. Diurnal and weekly rhythms are human-clock effects, so
# they are computed in local time and only the stored timestamp is UTC (DR-2).
LOCAL_OFFSET = timedelta(hours=5, minutes=30)

DEMO_SOURCE_NAME = "Demo Bundle (synthetic, offline)"

# The live adapters of specs §5.1. Seeded as rows so /data/sources (task 2.10)
# reports them as offline rather than as absent -- the distinction matters: an
# offline source is configured but keyless, which is the expected demo state.
LIVE_SOURCES: tuple[tuple[str, str], ...] = (
    ("AQICN", "https://api.waqi.info/feed/"),
    ("OpenWeather", "https://api.openweathermap.org/data/2.5/"),
    ("TomTom Traffic", "https://api.tomtom.com/traffic/services/4/"),
)


@dataclass(frozen=True, slots=True)
class Station:
    """A virtual monitoring station sitting on a district centroid."""

    district_id: str
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class DemoObservation:
    """One synthetic hourly reading, in the shape of an ``observations`` row."""

    timestamp: datetime
    lat: float
    lon: float
    district_id: str
    pm25: float | None
    pm10: float | None
    temp: float | None
    humidity: float | None
    traffic_score: float | None


def stations_from_districts(settings: Settings | None = None) -> list[Station]:
    """One station per district, placed on the centroid from task 1.8.

    Tying the two together means seeded points always fall inside the polygons
    the map draws, so the spatial gradient on the dashboard is real rather than
    a coincidence of two independent coordinate lists.
    """
    centroids = geo_service.district_centroids(settings)
    return [
        Station(district_id=district_id, lat=lat, lon=lon)
        for district_id, (lat, lon) in sorted(centroids.items())
    ]


def _station_offset(station: Station) -> float:
    """Baseline PM2.5 offset in ug/m3 giving the city a spatial gradient.

    North and east run dirtier than the south-west, which is the broad pattern
    in the region being imitated. Derived from the coordinates rather than a
    lookup table so it stays consistent if the district set changes.
    """
    north = (station.lat - 28.40) / 0.49  # 0 (south) .. 1 (north)
    east = (station.lon - 76.84) / 0.51  # 0 (west) .. 1 (east)
    return 16.0 * north + 7.0 * east - 11.0


def _hours_between(start: datetime, end: datetime) -> int:
    return int((end - start).total_seconds() // 3600)


def _seasonal(day_of_year: int, *, peak_day: int, amplitude: float) -> float:
    """Annual cycle peaking on ``peak_day``, in [-amplitude, +amplitude]."""
    return amplitude * math.cos(2 * math.pi * (day_of_year - peak_day) / 365.25)


def _traffic(local: datetime) -> float:
    """Congestion index 0-100 with morning and evening peaks."""
    hour = local.hour + local.minute / 60
    morning = 30.0 * math.exp(-(((hour - 9.0) / 1.6) ** 2))
    evening = 34.0 * math.exp(-(((hour - 18.5) / 1.9) ** 2))
    overnight = -18.0 * math.exp(-(((hour - 3.5) / 2.6) ** 2))
    weekend = -12.0 if local.weekday() >= 5 else 0.0
    return 42.0 + morning + evening + overnight + weekend


def _temperature(local: datetime, day_of_year: int) -> float:
    """Degrees Celsius: hot summer, mild winter, afternoon peak."""
    seasonal = 25.0 - _seasonal(day_of_year, peak_day=15, amplitude=9.0)
    diurnal = 6.0 * math.cos(2 * math.pi * (local.hour - 15) / 24)
    return seasonal + diurnal


def _humidity(local: datetime, day_of_year: int, temp: float) -> float:
    """Percent: monsoon-heavy, and inversely related to temperature."""
    monsoon = 22.0 * math.exp(-(((day_of_year - 210) / 45.0) ** 2))
    return 52.0 + monsoon - 0.9 * (temp - 25.0) - 4.0 * math.cos(
        2 * math.pi * (local.hour - 5) / 24
    )


def _pm25_signal(local: datetime, day_of_year: int, traffic: float, temp: float) -> float:
    """Deterministic part of PM2.5 in ug/m3, before noise and spikes."""
    seasonal = 68.0 + _seasonal(day_of_year, peak_day=10, amplitude=38.0)
    # Two peaks a day: the morning commute, and the evening inversion that traps
    # the day's emissions near the ground.
    diurnal = 11.0 * math.cos(2 * math.pi * (local.hour - 8) / 24) + 9.0 * math.cos(
        4 * math.pi * (local.hour - 9.5) / 24
    )
    weekly = -9.0 if local.weekday() == 6 else (4.0 if local.weekday() < 5 else 0.0)
    return (
        seasonal
        + diurnal
        + weekly
        + 0.30 * (traffic - 42.0)  # traffic contribution
        - 0.75 * (temp - 25.0)  # warmer air disperses better
    )


def _apply_missingness(
    rng: random.Random,
    *,
    pm25: float,
    pm10: float | None,
    temp: float | None,
    humidity: float | None,
    traffic: float | None,
    in_outage: bool,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Punch holes with two distinct mechanisms, as specs §5.3 requires.

    * **MCAR** -- a flat ~1.5% dropout on temperature and humidity, independent
      of everything: ordinary telemetry loss.
    * **MAR** -- PM10 goes missing far more often when PM2.5 is high, imitating
      an optical sensor saturating. Missingness depends on an *observed*
      variable, which is precisely what makes it MAR rather than MNAR, and what
      makes MICE (3.2) the right repair rather than a mean fill.

    Contiguous multi-hour outages arrive separately via ``in_outage`` -- those
    are the gaps forward/backward fill is meant for (3.3).
    """
    if in_outage:
        # A station dropout takes the whole telemetry package with it, except
        # the traffic feed, which comes from a different provider entirely.
        return None, None, None, None, traffic

    if temp is not None and rng.random() < 0.015:
        temp = None
    if humidity is not None and rng.random() < 0.015:
        humidity = None

    saturation_risk = 0.01 + (0.06 if pm25 > 150 else 0.0)
    if pm10 is not None and rng.random() < saturation_risk:
        pm10 = None

    if traffic is not None and rng.random() < 0.008:
        traffic = None

    return pm25, pm10, temp, humidity, traffic


def generate_station_series(
    station: Station,
    *,
    start: datetime,
    end: datetime,
    seed: int = DEFAULT_SEED,
) -> Iterator[DemoObservation]:
    """Hourly observations for one station over ``[start, end)``.

    The AR(1) residual and every random draw are replayed from ``DEMO_EPOCH``,
    so the values for a given hour do not depend on the requested window.
    """
    if start < DEMO_EPOCH:
        raise ValueError(f"start must be at or after {DEMO_EPOCH.isoformat()}")

    rng = random.Random(f"{seed}:{station.district_id}")
    offset = _station_offset(station)

    residual = 0.0
    spike_hours_left = 0
    spike_factor = 1.0

    total_hours = _hours_between(DEMO_EPOCH, end)
    emit_from = _hours_between(DEMO_EPOCH, start)

    for step in range(total_hours):
        moment = DEMO_EPOCH + timedelta(hours=step)
        local = moment + LOCAL_OFFSET
        day_of_year = local.timetuple().tm_yday

        # AR(1): today's residual remembers yesterday's, which is what gives the
        # lag features something to learn.
        residual = 0.86 * residual + rng.gauss(0.0, 6.5)

        traffic = _traffic(local) + rng.gauss(0.0, 4.0)
        temp = _temperature(local, day_of_year) + rng.gauss(0.0, 1.2)
        humidity = _humidity(local, day_of_year, temp) + rng.gauss(0.0, 3.5)
        pm25 = _pm25_signal(local, day_of_year, traffic, temp) + offset + residual

        # Episodic spikes: stubble burning, a festival, a still winter night.
        # Left unflagged on purpose -- detecting them is Phase 3's job (AC-5).
        if spike_hours_left > 0:
            spike_hours_left -= 1
        elif rng.random() < 0.0012:
            spike_hours_left = rng.randint(1, 4)
            spike_factor = rng.uniform(2.4, 4.0)
        pm25 = pm25 * spike_factor if spike_hours_left > 0 else pm25

        pm25 = max(3.0, pm25)
        pm10 = max(5.0, pm25 * rng.uniform(1.7, 2.1) + 14.0)
        humidity = min(100.0, max(5.0, humidity))
        traffic = min(100.0, max(0.0, traffic))

        # A ~7-hour station outage roughly every 19 days, phase-shifted per
        # station so the gaps do not line up across the city. crc32, not hash():
        # Python randomises string hashing per process, which would make the
        # gap positions differ between runs.
        outage_phase = (step + zlib.crc32(station.district_id.encode()) % 456) % 456
        in_outage = outage_phase < 7

        pm25_out, pm10_out, temp_out, humidity_out, traffic_out = _apply_missingness(
            rng,
            pm25=pm25,
            pm10=pm10,
            temp=temp,
            humidity=humidity,
            traffic=traffic,
            in_outage=in_outage,
        )

        if step < emit_from:
            continue

        yield DemoObservation(
            timestamp=moment,
            lat=station.lat,
            lon=station.lon,
            district_id=station.district_id,
            pm25=None if pm25_out is None else round(pm25_out, 2),
            pm10=None if pm10_out is None else round(pm10_out, 2),
            temp=None if temp_out is None else round(temp_out, 2),
            humidity=None if humidity_out is None else round(humidity_out, 2),
            traffic_score=None if traffic_out is None else round(traffic_out, 2),
        )


def floor_to_hour(moment: datetime) -> datetime:
    """DR-4: every demo timestamp sits exactly on the hourly grid."""
    return moment.replace(minute=0, second=0, microsecond=0)


def generate_observations(
    *,
    days: int = DEFAULT_DAYS,
    end: datetime | None = None,
    stations: Sequence[Station] | None = None,
    seed: int = DEFAULT_SEED,
    settings: Settings | None = None,
) -> Iterator[DemoObservation]:
    """The demo dataset: ``days`` of hourly history ending at ``end``.

    ``end`` defaults to the current hour so a fresh clone opens on a dashboard
    showing "now". The values themselves stay reproducible regardless (see the
    module docstring).
    """
    if days < 1:
        raise ValueError("days must be at least 1")

    end = floor_to_hour(end or datetime.now(timezone.utc))
    if end.tzinfo is None:
        raise ValueError("end must be timezone-aware (DR-2)")
    end = end.astimezone(timezone.utc)
    start = end - timedelta(days=days)

    for station in stations if stations is not None else stations_from_districts(settings):
        yield from generate_station_series(station, start=start, end=end, seed=seed)


__all__ = [
    "DEFAULT_DAYS",
    "DEFAULT_SEED",
    "DEMO_EPOCH",
    "DEMO_SOURCE_NAME",
    "LIVE_SOURCES",
    "DemoObservation",
    "Station",
    "floor_to_hour",
    "generate_observations",
    "generate_station_series",
    "stations_from_districts",
]
