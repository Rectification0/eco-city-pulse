"""Demo bundle adapter (task 2.9, DR-1, AC-2).

design §6.3 lists Demo as one of four modes sharing a single code path. This
adapter is what makes that literally true: the offline dataset enters the
database through the same fetch → validate → harmonize → quarantine → write
pipeline as a live API, rather than through a private shortcut.

That matters for AC-2. "Ingestion completes end-to-end in Demo mode with every
live API disabled" is only a meaningful claim if demo mode exercises the real
ingestion path — otherwise it tests a code path nobody uses in production.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.config import Settings
from services import demo_data
from services.adapters.base import AdapterSpec, SourceAdapter, SourceDomain
from services.geo_service import Station
from services.harmonizer import SourceReading, harmonize_coordinates, to_utc


class DemoPayload(BaseModel):
    """SEC-1. Ranges match the CHECK constraints on ``observations``, so a
    generator bug is caught here rather than by the database."""

    model_config = ConfigDict(extra="ignore")

    district_id: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    timestamp: datetime | str | int
    pm25: float | None = None
    pm10: float | None = None
    temp: float | None = None
    humidity: float | None = Field(default=None, ge=0, le=100)
    traffic_score: float | None = Field(default=None, ge=0, le=100)
    synthetic: bool = True


class DemoAdapter(SourceAdapter):
    """Replays the offline generator. Configurable window via the constructor.

    ``days`` and ``end`` are constructor arguments rather than fetch arguments
    because the adapter interface is deliberately uniform -- a live source has
    no equivalent knob, and adding one to the protocol for this adapter's sake
    would leak demo concerns into every other implementation.
    """

    spec = AdapterSpec(
        name=demo_data.DEMO_SOURCE_NAME,
        domain=SourceDomain.BUNDLE,
        api_url=None,
        requires_credentials=False,
        description="Offline synthetic history: air quality, weather, traffic.",
    )

    def __init__(
        self,
        *,
        days: int = demo_data.DEFAULT_DAYS,
        end: datetime | None = None,
        seed: int = demo_data.DEFAULT_SEED,
    ) -> None:
        self.days = days
        self.end = end
        self.seed = seed

    def fetch(
        self, settings: Settings, *, stations: Sequence[Station]
    ) -> Iterator[Mapping[str, Any]]:
        end = demo_data.floor_to_hour(self.end or datetime.now(timezone.utc))
        for record in demo_data.generate_observations(
            days=self.days,
            end=end,
            stations=list(stations),
            seed=self.seed,
            settings=settings,
        ):
            yield {
                "district_id": record.district_id,
                "lat": record.lat,
                "lon": record.lon,
                "timestamp": record.timestamp.isoformat(),
                "pm25": record.pm25,
                "pm10": record.pm10,
                "temp": record.temp,
                "humidity": record.humidity,
                "traffic_score": record.traffic_score,
                "synthetic": True,
            }

    def parse(self, payload: Mapping[str, Any]) -> list[SourceReading]:
        parsed = DemoPayload.model_validate(payload)
        lat, lon = harmonize_coordinates(parsed.lat, parsed.lon)

        return [
            SourceReading(
                timestamp=to_utc(parsed.timestamp),
                lat=lat,
                lon=lon,
                pm25=parsed.pm25,
                pm10=parsed.pm10,
                temp=parsed.temp,
                humidity=parsed.humidity,
                traffic_score=parsed.traffic_score,
            )
        ]


__all__ = ["DemoAdapter", "DemoPayload"]
