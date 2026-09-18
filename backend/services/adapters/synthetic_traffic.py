"""Synthetic traffic fallback (task 2.2, specs §5.1).

specs §5.1 lists the traffic proxy as "TomTom API **or synthetic baseline**".
This is that baseline: when no traffic key is configured, the platform still
produces a congestion series rather than leaving the column empty, because
PM2.5 without traffic loses the strongest explanatory variable the project has.

It is honest about what it is. The source is registered under a name that says
"synthetic", it never claims to be measured, and it yields the same 0–100
congestion index TomTom is converted to, so the two are interchangeable inputs
to everything downstream.

Needs no credential and no network, which is what lets it satisfy DR-1.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.config import Settings
from services.adapters.base import AdapterSpec, SourceAdapter, SourceDomain
from services.demo_data import synthetic_traffic_score
from services.geo_service import Station
from services.harmonizer import (
    SourceReading,
    floor_to_hour,
    harmonize_coordinates,
    to_utc,
)


class SyntheticTrafficPayload(BaseModel):
    """SEC-1: even a locally generated record goes through a schema.

    Not ceremony -- it means the synthetic path exercises exactly the same
    validation and quarantine machinery as a live feed, so a bug there cannot
    hide behind "well, we generated it ourselves".
    """

    model_config = ConfigDict(extra="ignore")

    district_id: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    observed_at: datetime | str | int
    traffic_score: float = Field(ge=0, le=100)
    synthetic: bool = True


class SyntheticTrafficAdapter(SourceAdapter):
    spec = AdapterSpec(
        name="Synthetic Traffic (fallback)",
        domain=SourceDomain.TRAFFIC,
        api_url=None,  # nothing to call
        requires_credentials=False,
        description="Modelled congestion index used when no traffic API key is set.",
    )

    def fetch(
        self, settings: Settings, *, stations: Sequence[Station]
    ) -> Iterator[Mapping[str, Any]]:
        # The current hour, so the fallback lands on the same grid as everything
        # else and a later real fetch for that hour can supersede it.
        moment = floor_to_hour(datetime.now(timezone.utc))
        for station in stations:
            yield {
                "district_id": station.district_id,
                "lat": station.lat,
                "lon": station.lon,
                "observed_at": moment.isoformat(),
                "traffic_score": synthetic_traffic_score(moment, station=station),
                "synthetic": True,
            }

    def parse(self, payload: Mapping[str, Any]) -> list[SourceReading]:
        parsed = SyntheticTrafficPayload.model_validate(payload)
        lat, lon = harmonize_coordinates(parsed.lat, parsed.lon)

        return [
            SourceReading(
                timestamp=to_utc(parsed.observed_at),
                lat=lat,
                lon=lon,
                traffic_score=parsed.traffic_score,
            )
        ]


__all__ = ["SyntheticTrafficAdapter", "SyntheticTrafficPayload"]
