"""TomTom traffic adapter (task 2.1, specs §5.1).

Endpoint: ``GET /traffic/services/4/flowSegmentData/absolute/10/json?point=lat,lon``

Two things this adapter has to decide, because the provider does not:

* **There is no timestamp in the response.** Flow data is "now". ``fetch``
  therefore stamps each payload with the moment it was retrieved, under a
  reserved key, so that ``parse`` stays a pure function of its input and the
  quarantined evidence still shows when the call was made.
* **The score.** TomTom reports speeds, not congestion. The stored
  ``traffic_score`` is the standard congestion index
  ``100 × (1 − current_speed ÷ free_flow_speed)``: 0 means traffic is moving at
  the road's free-flow speed, 100 means it is stopped. That matches the 0–100
  range the synthetic fallback produces, so the two are interchangeable.

The reading is stored at the **requested** station coordinates rather than at
the matched road segment's, so traffic lines up with the other sources sampled
at the same station.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from core.config import Settings
from core.exceptions import DataSourceUnavailableError, SchemaValidationError
from services.adapters.base import AdapterSpec, SourceAdapter, SourceDomain
from services.geo_service import Station
from services.harmonizer import SourceReading, harmonize_coordinates, to_utc

BASE_URL = "https://api.tomtom.com"
FLOW_PATH = "/traffic/services/4/flowSegmentData/absolute/10/json"
REQUEST_TIMEOUT = 10.0

# Keys this adapter adds because the provider supplies no equivalent. Prefixed
# so they cannot collide with a field TomTom might introduce later.
OBSERVED_AT_KEY = "_observed_at"
REQUESTED_POINT_KEY = "_requested_point"


class FlowSegmentData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    currentSpeed: float | None = None  # noqa: N815 - provider's spelling
    freeFlowSpeed: float | None = None  # noqa: N815
    confidence: float | None = Field(default=None, ge=0, le=1)
    roadClosure: bool = False  # noqa: N815


class TomTomResponse(BaseModel):
    """SEC-1. The two underscore-prefixed fields are stamped on by ``fetch``."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    flowSegmentData: FlowSegmentData  # noqa: N815
    observed_at: datetime | str | int = Field(alias=OBSERVED_AT_KEY)
    requested_point: tuple[float, float] = Field(alias=REQUESTED_POINT_KEY)


class TomTomAdapter(SourceAdapter):
    spec = AdapterSpec(
        name="TomTom Traffic",
        domain=SourceDomain.TRAFFIC,
        api_url=f"{BASE_URL}{FLOW_PATH}",
        authoritative_for=frozenset({"traffic_score"}),
        description="Road flow converted to a 0–100 congestion index.",
    )

    def credential(self, settings: Settings) -> str:
        return settings.tomtom_api_key.get_secret_value()

    def fetch(
        self, settings: Settings, *, stations: Sequence[Station]
    ) -> Iterator[Mapping[str, Any]]:
        key = self.credential(settings)
        with httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT) as client:
            for station in stations:
                try:
                    response = client.get(
                        FLOW_PATH,
                        params={"point": f"{station.lat},{station.lon}", "key": key},
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise DataSourceUnavailableError(
                        f"TomTom request failed for {station.district_id}.",
                        details={"station": station.district_id, "error": str(exc)},
                    ) from exc

                payload = dict(response.json())
                payload[OBSERVED_AT_KEY] = datetime.now(timezone.utc).isoformat()
                payload[REQUESTED_POINT_KEY] = [station.lat, station.lon]
                yield payload

    def parse(self, payload: Mapping[str, Any]) -> list[SourceReading]:
        parsed = TomTomResponse.model_validate(payload)
        flow = parsed.flowSegmentData

        lat, lon = harmonize_coordinates(*parsed.requested_point)

        return [
            SourceReading(
                timestamp=to_utc(parsed.observed_at),
                lat=lat,
                lon=lon,
                traffic_score=congestion_index(
                    current_speed=flow.currentSpeed,
                    free_flow_speed=flow.freeFlowSpeed,
                    road_closure=flow.roadClosure,
                ),
            )
        ]


def congestion_index(
    *,
    current_speed: float | None,
    free_flow_speed: float | None,
    road_closure: bool = False,
) -> float | None:
    """0 = free flowing, 100 = stopped. ``None`` when it cannot be computed."""
    if road_closure:
        return 100.0
    if current_speed is None or free_flow_speed is None:
        return None
    if free_flow_speed <= 0:
        raise SchemaValidationError(
            "TomTom reported a non-positive free-flow speed; the segment is unusable.",
            details={"free_flow_speed": free_flow_speed},
        )
    # Current speed can exceed free flow on a quiet road; clamping keeps the
    # score inside the range the column and the UI both assume.
    score = 100.0 * (1.0 - current_speed / free_flow_speed)
    return round(min(100.0, max(0.0, score)), 2)


__all__ = [
    "OBSERVED_AT_KEY",
    "REQUESTED_POINT_KEY",
    "TomTomAdapter",
    "TomTomResponse",
    "congestion_index",
]
