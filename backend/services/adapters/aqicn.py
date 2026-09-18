"""AQICN air-quality adapter (task 2.1, specs §5.1).

Endpoint: ``GET /feed/geo:{lat};{lon}/?token=...``

Two details worth stating because they change what lands in the database:

* ``iaqi.pm25`` is an **AQI index**, not µg/m³, and is converted through
  ``aqi_scale``. ``iaqi.t`` and ``iaqi.h`` are already real units.
* The station AQICN resolves for a coordinate is the *nearest* one, which can
  be several kilometres away. The reading is stored at that station's own
  coordinates, not at the requested point, so the map never claims a
  measurement was taken where it was not.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from core.config import Settings
from core.exceptions import DataSourceUnavailableError, SchemaValidationError
from services.adapters.aqi_scale import aqi_to_concentration
from services.adapters.base import AdapterSpec, SourceAdapter, SourceDomain
from services.geo_service import Station
from services.harmonizer import SourceReading, harmonize_coordinates, to_utc

BASE_URL = "https://api.waqi.info"
REQUEST_TIMEOUT = 10.0


class AqicnValue(BaseModel):
    """One ``iaqi`` entry. Missing pollutants are simply absent from the dict."""

    model_config = ConfigDict(extra="ignore")

    v: float | None = None


class AqicnIaqi(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pm25: AqicnValue | None = None
    pm10: AqicnValue | None = None
    t: AqicnValue | None = None  # temperature, °C
    h: AqicnValue | None = None  # relative humidity, %


class AqicnCity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # [lat, lon] -- AQICN's order, the opposite of GeoJSON's.
    geo: tuple[float, float]
    name: str | None = None


class AqicnTime(BaseModel):
    model_config = ConfigDict(extra="ignore")

    iso: str | None = None
    s: str | None = None
    tz: str | None = None


class AqicnData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    city: AqicnCity
    iaqi: AqicnIaqi = Field(default_factory=AqicnIaqi)
    time: AqicnTime = Field(default_factory=AqicnTime)


class AqicnResponse(BaseModel):
    """SEC-1: the upstream payload is validated, never trusted as-is."""

    model_config = ConfigDict(extra="ignore")

    status: str
    # AQICN returns a *string* here when status != "ok" (e.g. "Unknown station"),
    # so this cannot be typed as AqicnData unconditionally.
    data: AqicnData | str


class AqicnAdapter(SourceAdapter):
    spec = AdapterSpec(
        name="AQICN",
        domain=SourceDomain.AIR_QUALITY,
        api_url=f"{BASE_URL}/feed/",
        description="World Air Quality Index — PM2.5, PM10, temperature, humidity.",
    )

    def credential(self, settings: Settings) -> str:
        return settings.aqicn_api_key.get_secret_value()

    def fetch(
        self, settings: Settings, *, stations: Sequence[Station]
    ) -> Iterator[Mapping[str, Any]]:
        token = self.credential(settings)
        with httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT) as client:
            for station in stations:
                try:
                    response = client.get(
                        f"/feed/geo:{station.lat};{station.lon}/",
                        params={"token": token},
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    # Raised, not swallowed: the ingestion service decides that
                    # an unreachable upstream degrades the source to offline
                    # (DR-1). The adapter only reports what happened.
                    raise DataSourceUnavailableError(
                        f"AQICN request failed for {station.district_id}.",
                        details={"station": station.district_id, "error": str(exc)},
                    ) from exc
                yield response.json()

    def parse(self, payload: Mapping[str, Any]) -> list[SourceReading]:
        parsed = AqicnResponse.model_validate(payload)

        if parsed.status != "ok" or isinstance(parsed.data, str):
            raise SchemaValidationError(
                "AQICN reported a non-ok status.",
                details={"status": parsed.status, "message": parsed.data
                         if isinstance(parsed.data, str) else None},
            )

        data = parsed.data
        lat, lon = harmonize_coordinates(data.city.geo[0], data.city.geo[1])

        stamp = data.time.iso or data.time.s
        if stamp is None:
            raise SchemaValidationError("AQICN payload carries no timestamp.")
        # time.s is local without an offset; time.tz supplies it separately.
        if data.time.iso is None and data.time.tz:
            stamp = f"{stamp.replace(' ', 'T')}{data.time.tz}"

        return [
            SourceReading(
                timestamp=to_utc(stamp),
                lat=lat,
                lon=lon,
                pm25=_concentration(data.iaqi.pm25, "pm25"),
                pm10=_concentration(data.iaqi.pm10, "pm10"),
                temp=data.iaqi.t.v if data.iaqi.t else None,
                humidity=data.iaqi.h.v if data.iaqi.h else None,
            )
        ]


def _concentration(entry: AqicnValue | None, pollutant: str) -> float | None:
    if entry is None or entry.v is None:
        return None
    return aqi_to_concentration(entry.v, pollutant=pollutant)


__all__ = ["AqicnAdapter", "AqicnResponse"]
