"""OpenWeather adapter (task 2.1, specs §5.1).

Endpoint: ``GET /data/2.5/weather?lat=&lon=&appid=&units=metric``

``units=metric`` is not optional here: the default is Kelvin, and a silent
273-degree offset in the temperature column would poison every correlation and
every model that follows. The request asks for Celsius explicitly, and the
schema range-checks what comes back.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from core.config import Settings
from core.exceptions import DataSourceUnavailableError, SchemaValidationError
from services.adapters.base import AdapterSpec, SourceAdapter, SourceDomain
from services.geo_service import Station
from services.harmonizer import SourceReading, harmonize_coordinates, to_utc

BASE_URL = "https://api.openweathermap.org"

# As in the AQICN and TomTom adapters: the station asked about, carried through
# the payload so every source keys its rows the same way.
REQUESTED_POINT_KEY = "_requested_point"
REQUEST_TIMEOUT = 10.0

# Recorded extremes on Earth are roughly -90 °C to +57 °C. A value outside this
# is a unit error (Kelvin, Fahrenheit) rather than weather.
TEMPERATURE_BOUNDS = (-95.0, 65.0)


class OpenWeatherCoord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lat: float
    lon: float


class OpenWeatherMain(BaseModel):
    model_config = ConfigDict(extra="ignore")

    temp: float | None = None
    humidity: float | None = Field(default=None, ge=0, le=100)


class OpenWeatherResponse(BaseModel):
    """SEC-1: validated at the boundary, including the unit sanity check."""

    model_config = ConfigDict(extra="ignore")

    coord: OpenWeatherCoord
    requested_point: tuple[float, float] = Field(alias=REQUESTED_POINT_KEY)
    main: OpenWeatherMain = Field(default_factory=OpenWeatherMain)
    dt: int  # epoch seconds, UTC by OpenWeather's contract
    name: str | None = None


class OpenWeatherAdapter(SourceAdapter):
    spec = AdapterSpec(
        name="OpenWeather",
        domain=SourceDomain.WEATHER,
        api_url=f"{BASE_URL}/data/2.5/weather",
        authoritative_for=frozenset({"temp", "humidity"}),
        description="Current conditions — temperature (°C) and relative humidity.",
    )

    def credential(self, settings: Settings) -> str:
        return settings.openweather_api_key.get_secret_value()

    def fetch(
        self, settings: Settings, *, stations: Sequence[Station]
    ) -> Iterator[Mapping[str, Any]]:
        appid = self.credential(settings)
        with httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT) as client:
            for station in stations:
                try:
                    response = client.get(
                        "/data/2.5/weather",
                        params={
                            "lat": station.lat,
                            "lon": station.lon,
                            "appid": appid,
                            "units": "metric",
                        },
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise DataSourceUnavailableError(
                        f"OpenWeather request failed for {station.district_id}.",
                        details={"station": station.district_id, "error": str(exc)},
                    ) from exc
                payload = dict(response.json())
                payload[REQUESTED_POINT_KEY] = [station.lat, station.lon]
                yield payload

    def parse(self, payload: Mapping[str, Any]) -> list[SourceReading]:
        parsed = OpenWeatherResponse.model_validate(payload)

        temperature = parsed.main.temp
        if temperature is not None:
            low, high = TEMPERATURE_BOUNDS
            if not low <= temperature <= high:
                raise SchemaValidationError(
                    f"Temperature {temperature} is outside plausible bounds; "
                    "the response is probably not in Celsius.",
                    details={"temp": temperature, "bounds": list(TEMPERATURE_BOUNDS)},
                )

        # The requested point, not `coord`. OpenWeather usually echoes the
        # queried coordinate back, which is why this was aligned by accident
        # rather than by design -- but it is free to answer from its own grid
        # cell, and a reading that lands a few hundred metres off merges with
        # nothing. Keying every source on the station asked about is what makes
        # three feeds collapse into one row.
        lat, lon = harmonize_coordinates(*parsed.requested_point)

        return [
            SourceReading(
                timestamp=to_utc(parsed.dt),
                lat=lat,
                lon=lon,
                temp=temperature,
                humidity=parsed.main.humidity,
            )
        ]


__all__ = ["OpenWeatherAdapter", "OpenWeatherResponse", "TEMPERATURE_BOUNDS"]
