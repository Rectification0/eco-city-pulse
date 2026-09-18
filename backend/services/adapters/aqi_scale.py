"""AQI index → mass concentration.

AQICN reports every pollutant as a **US EPA AQI index**, not as a
concentration. The ``observations.pm25`` column is documented in µg/m³, so
storing the index there would put two different quantities in one column --
an AQI of 155 and 155 µg/m³ are wildly different amounts of pollution, and no
downstream statistic would be meaningful.

The EPA scale is piecewise linear between fixed breakpoints, so the inverse is
exact within each segment. These are the pre-2024 PM2.5 breakpoints, which are
the ones AQICN's published scale uses.

The conversion is lossy in one direction only: AQI is reported as an integer,
so the recovered concentration carries the rounding of that integer. That is a
property of the upstream feed, not of this code, and it is why the demo bundle
(which needs no conversion) remains the better dataset for analysis.
"""

from __future__ import annotations

# (aqi_low, aqi_high, concentration_low, concentration_high)
Breakpoint = tuple[float, float, float, float]

PM25_BREAKPOINTS: tuple[Breakpoint, ...] = (
    (0.0, 50.0, 0.0, 12.0),
    (51.0, 100.0, 12.1, 35.4),
    (101.0, 150.0, 35.5, 55.4),
    (151.0, 200.0, 55.5, 150.4),
    (201.0, 300.0, 150.5, 250.4),
    (301.0, 400.0, 250.5, 350.4),
    (401.0, 500.0, 350.5, 500.4),
)

PM10_BREAKPOINTS: tuple[Breakpoint, ...] = (
    (0.0, 50.0, 0.0, 54.0),
    (51.0, 100.0, 55.0, 154.0),
    (101.0, 150.0, 155.0, 254.0),
    (151.0, 200.0, 255.0, 354.0),
    (201.0, 300.0, 355.0, 424.0),
    (301.0, 400.0, 425.0, 504.0),
    (401.0, 500.0, 505.0, 604.0),
)

BREAKPOINTS = {"pm25": PM25_BREAKPOINTS, "pm10": PM10_BREAKPOINTS}


def aqi_to_concentration(aqi: float, *, pollutant: str) -> float | None:
    """Invert the EPA AQI for one pollutant, in µg/m³.

    Returns ``None`` for an AQI outside the published scale rather than
    extrapolating: beyond 500 the scale is undefined, and inventing a number
    there would be a fabricated measurement.
    """
    table = BREAKPOINTS.get(pollutant)
    if table is None:
        raise ValueError(f"no AQI breakpoints for {pollutant!r}")

    if aqi < 0:
        return None

    for aqi_low, aqi_high, conc_low, conc_high in table:
        if aqi_low <= aqi <= aqi_high:
            span = aqi_high - aqi_low
            if span == 0:  # pragma: no cover - defensive
                return conc_low
            ratio = (aqi - aqi_low) / span
            return round(conc_low + ratio * (conc_high - conc_low), 2)

    return None


__all__ = ["BREAKPOINTS", "PM10_BREAKPOINTS", "PM25_BREAKPOINTS", "aqi_to_concentration"]
