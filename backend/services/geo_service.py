"""District boundary access (task 1.8).

The GeoJSON is a static asset in the raw landing zone. It is read once and held
in memory: it is small, it never changes at runtime, and the map layer requests
it on every dashboard load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config import Settings, get_settings
from core.exceptions import DatasetNotFoundError

_CACHE: dict[Path, dict[str, Any]] = {}


def load_districts(settings: Settings | None = None) -> dict[str, Any]:
    """Return the district FeatureCollection.

    Raises ``DatasetNotFoundError`` (404) rather than letting an OSError become
    a 500: a missing boundary file is a deployment state the caller can act on,
    not a crash.
    """
    settings = settings or get_settings()
    path = settings.districts_geojson_path

    cached = _CACHE.get(path)
    if cached is not None:
        return cached

    if not path.is_file():
        raise DatasetNotFoundError(
            "District boundaries are not available.",
            details={"expected_path": str(path)},
        )

    try:
        collection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetNotFoundError(
            "District boundaries could not be read.",
            details={"expected_path": str(path), "reason": str(exc)},
        ) from exc

    _CACHE[path] = collection
    return collection


def district_centroids(settings: Settings | None = None) -> dict[str, tuple[float, float]]:
    """``district_id -> (lat, lon)``.

    The demo seed places its virtual stations on these centroids, so seeded
    observations land inside the polygons the map renders (task 1.9).
    """
    collection = load_districts(settings)
    return {
        feature["properties"]["district_id"]: (
            float(feature["properties"]["centroid_lat"]),
            float(feature["properties"]["centroid_lon"]),
        )
        for feature in collection["features"]
    }


def clear_cache() -> None:
    """Drop the in-memory copy. Used by tests that swap the data directory."""
    _CACHE.clear()


__all__ = ["clear_cache", "district_centroids", "load_districts"]
