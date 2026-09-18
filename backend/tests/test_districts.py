"""District boundaries and the route that serves them (task 1.8, specs §5.1)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from core.exceptions import DatasetNotFoundError
from services import geo_service

EXPECTED_DISTRICTS = 11


@pytest.fixture(autouse=True)
def _clear_geo_cache() -> None:
    geo_service.clear_cache()


def test_boundary_file_is_committed(settings: Settings) -> None:
    """The demo must work on a fresh clone, so the file cannot be generated."""
    assert settings.districts_geojson_path.is_file()


def test_boundary_file_is_valid_geojson(settings: Settings) -> None:
    collection = json.loads(settings.districts_geojson_path.read_text(encoding="utf-8"))

    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == EXPECTED_DISTRICTS
    for feature in collection["features"]:
        ring = feature["geometry"]["coordinates"][0]
        # RFC 7946: a linear ring is closed -- first position repeated last.
        assert ring[0] == ring[-1]
        assert len(ring) >= 4


def test_coordinates_are_lon_lat_in_decimal_degrees(settings: Settings) -> None:
    """GeoJSON orders positions [lon, lat]; swapping them is the classic bug,
    and it would put Delhi in the Indian Ocean."""
    collection = geo_service.load_districts(settings)

    for feature in collection["features"]:
        for lon, lat in feature["geometry"]["coordinates"][0]:
            assert 76.0 <= lon <= 78.0
            assert 28.0 <= lat <= 29.0


def test_district_ids_are_unique(settings: Settings) -> None:
    collection = geo_service.load_districts(settings)
    ids = [f["properties"]["district_id"] for f in collection["features"]]

    assert len(set(ids)) == len(ids)


def test_centroids_fall_inside_their_own_polygon(settings: Settings) -> None:
    """The demo seed places stations on these centroids (task 1.9)."""
    collection = geo_service.load_districts(settings)

    for feature in collection["features"]:
        ring = feature["geometry"]["coordinates"][0]
        lons = [position[0] for position in ring]
        lats = [position[1] for position in ring]
        properties = feature["properties"]

        assert min(lons) <= properties["centroid_lon"] <= max(lons)
        assert min(lats) <= properties["centroid_lat"] <= max(lats)


def test_accuracy_is_declared(settings: Settings) -> None:
    """The polygons are schematic. Saying so travels with the data (ETH-1)."""
    metadata = geo_service.load_districts(settings)["metadata"]

    assert "approximation" in metadata["accuracy"].lower()
    assert metadata["crs"] == "EPSG:4326"


def test_missing_boundary_file_is_a_404_not_a_crash(tmp_path) -> None:
    missing = Settings(_env_file=None, data_raw_dir=str(tmp_path / "nowhere"))

    with pytest.raises(DatasetNotFoundError) as caught:
        geo_service.load_districts(missing)

    assert caught.value.status_code == 404


def test_malformed_boundary_file_is_a_404_not_a_crash(tmp_path) -> None:
    (tmp_path / "districts.geojson").write_text("{ not json", encoding="utf-8")
    broken = Settings(_env_file=None, data_raw_dir=str(tmp_path))

    with pytest.raises(DatasetNotFoundError):
        geo_service.load_districts(broken)


def test_centroid_lookup_covers_every_district(settings: Settings) -> None:
    centroids = geo_service.district_centroids(settings)

    assert len(centroids) == EXPECTED_DISTRICTS
    assert all(isinstance(lat, float) and isinstance(lon, float)
               for lat, lon in centroids.values())


# --- Route -------------------------------------------------------------------


def test_districts_endpoint_returns_the_collection(
    client: TestClient, settings: Settings
) -> None:
    response = client.get(f"{settings.api_v1_prefix}/data/districts")

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == EXPECTED_DISTRICTS


def test_districts_response_is_schema_validated(
    client: TestClient, settings: Settings
) -> None:
    """SEC-1: the response is a Pydantic model, so its shape is guaranteed."""
    feature = client.get(f"{settings.api_v1_prefix}/data/districts").json()["features"][0]

    assert set(feature) == {"type", "id", "properties", "geometry"}
    assert set(feature["properties"]) == {
        "district_id",
        "name",
        "city",
        "state",
        "country",
        "centroid_lat",
        "centroid_lon",
    }


def test_districts_endpoint_is_documented(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/data/districts" in paths
