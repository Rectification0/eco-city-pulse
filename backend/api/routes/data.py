"""Data-layer routes.

Phase 1 exposes the district boundaries (task 1.8). ``GET /data/sources`` lands
with the ingestion engine that gives it something truthful to report (task 2.10).
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.dependencies import SettingsDep
from services import geo_service

router = APIRouter(prefix="/data", tags=["data"])

Position = Annotated[
    tuple[float, float],
    Field(description="[longitude, latitude] in decimal degrees (DR-3)."),
]


class DistrictProperties(BaseModel):
    """SEC-1: the boundary file is validated on the way out, not trusted."""

    district_id: str
    name: str
    city: str
    state: str
    country: str
    centroid_lat: float = Field(ge=-90, le=90)
    centroid_lon: float = Field(ge=-180, le=180)


class DistrictGeometry(BaseModel):
    type: Literal["Polygon"]
    # GeoJSON nests rings: [exterior, *holes], each a list of positions.
    coordinates: list[list[Position]]


class DistrictFeature(BaseModel):
    type: Literal["Feature"]
    id: str
    properties: DistrictProperties
    geometry: DistrictGeometry


class DistrictMetadata(BaseModel):
    """Provenance travels with the data.

    ``accuracy`` reaches the UI deliberately: the polygons are schematic, and a
    map that silently implies surveyed boundaries would be the same kind of
    overclaim the ethics requirement rules out elsewhere (ETH-1).
    """

    city: str
    crs: str
    units: str
    district_count: int
    bbox: tuple[float, float, float, float]
    accuracy: str
    license: str


class DistrictCollection(BaseModel):
    type: Literal["FeatureCollection"]
    name: str
    metadata: DistrictMetadata
    features: list[DistrictFeature]


@router.get(
    "/districts",
    response_model=DistrictCollection,
    summary="City district boundaries (GeoJSON)",
)
async def get_districts(settings: SettingsDep) -> DistrictCollection:
    """District polygons for the dashboard map layer (specs §5.1, task 10.4)."""
    return DistrictCollection.model_validate(geo_service.load_districts(settings))
