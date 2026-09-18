"""CSV / JSON upload adapter (task 2.3, FEAT-01, SEC-1).

The Upload mode of design §6.3: an analyst hands the platform a file instead of
an API. The file is the least trustworthy input the system has — hand-edited,
exported from a spreadsheet, or downloaded from a portal with its own column
names — so this is where validation earns its keep.

Design choices:

* **Column aliases, not a rigid header.** ``pm2.5``, ``pm2_5`` and ``pm25`` all
  mean the same thing, as do ``lon``/``lng``/``longitude``. Rejecting a file
  over a header spelling would push people into editing data by hand before
  uploading it, which is how data gets corrupted.
* **Per-row failure, not per-file.** One bad row quarantines that row; the rest
  of the file still loads. A single malformed line in a 50,000-row export is
  not a reason to reject the export.
* **Explicit timezone for naive timestamps.** A CSV rarely carries an offset.
  Rather than guess, the caller states the assumption and it is recorded in the
  ingestion log (DR-2).
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from core.config import Settings
from core.exceptions import UploadRejectedError
from services.adapters.base import AdapterSpec, SourceAdapter, SourceDomain
from services.geo_service import Station
from services.harmonizer import SourceReading, harmonize_coordinates, to_utc

UPLOAD_SOURCE_NAME = "Analyst Upload"

# Values a spreadsheet writes for "nothing". Treated as absent rather than as a
# parse failure -- an empty cell is missingness, which Phase 3 is built for.
NULL_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "-", "--"})


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class UploadRecord(BaseModel):
    """One row of an uploaded file (SEC-1).

    ``timestamp``/``lat``/``lon`` stay loosely typed here on purpose: the
    harmonizer owns the conversion rules for both, and duplicating them in a
    Pydantic coercion would give two places where DR-2/DR-3 could disagree.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True, str_strip_whitespace=True)

    timestamp: datetime | str | float = Field(
        validation_alias=AliasChoices(
            "timestamp", "time", "datetime", "date_time", "observed_at", "ts"
        )
    )
    lat: float | str = Field(validation_alias=AliasChoices("lat", "latitude"))
    lon: float | str = Field(
        validation_alias=AliasChoices("lon", "lng", "long", "longitude")
    )

    pm25: float | None = Field(
        default=None, validation_alias=AliasChoices("pm25", "pm2_5", "pm2.5", "PM2.5")
    )
    pm10: float | None = Field(
        default=None, validation_alias=AliasChoices("pm10", "pm_10", "PM10")
    )
    temp: float | None = Field(
        default=None,
        validation_alias=AliasChoices("temp", "temperature", "temp_c", "t"),
    )
    humidity: float | None = Field(
        default=None,
        ge=0,
        le=100,
        validation_alias=AliasChoices("humidity", "relative_humidity", "rh", "h"),
    )
    traffic_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        validation_alias=AliasChoices(
            "traffic_score", "traffic", "congestion", "congestion_index"
        ),
    )

    @field_validator("*", mode="before")
    @classmethod
    def _normalize_blanks(cls, value: Any) -> Any:
        return _blank_to_none(value)


def parse_upload(content: bytes, *, filename: str | None = None) -> list[dict[str, Any]]:
    """Turn an uploaded file into raw row dicts, without validating them.

    Format is decided by content, not by the file extension: a browser will
    happily label a JSON file ``.csv``, and trusting the label would produce a
    confusing "every row is malformed" instead of a clear parse error.
    """
    if not content.strip():
        raise UploadRejectedError("The uploaded file is empty.")

    try:
        text = content.decode("utf-8-sig")  # -sig strips Excel's BOM
    except UnicodeDecodeError as exc:
        raise UploadRejectedError(
            "The uploaded file is not UTF-8 text.",
            details={"filename": filename, "error": str(exc)},
        ) from exc

    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        return _parse_json(stripped, filename=filename)
    return _parse_csv(text, filename=filename)


def _parse_json(text: str, *, filename: str | None) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UploadRejectedError(
            "The uploaded file is not valid JSON.",
            details={"filename": filename, "error": str(exc)},
        ) from exc

    # Accept a bare list, or an object wrapping one under a conventional key.
    if isinstance(payload, dict):
        for key in ("records", "data", "observations", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]

    if not isinstance(payload, list):
        raise UploadRejectedError(
            "JSON uploads must be a list of records, or an object containing one.",
            details={"filename": filename},
        )

    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        raise UploadRejectedError(
            "The JSON upload contains no record objects.", details={"filename": filename}
        )
    return rows


def _parse_csv(text: str, *, filename: str | None) -> list[dict[str, Any]]:
    sample = text[:4096]
    try:
        dialect: Any = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel  # a single-column file gives the sniffer nothing

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise UploadRejectedError(
            "The CSV upload has no header row.", details={"filename": filename}
        )

    # Normalise header spelling once, so the alias table does not need a
    # variant for every combination of case and whitespace.
    reader.fieldnames = [
        (name or "").strip().lstrip("﻿").lower() for name in reader.fieldnames
    ]

    rows = [{k: v for k, v in row.items() if k} for row in reader]
    if not rows:
        raise UploadRejectedError(
            "The CSV upload contains a header but no rows.",
            details={"filename": filename},
        )
    return rows


class UploadAdapter(SourceAdapter):
    """Wraps already-parsed rows so uploads share the one ingestion path.

    Constructed per request with the file's rows; ``fetch`` simply replays them.
    """

    spec = AdapterSpec(
        name=UPLOAD_SOURCE_NAME,
        domain=SourceDomain.BUNDLE,
        api_url=None,
        requires_credentials=False,
        description="CSV or JSON supplied by an analyst.",
    )

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        assume_timezone: timezone = timezone.utc,
        filename: str | None = None,
    ) -> None:
        self.rows = list(rows)
        self.assume_timezone = assume_timezone
        self.filename = filename

    def fetch(
        self, settings: Settings, *, stations: Sequence[Station]
    ) -> Iterator[Mapping[str, Any]]:
        # stations is irrelevant here: the file carries its own coordinates.
        yield from self.rows

    def parse(self, payload: Mapping[str, Any]) -> list[SourceReading]:
        record = UploadRecord.model_validate(payload)
        lat, lon = harmonize_coordinates(record.lat, record.lon)

        return [
            SourceReading(
                timestamp=to_utc(record.timestamp, assume_timezone=self.assume_timezone),
                lat=lat,
                lon=lon,
                pm25=record.pm25,
                pm10=record.pm10,
                temp=record.temp,
                humidity=record.humidity,
                traffic_score=record.traffic_score,
            )
        ]


__all__ = [
    "NULL_TOKENS",
    "UPLOAD_SOURCE_NAME",
    "UploadAdapter",
    "UploadRecord",
    "parse_upload",
]
