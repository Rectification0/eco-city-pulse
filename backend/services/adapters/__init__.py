"""Source adapter registry (task 2.1).

One place that knows which sources exist, so ``ingestion_service`` can ask for
"the live ones" or "the one called X" without importing every module itself.

``registered_adapters`` is what gets reflected into ``data_sources``: a source
appears in the admin view whether or not it is configured, because "AQICN is
offline for want of a key" is information, and a missing row is not.
"""

from __future__ import annotations

from services.adapters.aqicn import AqicnAdapter
from services.adapters.base import AdapterSpec, SourceAdapter, SourceDomain
from services.adapters.demo import DemoAdapter
from services.adapters.openweather import OpenWeatherAdapter
from services.adapters.synthetic_traffic import SyntheticTrafficAdapter
from services.adapters.tomtom import TomTomAdapter
from services.adapters.upload import UPLOAD_SOURCE_NAME, UploadAdapter, parse_upload

# Sources that a Scheduled or Manual run attempts. Order is the order they are
# ingested in, and it is stable so the run log reads the same way every time.
LIVE_ADAPTER_TYPES: tuple[type[SourceAdapter], ...] = (
    AqicnAdapter,
    OpenWeatherAdapter,
    TomTomAdapter,
)


def live_adapters() -> list[SourceAdapter]:
    return [adapter_type() for adapter_type in LIVE_ADAPTER_TYPES]


def registered_specs() -> list[AdapterSpec]:
    """Every source that should exist as a row in ``data_sources``.

    Includes the upload and demo pseudo-sources: an uploaded file has real
    provenance and belongs in the same health view as an API.
    """
    return [
        AqicnAdapter.spec,
        OpenWeatherAdapter.spec,
        TomTomAdapter.spec,
        SyntheticTrafficAdapter.spec,
        DemoAdapter.spec,
        UploadAdapter.spec,
    ]


__all__ = [
    "LIVE_ADAPTER_TYPES",
    "UPLOAD_SOURCE_NAME",
    "AdapterSpec",
    "AqicnAdapter",
    "DemoAdapter",
    "OpenWeatherAdapter",
    "SourceAdapter",
    "SourceDomain",
    "SyntheticTrafficAdapter",
    "TomTomAdapter",
    "UploadAdapter",
    "live_adapters",
    "parse_upload",
    "registered_specs",
]
