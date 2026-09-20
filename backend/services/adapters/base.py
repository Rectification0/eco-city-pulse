"""Source adapter interface (task 2.1).

design §6.3: the four ingestion modes "share one code path, differing only in
the source adapter". This module defines that seam. An adapter's entire job is:

1. say whether it is configured at all,
2. **fetch** raw payloads from wherever it gets them, and
3. **parse** one payload into ``SourceReading`` objects, applying its own
   Pydantic schema and the DR-2/DR-3 conversions.

Everything after that -- resampling, quarantine, the write, the log -- belongs
to ``ingestion_service`` and is identical for every source. That is what makes
demo mode a genuine exercise of the live path rather than a parallel one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.config import Settings
from services.geo_service import Station
from services.harmonizer import SourceReading


class SourceDomain(str, Enum):
    """What a source measures. Drives nothing but the admin view's grouping."""

    AIR_QUALITY = "air_quality"
    WEATHER = "weather"
    TRAFFIC = "traffic"
    BUNDLE = "bundle"


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """The static description of a source, registered into ``data_sources``."""

    name: str
    domain: SourceDomain
    api_url: str | None
    # False for the demo bundle and the synthetic fallback: they are always
    # available, so "no API key" is not a reason to consider them offline.
    requires_credentials: bool = True
    # True for the demo bundle and the traffic fallback: their values are
    # modelled, not measured. Reflected into ``data_sources.is_synthetic`` so
    # the analytics scope can keep the two provenances apart without holding a
    # list of source names (see ``datasets.resolve_source_ids``).
    synthetic: bool = False
    # The measurement columns this source is the authority on. Once several
    # sources merge into one row, `temp` and `humidity` arrive from two of them
    # at once and something has to decide. Without this the decision was made
    # by tuple order in LIVE_ADAPTER_TYPES, which is deterministic but
    # accidental -- and it made the merged value depend on ingestion order.
    #
    # An authoritative writer overwrites the field; a non-authoritative one may
    # only fill it when it is still NULL. That pair of rules is what makes the
    # merge order-independent: whichever of the two runs first, the authority's
    # value is what survives.
    authoritative_for: frozenset[str] = frozenset()
    description: str = ""


class SourceAdapter(ABC):
    """Base class for every source. See the module docstring for the contract."""

    spec: AdapterSpec

    @property
    def name(self) -> str:
        return self.spec.name

    def is_configured(self, settings: Settings) -> bool:
        """Whether a fetch is worth attempting at all (DR-1).

        A source with no credential is not broken, it is simply not switched
        on; ingestion records it as ``skipped`` rather than as a failure.
        """
        if not self.spec.requires_credentials:
            return True
        return bool(self.credential(settings))

    def credential(self, settings: Settings) -> str:
        """The API key for this source, or an empty string. Never logged."""
        return ""

    @abstractmethod
    def fetch(
        self, settings: Settings, *, stations: Sequence[Station]
    ) -> Iterator[Mapping[str, Any]]:
        """Yield raw payloads exactly as the provider returned them.

        Raw on purpose: whatever is yielded here is what gets quarantined if
        parsing fails, so the stored evidence is the provider's own bytes and
        not something this code already reshaped.
        """

    @abstractmethod
    def parse(self, payload: Mapping[str, Any]) -> list[SourceReading]:
        """Validate one payload and convert it to readings.

        Raises ``SchemaValidationError``/``HarmonizationError`` (or a Pydantic
        ``ValidationError``) for a record that cannot be used; the caller
        quarantines it and carries on with the rest.
        """


__all__ = ["AdapterSpec", "SourceAdapter", "SourceDomain", "Station"]
