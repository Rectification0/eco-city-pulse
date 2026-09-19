"""The single deterministic transformer (task 5.5, design §8).

design §8 in one sentence: "Computed by a single deterministic transformer so
training and inference build features identically." This module is that
transformer, and the word carrying the weight is *single* -- not "two functions
that agree", not "the same steps in the same order". One object, fitted once,
serialised, and replayed.

**What fitting learns.** Exactly one thing: which pollutants are skewed enough
to deserve a log transform (task 5.4). Everything else -- the lags, the
windows, the calendar arithmetic, the column order -- is fixed by the
``FeatureSpec`` before any data is seen. Keeping the fitted surface this small
is the point: the smaller it is, the less there is to drift, and the fitted part
is written into the spec so it travels with the model artefact.

**Why no imputation lives here.** MICE (task 3.2) is fitted across a whole
slice, so the value it invents for a given hour depends on which rows were
loaded alongside it. Run it in the training path over a year and in the serving
path over a two-day window and the same hour gets two different numbers -- a
silent violation of the guarantee this module exists to make. So the feature
path uses only window-local repairs (``services.features.service.prepare``),
and a gap too long to repair locally surfaces as NaN rather than as a number
that would not reproduce. The quality engine's MICE remains where it belongs:
preparing the analysis frame of Phase 3.

**The guarantee, stated precisely.** Every feature at time *t* is a function of
that station's observations in ``[t - history_hours, t]`` and of *t* itself.
Nothing else. Therefore a transform over a long training frame and a transform
over the trailing window of a serving request produce the same row for *t* --
which is the exit criterion of this phase, and is asserted directly in
``tests/test_features_transformer.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from services.datasets import STATION_COLUMN, station_key
from services.features import temporal, transforms, windows
from services.features.spec import (
    DEFAULT_SPEC,
    POLLUTANT_COLUMNS,
    FeatureSpec,
)

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp",)


@dataclass(frozen=True, slots=True)
class FitWindow:
    """The slice a transformer was fitted on.

    Recorded because the log decision is the one data-dependent choice in the
    phase, and AC-8 turns on it having been made from training data alone. A
    reviewer can check that from the stored spec without rerunning anything.
    """

    rows: int
    stations: int
    start: datetime | None
    end: datetime | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "stations": self.stations,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FitWindow:
        return cls(
            rows=int(payload.get("rows", 0)),
            stations=int(payload.get("stations", 0)),
            start=_parse(payload.get("start")),
            end=_parse(payload.get("end")),
        )


def _parse(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True, slots=True)
class FeatureTransformer:
    """Builds the feature set from observations. Deterministic by construction."""

    spec: FeatureSpec = DEFAULT_SPEC
    fitted_at: datetime | None = None
    fit_window: FitWindow | None = None
    # Skew before and after log1p, per candidate pollutant, plus the sentence
    # explaining the verdict. Kept so the stored spec says why, not just what.
    skew_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- Fitting ------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        base_spec: FeatureSpec = DEFAULT_SPEC,
        log_candidates: tuple[str, ...] = POLLUTANT_COLUMNS,
    ) -> FeatureTransformer:
        """Choose the log columns from this frame and freeze the spec (AC-8).

        Call this on the **training** slice only. Fitting on the full dataset
        would let the test period's distribution influence a decision applied to
        the training period -- a mild leak, but the same class of mistake as
        fitting a scaler on everything, and the pipeline rejects that one too.
        """
        chosen, evidence = transforms.choose_log_columns(frame, log_candidates)

        return cls(
            spec=base_spec.with_log_columns(chosen),
            fitted_at=datetime.now(timezone.utc),
            fit_window=_describe(frame),
            skew_evidence=evidence,
        )

    @property
    def is_fitted(self) -> bool:
        return self.fitted_at is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.spec.feature_names

    @property
    def fingerprint(self) -> str:
        """A short hash of the spec.

        Written next to a model artefact in Phase 7 so serving can assert that
        the features it is about to build are the ones the model was trained on,
        rather than trusting that nobody edited the spec in between.
        """
        payload = json.dumps(self.spec.as_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # --- Transforming -------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build every feature for every row (tasks 5.1-5.4).

        The input rows come back unchanged and in their original order, with the
        feature columns appended. Rows whose window is not yet covered -- the
        first 24 hours of a station's series, or a row sitting after a long gap
        -- carry NaN.
        """
        _validate(frame)
        frame = _with_station(frame)

        if frame.index.has_duplicates:
            raise ValueError(
                "feature construction needs a unique row index; "
                "call DataFrame.reset_index() first."
            )

        original = frame.index
        # Windowed features assume chronological order within a station. The
        # loaded frame already is (datasets.prepare_frame sorts it), but a
        # hand-built one need not be, and an unsorted shift is wrong rather than
        # merely untidy.
        ordered = frame.sort_values([STATION_COLUMN, "timestamp"])

        if self.spec.temporal:
            ordered = temporal.add_temporal(
                ordered, offset_minutes=self.spec.local_offset_minutes
            )

        ordered = windows.add_window_features(
            ordered,
            lags=self.spec.lags,
            rollings=self.spec.rollings,
            min_window_coverage=self.spec.min_window_coverage,
            exact_windows=self.spec.exact_windows,
        )

        ordered = transforms.add_log_features(ordered, self.spec.log_columns)

        return ordered.loc[original]

    def fit_transform(self, frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Convenience for the training path only -- never for serving."""
        return self.fit(frame, **kwargs).transform(frame)

    # --- Serialisation ------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "fingerprint": self.fingerprint,
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
            "fit_window": self.fit_window.as_dict() if self.fit_window else None,
            "skew_evidence": self.skew_evidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureTransformer:
        window = payload.get("fit_window")
        return cls(
            spec=FeatureSpec.from_dict(payload["spec"]),
            fitted_at=_parse(payload.get("fitted_at")),
            fit_window=FitWindow.from_dict(window) if window else None,
            skew_evidence=dict(payload.get("skew_evidence", {})),
        )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> FeatureTransformer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --- Helpers ----------------------------------------------------------------


def _validate(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")


def _with_station(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the grouping key exists.

    Derived from the coordinates the same way ``datasets`` derives it, so a
    frame built by hand for a test or by an inference fetcher gets the identical
    key without having to remember the format.
    """
    if STATION_COLUMN in frame.columns:
        return frame

    frame = frame.copy()
    if {"lat", "lon"} <= set(frame.columns):
        frame[STATION_COLUMN] = [
            station_key(lat, lon)
            for lat, lon in zip(frame["lat"], frame["lon"], strict=True)
        ]
        return frame

    raise ValueError(
        f"frame needs a {STATION_COLUMN!r} column, or lat/lon to derive one from."
    )


def _describe(frame: pd.DataFrame) -> FitWindow:
    if frame.empty:
        return FitWindow(rows=0, stations=0, start=None, end=None)

    stations = (
        int(frame[STATION_COLUMN].nunique()) if STATION_COLUMN in frame.columns else 1
    )
    return FitWindow(
        rows=len(frame),
        stations=stations,
        start=frame["timestamp"].min().to_pydatetime(),
        end=frame["timestamp"].max().to_pydatetime(),
    )


def complete_mask(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """Rows whose every feature is present -- the modellable subset."""
    names = [name for name in spec.feature_names if name in frame.columns]
    if not names:
        return pd.Series(True, index=frame.index)
    return frame[names].notna().all(axis=1)


def feature_matrix(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """The feature columns alone, in the spec's order.

    What Phase 7 hands to a model. Column order comes from the spec rather than
    from the frame, so a transform that happened to append in another order
    still produces the matrix the model was fitted on.
    """
    missing = [name for name in spec.feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing engineered columns: {missing}")
    return frame[list(spec.feature_names)]


__all__ = [
    "FeatureTransformer",
    "FitWindow",
    "complete_mask",
    "feature_matrix",
]
