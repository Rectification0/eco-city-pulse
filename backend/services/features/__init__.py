"""Feature Engineering — Phase 5 (specs §6.1, design §8).

design §8 asks for "a single deterministic transformer so training and
inference build features identically". The package is organised around that
sentence:

- ``spec``        — *what* the features are: a frozen, serialisable contract
- ``temporal``    — calendar features from the timestamp alone (5.1)
- ``windows``     — lags and rolling statistics on an hourly grid (5.2, 5.3)
- ``transforms``  — the log transform for skewed pollutants (5.4)
- ``transformer`` — the one object that applies all of it (5.5)
- ``store``       — the engineered frame, its spec and its manifest (5.6)
- ``service``     — the training and inference entry points

Two invariants hold throughout, and the tests assert both:

**Nothing looks forward.** A feature at *t* depends only on that station's
observations up to *t*. That is what makes the time-aware split of Phase 7
meaningful; a single forward-looking window would make every metric after it
optimistic (AC-8).

**Nothing is fitted at transform time.** The only data-dependent choice -- which
pollutants to log -- is made once, during ``fit``, and frozen into the spec that
ships with the model.
"""

from services.features import (
    service,
    spec,
    store,
    temporal,
    transformer,
    transforms,
    windows,
)
from services.features.service import (
    FeatureBuildResult,
    build,
    build_features,
    features_at,
    prepare,
)
from services.features.spec import (
    DEFAULT_SPEC,
    SEASONS,
    TEMPORAL_FEATURES,
    FeatureSpec,
    LagSpec,
    RollingSpec,
)
from services.features.store import StoredFeatures
from services.features.transformer import (
    FeatureTransformer,
    complete_mask,
    feature_matrix,
)

__all__ = [
    "DEFAULT_SPEC",
    "SEASONS",
    "TEMPORAL_FEATURES",
    "FeatureBuildResult",
    "FeatureSpec",
    "FeatureTransformer",
    "LagSpec",
    "RollingSpec",
    "StoredFeatures",
    "build",
    "build_features",
    "complete_mask",
    "feature_matrix",
    "features_at",
    "prepare",
    "service",
    "spec",
    "store",
    "temporal",
    "transformer",
    "transforms",
    "windows",
]
