"""EDA & Statistical Engine — Phase 4, BACSE301 Modules 3-5 (FEAT-02).

design §5 names a single ``eda_service.py``. It is realised as a package for
the same reason Phase 3's quality engine is: the phase covers four separable
concerns, each with its own statistical reasoning worth reading on its own.

- ``profile``       — univariate, bivariate and distribution analysis (4.1, 4.2, 4.4)
- ``decomposition`` — STL trend / seasonal / residual (4.5)
- ``cache``         — results keyed by a dataset fingerprint (4.6)
- ``report``        — self-contained HTML report (4.7)

``service`` is the entry point the routes use: it loads the slice, consults the
cache, and returns the profile.

**STL needs statsmodels**, which is imported lazily by ``decomposition`` so
that a machine where the compiled extension cannot load still gets the rest of
the engine rather than an import error at startup.
"""

from services.eda import cache, profile, report, service
from services.eda.cache import PROFILE_CACHE, DatasetVersion, dataset_version
from services.eda.profile import (
    BivariateProfile,
    CorrelationPair,
    DistributionAssessment,
    StatisticalProfile,
    UnivariateStats,
)
from services.eda.service import build_profile, decompose_series, generate_report

__all__ = [
    "PROFILE_CACHE",
    "BivariateProfile",
    "CorrelationPair",
    "DatasetVersion",
    "DistributionAssessment",
    "StatisticalProfile",
    "UnivariateStats",
    "build_profile",
    "cache",
    "dataset_version",
    "decompose_series",
    "generate_report",
    "profile",
    "report",
    "service",
]
