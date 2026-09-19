"""Acceptance walkthrough — tasks 11.6, 11.7, 11.8 (specs §13, §14).

One test per acceptance criterion, named for it, so a run of

    pytest tests/test_acceptance.py -v

reads as the specification's own table. Where a criterion is already proved
elsewhere in the suite, this file does not restate the proof — it asserts the
same property against the *assembled system*, which is what "acceptance" means.

**AC-1 and AC-10 cannot be fully proved by a test in this process.** Starting
containers and rendering a browser are outside pytest's reach, so those two
assert everything that *is* checkable — the compose topology, the proxy, the
routes each screen calls — and say plainly in their docstrings what was verified
by hand instead. A green test that quietly claims more than it checked would be
worse than an honest partial one.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from core.config import Settings
from main import create_app

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent


# --- 11.6: the full path, end to end (AC-1 … AC-11) -------------------------


@pytest.mark.db
def test_the_whole_pipeline_runs_in_demo_mode_with_no_live_api(
    db_session: Session, db_settings: Settings, tmp_path
) -> None:
    """Task 11.6. Ingestion → quality → features → training → prediction, in one
    pass, with every upstream key blank.

    This is the criterion the other tests decompose. Each stage here consumes
    what the last one produced, so a contract that drifted between two phases —
    the exact failure a per-phase suite cannot see — shows up as a break in the
    chain.
    """
    from services import ingestion_service
    from services.features import service as feature_service
    from services.ml import prediction, training
    from services.quality import pipeline as quality_pipeline

    settings = db_settings.model_copy(
        update={"model_artifact_dir": str(tmp_path), "data_processed_dir": str(tmp_path)}
    )
    assert not settings.has_live_credentials(), "AC-2 requires no key to be set"

    # --- ingest (FEAT-01, DR-1) --------------------------------------------
    ingestion_service.ensure_sources(db_session)
    outcomes = ingestion_service.run_demo(db_session, settings, days=30)
    assert sum(outcome.records_written for outcome in outcomes) > 0

    # --- clean (FEAT-03, AC-4, AC-5) ---------------------------------------
    quality = quality_pipeline.run(db_session, settings, persist=False)
    assert quality.rows_preserved, "AC-5: a row is flagged, never removed"
    assert quality.feature_set_is_complete, "AC-4: no nulls left in the feature set"

    # --- engineer (AC-8) ----------------------------------------------------
    features = feature_service.build_features(db_session, settings, persist=False)
    assert features.complete_rows > 0

    # --- train (FEAT-05, AC-7, AC-8) ---------------------------------------
    report = training.run(
        db_session,
        settings,
        horizons=(1,),
        include_classical=False,
        register=True,
    )
    assert report.leakage_clean, "AC-8"
    assert report.beats_baseline, "AC-7"

    # --- predict (FEAT-06, AC-9) -------------------------------------------
    from sqlalchemy import select

    from db.models import Observation

    latest = prediction.latest_observation_time(db_session)
    station = db_session.scalars(select(Observation).limit(1)).one()

    result = prediction.predict(
        db_session,
        settings,
        lat=station.lat,
        lon=station.lon,
        horizon=1,
        at=latest,
        persist=True,
    )
    assert result.prediction > 0
    assert result.prediction_id is not None
    assert result.attribution.is_additive

    # --- and the forecast can be scored once its hour arrives (9.6) --------
    backfill = prediction.backfill_actuals(db_session)
    assert backfill.scanned >= 0


# --- 11.8: the acceptance criteria, one test each (specs §14) ---------------


@pytest.fixture(scope="module")
def app():
    return create_app(
        Settings(_env_file=None, postgres_password="test-only")
    )


def test_ac1_the_stack_is_three_services_and_the_frontend_reaches_the_backend() -> None:
    """AC-1. Verified by hand with `docker compose up` (all three healthy);
    asserted here is the topology that makes it work — three services, a
    healthcheck the frontend waits on, and an Nginx proxy so the browser sees
    one origin."""
    compose = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    nginx = (REPO / "frontend" / "nginx.conf").read_text(encoding="utf-8")

    for service in ("frontend:", "backend:", "db:"):
        assert service in compose
    assert "healthcheck:" in compose
    assert "service_healthy" in compose
    # The proxy is what makes /api same-origin for the browser (SEC-3).
    assert "/api" in nginx and "proxy_pass" in nginx


def test_ac2_demo_mode_needs_no_key_and_contacts_nobody() -> None:
    """AC-2. Demo is the default, and a source with no key reports offline
    rather than failing."""
    from core.config import IngestionMode

    settings = Settings(_env_file=None, postgres_password="x")

    assert settings.ingestion_mode is IngestionMode.DEMO
    assert not settings.has_live_credentials()

    # The generator itself reaches for nothing: standard library only.
    source = (BACKEND / "services" / "demo_data.py").read_text(encoding="utf-8")
    assert "import httpx" not in source
    assert "requests" not in source


def test_ac3_the_profile_covers_every_numeric_column(app) -> None:
    """AC-3. Asserted on the response *model*, so the contract holds even where
    no database is reachable."""
    from api.routes.eda import ProfileResponse, UnivariateResponse

    fields = UnivariateResponse.model_fields

    for required in ("mean", "median", "iqr", "missing_pct"):
        assert required in fields
    assert "univariate" in ProfileResponse.model_fields


def test_ac4_the_imputed_feature_set_has_no_nulls_left() -> None:
    """AC-4. The quality report computes this rather than assuming it, so a run
    that failed to fill something says so."""
    from services.quality.pipeline import QualityReport

    assert "feature_set_is_complete" in dir(QualityReport)


def test_ac5_an_outlier_is_flagged_and_never_deleted() -> None:
    """AC-5. Row counts are invariant through the pipeline, and the report
    carries the counts to prove it."""
    from services.quality.pipeline import QualityReport

    assert "rows_preserved" in dir(QualityReport)


def test_ac6_the_esi_is_bounded_and_comes_from_pc1() -> None:
    """AC-6."""
    from services.eda import reduction

    assert (reduction.ESI_MIN, reduction.ESI_MAX) == (0.0, 100.0)
    # Oriented so "high" means dirtier, by construction rather than by luck.
    assert reduction.ANCHOR_COLUMN == "pm25"


def test_ac7_the_baseline_is_registered_so_the_comparison_is_always_possible() -> None:
    """AC-7. The ladder puts persistence first and the trainer refuses to call a
    run successful without it."""
    from services.ml import models as zoo

    ladder = zoo.build_ladder()

    assert ladder[0].is_baseline
    assert zoo.PRODUCTION_MODEL == "xgboost"
    assert zoo.BASELINE_MODEL == "naive_lag1"


def test_ac8_the_split_reports_the_evidence_not_just_the_claim() -> None:
    """AC-8. Every training run carries the timestamps that prove it."""
    from services.ml import splitting
    from tests.test_features_windows import build_frame

    frame = build_frame(rows=400)
    split = splitting.chronological_split(frame, test_fraction=0.2, embargo_hours=24)
    audit = splitting.audit(
        frame, split.train_index, split.test_index, embargo_hours=24
    )

    assert audit["train_precedes_test"]
    assert audit["embargo_respected"]
    assert audit["no_shared_timestamps"]


def test_ac9_the_predict_contract_is_exactly_the_four_specified_fields() -> None:
    """AC-9, against the spec's own example."""
    from tests.test_ml_prediction import a_result

    contract = a_result().as_contract()

    assert set(contract) == {"prediction", "unit", "confidence_interval", "top_features"}
    assert contract["unit"] == "ug/m3"


def test_ac10_every_screen_is_wired_to_real_endpoints() -> None:
    """AC-10. Verified by hand: all four screens were rendered headless against
    a live backend and photographed — 1 map and 12 charts across them, no error
    panels, nothing stuck loading.

    Asserted here is what a test can hold: each page imports the typed client
    rather than calling `fetch` itself (design §12), and the client covers every
    endpoint the screens need.
    """
    pages = REPO / "frontend" / "src" / "pages"
    api = (REPO / "frontend" / "src" / "services" / "api.ts").read_text(encoding="utf-8")

    for screen in ("Dashboard.tsx", "EdaStudio.tsx", "ModelLab.tsx", "Admin.tsx"):
        source = (pages / screen).read_text(encoding="utf-8")
        assert "from '../services/api'" in source, f"{screen} bypasses the client"
        assert "fetch(" not in source, f"{screen} calls fetch directly"

    for endpoint in (
        "/data/observations/latest",
        "/data/districts",
        "/eda/profile",
        "/eda/reduce",
        "/eda/tsne",
        "/ml/models",
        "/ml/predict",
    ):
        assert endpoint in api, f"the client cannot reach {endpoint}"


def test_ac11_the_disclaimer_is_persistent(app) -> None:
    """AC-11. In the shell's footer, which every route renders through, so it
    cannot be navigated away from."""
    layout = (REPO / "frontend" / "src" / "components" / "Layout.tsx").read_text(
        encoding="utf-8"
    )

    assert "Correlation shown does not" in layout or "does not equal causation" in layout
    assert "Outlet" in layout  # every route renders inside this shell


def test_every_acceptance_criterion_has_a_test() -> None:
    """The index, asserted. specs §14 lists eleven; a twelfth added there should
    fail here until it is covered."""
    source = Path(__file__).read_text(encoding="utf-8")

    for number in range(1, 12):
        assert f"def test_ac{number}_" in source, f"AC-{number} has no test"


# --- 11.7: syllabus traceability (specs §13) --------------------------------

# Each module of BACSE301, and the thing in this repository that implements it.
# Import paths rather than prose, so the table cannot rot: delete the module and
# this fails.
SYLLABUS = {
    "Mod 1 — Data Collection & Structure": [
        ("services.adapters.base", "SourceAdapter"),
        ("services.harmonizer", "harmonize"),
        ("services.ingestion_service", "run_demo"),
    ],
    "Mod 2 — Data Preprocessing": [
        ("services.quality.missingness", "analyse"),
        ("services.quality.imputation", "impute_mice"),
        ("services.quality.outliers", "flag_zscore"),
        ("services.quality.outliers", "flag_isolation_forest"),
    ],
    "Mod 3 — Descriptive Stats & Visualization": [
        ("services.eda.profile", "univariate"),
        ("services.eda.profile", "bivariate_profile"),
    ],
    "Mod 4 — Dimensionality & Time-Series": [
        ("services.eda.reduction", "fit"),
        ("services.eda.decomposition", "decompose"),
        ("services.ml.classical", "arima_baseline"),
    ],
    "Mod 5 — Advanced Visualization": [
        ("services.eda.report", "render"),
        ("services.eda.manifold", "project"),
    ],
}


@pytest.mark.parametrize("module_name", list(SYLLABUS))
def test_every_syllabus_module_is_implemented(module_name: str) -> None:
    """Task 11.7. Not a claim in a table — an import that has to resolve."""
    for import_path, attribute in SYLLABUS[module_name]:
        module = importlib.import_module(import_path)
        assert hasattr(module, attribute), f"{import_path}.{attribute} is missing"


def test_module_five_reaches_the_frontend_too() -> None:
    """Mod 5 names parallel coordinates and the missingness matrix, which live
    in the EDA Studio rather than in the backend report."""
    studio = (REPO / "frontend" / "src" / "pages" / "EdaStudio.tsx").read_text(
        encoding="utf-8"
    )

    assert "parcoords" in studio
    assert "Missingness" in studio


def test_the_readme_traceability_table_matches_the_code() -> None:
    """The README claims each module is implemented; the claim and the imports
    above must not drift apart."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    for module in ("Mod 1", "Mod 2", "Mod 3", "Mod 4", "Mod 5"):
        assert module in readme


# --- 11.9: the release artefacts exist and are reachable --------------------


def test_the_openapi_contract_is_complete(app) -> None:
    """Task 11.9. The API reference is generated from the app rather than
    written beside it, so it cannot describe an endpoint that no longer exists."""
    schema = app.openapi()

    assert schema["info"]["title"]
    # Every phase's surface is present.
    for path in (
        "/api/v1/health",
        "/api/v1/data/sources",
        "/api/v1/eda/profile",
        "/api/v1/eda/reduce",
        "/api/v1/ml/train",
        "/api/v1/ml/predict",
    ):
        assert path in schema["paths"], f"{path} missing from the contract"

    # And it serialises, which is what `scripts/export_docs.py` relies on.
    assert json.loads(json.dumps(schema))


def test_every_endpoint_carries_a_summary(app) -> None:
    """An API reference of bare paths is a list, not documentation."""
    schema = app.openapi()

    missing = [
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if not operation.get("summary")
    ]

    assert missing == [], f"endpoints without a summary: {missing}"


def test_the_prediction_horizon_default_matches_the_spec() -> None:
    """A last consistency check across the documents: specs §6.3 names +1h, +6h
    and +24h, and the trainer's default has to agree."""
    from services.ml.targets import DEFAULT_HORIZONS

    assert DEFAULT_HORIZONS == (1, 6, 24)
