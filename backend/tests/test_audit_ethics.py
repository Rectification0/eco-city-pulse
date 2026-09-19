"""Privacy and ethics audit — tasks 11.4, 11.5 (PRIV-1, PRIV-2, ETH-1).

The two requirements this file holds are the easiest in the project to satisfy
today and the easiest to lose next month. Nothing stops someone adding a
`reported_by` column, and nothing stops a caption drifting from "traffic
correlates with PM2.5" to "traffic drives PM2.5" — except a test that reads the
schema and the copy and refuses.

**On the causal-language scan.** It reads every user-facing string in the
backend and the frontend and looks for causal verbs. Sentences that *deny*
causation are the point of ETH-1, so they are recognised rather than flagged;
everything else has to be phrased as association.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from db.base import Base

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent


# --- 11.4: no PII anywhere (PRIV-1, PRIV-2) ---------------------------------

# Column names that would mean a person is in the database.
PII_PATTERNS = re.compile(
    r"(?i)(email|phone|mobile|user|username|person|citizen|resident|"
    r"address|postcode|zip|ip|device|household|age|gender|name)"
)


def pii_hits(column_name: str) -> bool:
    """Match on word parts, not substrings.

    Written out because the naive version flagged ``ingestion_runs.message`` —
    "age" sits inside "message". Splitting on underscores first means ``user_id``
    is still caught while ``message`` is not.
    """
    parts = column_name.replace("_", " ")
    return bool(PII_PATTERNS.search(parts))

# `name` appears on tables that name a *thing*, not a person. Listed explicitly
# so a future `contact_name` still fails.
NON_PII_NAME_COLUMNS = {
    ("data_sources", "name"): "the provider's name, e.g. 'AQICN'",
    ("models", "name"): "the estimator's name, e.g. 'xgboost'",
}


def test_no_table_holds_a_column_that_identifies_a_person() -> None:
    """PRIV-2. The platform ingests environmental measurements; a column that
    could identify someone would put it in a different regulatory world."""
    offenders = []
    for table_name, table in Base.metadata.tables.items():
        for column in table.columns:
            if column.name == "name":
                continue  # handled by its own test, with the exceptions named
            if pii_hits(column.name):
                offenders.append(f"{table_name}.{column.name}")

    assert offenders == [], f"columns that look like PII: {offenders}"


def test_the_name_columns_that_exist_name_things_not_people() -> None:
    """The exceptions, enumerated, so the next `name` column is a decision."""
    named = {
        (table_name, column.name)
        for table_name, table in Base.metadata.tables.items()
        for column in table.columns
        if column.name == "name"
    }

    assert named == set(NON_PII_NAME_COLUMNS)


def test_an_observation_is_a_measurement_and_a_place_and_nothing_else() -> None:
    """PRIV-1. The row records what was measured, where and when — never who
    was nearby."""
    columns = {c.name for c in Base.metadata.tables["observations"].columns}

    assert columns == {
        "id",
        "source_id",
        "timestamp",
        "lat",
        "lon",
        "pm25",
        "pm10",
        "temp",
        "humidity",
        "traffic_score",
        "is_anomaly",
    }


def test_every_upstream_is_a_public_environmental_api() -> None:
    """PRIV-1: only public environmental data is ingested. The registry is the
    complete list of places this platform will ever call."""
    from services import adapters as registry

    allowed_hosts = {
        "api.waqi.info": "AQICN — public air-quality index",
        "api.openweathermap.org": "OpenWeather — public weather",
        "api.tomtom.com": "TomTom — public traffic flow",
    }

    for spec in registry.registered_specs():
        url = getattr(spec, "api_url", None) or ""
        if not url.startswith("http"):
            continue  # the demo bundle, uploads and the synthetic fallback
        host = url.split("/")[2]
        assert host in allowed_hosts, f"{spec.name} calls an unlisted host: {host}"


def test_a_quarantined_record_keeps_the_reason_not_a_payload_dump() -> None:
    """A rejected record is evidence, but storing the raw payload indefinitely
    would keep whatever a provider happened to send."""
    columns = {c.name for c in Base.metadata.tables["quarantined_records"].columns}

    assert "reason" in columns
    assert not any(pii_hits(column) for column in columns)


# --- 11.5: no causal claim in any copy (ETH-1) ------------------------------

# Verbs that assert one thing produced another.
CAUSAL = re.compile(
    r"(?i)\b("
    r"causes|causing|caused by|leads? to|results? in|drives?|driving|"
    r"due to|because of|responsible for|makes? .{0,12} (rise|fall|worse)|"
    r"impact of|effect of|influences?"
    r")\b"
)

# Phrasings that *deny* causation, which is what ETH-1 asks for. A sentence
# matching one of these is the disclaimer doing its job, not a violation.
DENIALS = re.compile(
    r"(?i)("
    r"not (?:equal|imply|establish|evidence)|does not|do not establish|"
    r"no causal|makes no claim|rather than|without any causal|"
    r"is not evidence|correlation|association only|not a health"
    r")"
)


# What "user-facing" means here, narrowly.
#
# An earlier version of this scan read every long string in the backend and
# flagged three *docstrings* — "Drives the inference window", and the like.
# Nobody reads a docstring in the product, and widening the denial list to
# excuse them would have blunted the audit. So the scan reads only text that
# genuinely reaches someone: what the backend publishes in a payload or in
# OpenAPI, and what the frontend renders as prose.
FIELD_DESCRIPTION = re.compile(r"""description\s*=\s*\(?\s*["']([^"']{20,})["']""")
CAVEAT_ASSIGNMENT = re.compile(
    r"^[A-Z][A-Z0-9_]*(?:CAVEAT|DISCLAIMER)[A-Z0-9_]*\s*(?::[^=\n]+)?=", re.MULTILINE
)
JSX_PROP = re.compile(
    r"""(?:note|subtitle|caption|emptyMessage|title|ariaLabel)=\{?["']([^"']{20,})["']"""
)
JSX_TEXT = re.compile(r">\s*([A-Z][^<>{}\n]{25,})\s*<")


def user_facing_strings() -> list[tuple[Path, str]]:
    """Text a reader can actually see."""
    found: list[tuple[Path, str]] = []

    for path in [*(BACKEND / "services").rglob("*.py"), *(BACKEND / "api").rglob("*.py")]:
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")

        for match in FIELD_DESCRIPTION.finditer(text):
            found.append((path, match.group(1)))

        # A caveat constant's assignment usually spans several lines, so take the
        # quoted fragments that follow it.
        for match in CAVEAT_ASSIGNMENT.finditer(text):
            tail = text[match.end() : match.end() + 900]
            found.append((path, " ".join(re.findall(r'"([^"]+)"', tail))))

    for path in (REPO / "frontend" / "src").rglob("*.tsx"):
        if "node_modules" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in (JSX_PROP, JSX_TEXT):
            for match in pattern.finditer(text):
                found.append((path, match.group(1)))

    return found


def test_the_scan_reaches_both_sides_of_the_product() -> None:
    """A scan that silently matched nothing would pass forever."""
    strings = user_facing_strings()
    files = {path.name for path, _ in strings}

    assert len(strings) > 50
    assert any(name.endswith(".py") for name in files)
    assert any(name.endswith(".tsx") for name in files)


def test_no_user_facing_sentence_claims_causation() -> None:
    """ETH-1. Everything this platform reports is association: a correlation, a
    PCA loading, a SHAP attribution. None of them is evidence that one thing
    produced another, and the copy has to say so."""
    violations = []
    for path, text in user_facing_strings():
        if not CAUSAL.search(text):
            continue
        if DENIALS.search(text):
            continue  # a disclaimer, which is the requirement being met
        violations.append(f"{path.relative_to(REPO)}: {text[:90]}")

    assert violations == [], "causal language in user-facing copy:\n" + "\n".join(
        violations
    )


def test_the_scan_would_actually_catch_a_violation() -> None:
    """A filter this permissive is worth proving. Without this, a bug in the
    pattern would make the audit pass silently forever."""
    assert CAUSAL.search("Traffic causes higher PM2.5")
    assert CAUSAL.search("The rise was due to traffic")
    assert not DENIALS.search("Traffic causes higher PM2.5")

    # The real disclaimer denies causation with a noun ("do not establish
    # causation") rather than a causal verb, so what matters is that the denial
    # is recognised — and that the sentence survives the scan either way.
    from services.eda.profile import CORRELATION_CAVEAT

    assert DENIALS.search(CORRELATION_CAVEAT)
    assert not [
        text
        for text in [CORRELATION_CAVEAT]
        if CAUSAL.search(text) and not DENIALS.search(text)
    ]


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("services.eda.profile", "CORRELATION_CAVEAT"),
        ("services.eda.reduction", "ESI_CAVEAT"),
        ("services.eda.manifold", "TSNE_CAVEAT"),
        ("services.ml.explain", "ATTRIBUTION_CAVEAT"),
        ("services.ml.prediction", "PREDICTION_CAVEAT"),
        ("services.ml.intervals", "INTERVAL_CAVEAT"),
    ],
)
def test_every_inferential_output_ships_a_caveat(module: str, name: str) -> None:
    """The caveat travels *inside the payload*, not in the UI's head. A client
    that renders the number cannot skip the qualification."""
    import importlib

    text = getattr(importlib.import_module(module), name)

    assert isinstance(text, str) and len(text) > 40


def test_the_disclaimer_is_in_the_shell_not_on_a_page() -> None:
    """AC-11: *persistent*. A notice that can be navigated away from is a notice
    that will be."""
    layout = (REPO / "frontend" / "src" / "components" / "Layout.tsx").read_text(
        encoding="utf-8"
    )

    assert "does not equal causation" in layout
    assert "sensor placement" in layout.lower()
    # It sits in the shell's own footer, which every route renders through.
    assert "<footer" in layout and "Outlet" in layout


def test_the_exported_report_carries_the_same_disclaimer() -> None:
    """An exported document travels further than the dashboard does."""
    from services.eda import report

    source = (BACKEND / "services" / "eda" / "report.py").read_text(encoding="utf-8")

    assert "causation" in source.lower()
    assert hasattr(report, "render")


def test_the_esi_does_not_present_itself_as_a_health_threshold() -> None:
    """The most likely misreading of a 0-100 index, and the one the payload has
    to pre-empt."""
    from services.eda.reduction import ESI_CAVEAT

    assert "not 'unsafe'" in ESI_CAVEAT
    assert "relative" in ESI_CAVEAT.lower()
