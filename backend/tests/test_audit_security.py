"""Security audit — tasks 11.1, 11.2, 11.3 (SEC-1, SEC-2, SEC-3).

An audit written as a document is a claim about the past. These are assertions
about the code as it is now, so the next endpoint added without a response model,
the next credential pasted into a compose file, and the next `allow_origins=["*"]`
all fail the suite rather than surviving until someone re-reads a checklist.

Where a rule has an exception, the exception is **named here with its reason**.
An audit that cannot express "this one, because…" gets silenced instead.
"""

from __future__ import annotations

import re
import types
from pathlib import Path
from typing import Union, get_args, get_origin

import pytest
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from core.config import Settings
from main import create_app

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent


@pytest.fixture(scope="module")
def app():
    return create_app(
        Settings(
            _env_file=None,
            postgres_password="test-only",
            frontend_origins="http://localhost:5173",
        )
    )


def api_routes(app) -> list[APIRoute]:
    return [route for route in app.routes if isinstance(route, APIRoute)]


# --- 11.1: every request and response is a Pydantic model (SEC-1) -----------

# The one endpoint that returns something other than JSON, and why.
NON_JSON_ROUTES = {
    "/api/v1/eda/report": "returns text/html — the report artefact *is* the document",
}


def test_every_endpoint_declares_a_response_model(app) -> None:
    """Validation is the security boundary: an undeclared response is an
    unvalidated one, and it also vanishes from the OpenAPI contract."""
    undeclared = [
        route.path
        for route in api_routes(app)
        if route.response_model is None and route.path not in NON_JSON_ROUTES
    ]

    assert undeclared == [], f"routes without a response_model: {undeclared}"


def _body_members(annotation: object) -> list[object]:
    """The annotation split into the alternatives a caller may actually send.

    Several endpoints declare `Model | None` because the body is optional --
    POST with nothing and the service runs on its defaults. That is still a
    closed contract, so the audit compares the *members* of the union rather
    than the union object: `Model | None` passes, `dict | None` does not.
    `NoneType` is dropped because "no body" is the absence of input, not an
    unvalidated one.

    Unwrapping is done here, not in the route signatures, because the
    alternative -- dropping `| None` to satisfy the check -- would make a
    missing body a 422 and break FEAT-02's "profile everything by default".
    """
    if get_origin(annotation) in (Union, types.UnionType):
        return [arg for arg in get_args(annotation) if arg is not type(None)]
    return [annotation]


def test_every_request_body_is_a_pydantic_model(app) -> None:
    """A dict body would accept anything, which is the failure SEC-1 names."""
    loose = []
    for route in api_routes(app):
        field = getattr(route, "body_field", None)
        if field is None:
            continue
        for member in _body_members(field.type_):
            # A file upload is not a BaseModel and cannot be; it is validated by
            # size and content type in the handler instead (task 2.3).
            if getattr(member, "__name__", "") in {"UploadFile", "bytes"}:
                continue
            if not (isinstance(member, type) and issubclass(member, BaseModel)):
                loose.append((route.path, member))

    assert loose == [], f"routes with a non-model body: {loose}"


def test_an_optional_body_still_has_to_name_a_model(app) -> None:
    """The unwrapping above must not become a hole: `X | None` is accepted
    only because X is checked, so a union hiding a raw dict still fails."""
    assert _body_members(dict | None) == [dict]
    assert not all(
        isinstance(m, type) and issubclass(m, BaseModel)
        for m in _body_members(dict | None)
    )


def test_the_error_envelope_is_the_only_error_shape(app) -> None:
    """Every failure leaves through `core.exceptions`, so a caller parses one
    shape rather than guessing per endpoint."""
    from core.exceptions import EcoCityPulseError

    handlers = app.exception_handlers

    assert EcoCityPulseError in handlers
    payload = EcoCityPulseError("x", details={"a": 1}).to_payload()
    assert set(payload["error"]) == {"code", "message", "details"}


# --- 11.2: no secret in source, image, or compose (SEC-2) -------------------

# Credential-shaped assignments. Deliberately crude: the point is to catch a
# pasted key, and a false positive is cheap to add an exception for.
SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (api[_-]?key|secret|password|token|passwd)
    \s*[:=]\s*
    ['"]([^'"\n]{8,})['"]
    """
)

# Values that look like credentials but are not. Each needs a reason.
ALLOWED_LITERALS = {
    "ecocity-dev-password": (
        "the local compose default. AC-1 requires `docker compose up` to work "
        "with no .env at all, so the database needs *a* password; it is "
        "reachable only inside the compose network and is overridden by "
        "POSTGRES_PASSWORD in any real deployment."
    ),
}

# Placeholders `.env.example` is allowed to carry. The file exists to show the
# *shape* of the configuration, so a password line with nothing after the `=`
# would teach a reader that the field is optional, which it is not.
PLACEHOLDERS = {"changeme", "change-me", "your-key-here", "replace-me"}

# `tests/` is excluded from the credential scan, and the exclusion is the honest
# one: these files are *supposed* to contain credential-shaped strings, because
# their job is to prove that a secret stays out of a repr, that an absent key
# degrades a source to offline, and that a wrong fingerprint is refused. Testing
# credential handling without writing a fake credential is not possible. The
# scan therefore covers everything that ships.
SCAN_EXCLUDED = {"tests"}


def source_files() -> list[Path]:
    """Committed Python, TypeScript, YAML and Docker files that ship."""
    roots = [BACKEND, REPO / "frontend" / "src", REPO]
    seen: dict[Path, None] = {}
    for root in roots:
        for pattern in ("*.py", "*.ts", "*.tsx", "*.yml", "*.yaml"):
            for path in root.rglob(pattern):
                if any(
                    part in {".venv", "node_modules", "dist", "__pycache__", ".git"}
                    | SCAN_EXCLUDED
                    for part in path.parts
                ):
                    continue
                seen[path] = None
    for name in ("Dockerfile", "backend/Dockerfile", "frontend/Dockerfile"):
        candidate = REPO / name
        if candidate.exists():
            seen[candidate] = None
    return list(seen)


def test_no_credential_is_hardcoded_anywhere() -> None:
    findings = []
    for path in source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in SECRET_ASSIGNMENT.finditer(text):
            value = match.group(2)
            if value in ALLOWED_LITERALS:
                continue
            # Interpolation and env lookups are the correct pattern, not a leak.
            if value.startswith("${") or "os.environ" in value or value.startswith("/"):
                continue
            findings.append(f"{path.relative_to(REPO)}: {match.group(0).strip()[:70]}")

    assert findings == [], "possible hardcoded credentials:\n" + "\n".join(findings)


def test_the_env_file_is_ignored_and_the_example_is_not() -> None:
    """The example is committed so a clone knows what to set; the real file
    never is."""
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")

    assert "\n.env\n" in gitignore
    assert "!.env.example" in gitignore
    assert (REPO / ".env.example").exists()


def test_the_example_env_carries_no_real_values() -> None:
    """An example with a working key in it is a leak with extra steps."""
    populated = []
    for line in (REPO / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if not any(
            marker in key.upper() for marker in ("KEY", "SECRET", "TOKEN", "PASSWORD")
        ):
            continue
        value = value.strip().strip("'\"")
        if value and value.lower() not in PLACEHOLDERS:
            populated.append(line)

    assert populated == [], f".env.example holds real-looking values: {populated}"


def test_the_example_still_shows_which_fields_exist() -> None:
    """A placeholder, not an empty line: `POSTGRES_PASSWORD=` would read as
    "optional", and it is not."""
    text = (REPO / ".env.example").read_text(encoding="utf-8")

    assert "POSTGRES_PASSWORD=changeme" in text
    # The API keys *are* optional — blank is the correct example (DR-1).
    for key in ("AQICN_API_KEY", "OPENWEATHER_API_KEY", "TOMTOM_API_KEY"):
        empty = f"{key}="
        assert f"{empty}\n" in text or text.rstrip().endswith(empty)


def test_api_keys_have_no_default_anywhere() -> None:
    """A defaulted key would let a deployment think it is configured when it is
    not — and DR-1 depends on an absent key meaning *offline*, loudly."""
    settings = Settings(_env_file=None, postgres_password="x")

    for name in ("aqicn_api_key", "openweather_api_key", "tomtom_api_key"):
        assert getattr(settings, name).get_secret_value() == ""


def test_a_secret_never_renders_in_a_dump_or_repr() -> None:
    """SecretStr keeps the value out of logs and tracebacks; the assembled
    database URL is a plain property for the same reason, so it stays out of
    `model_dump()`."""
    settings = Settings(_env_file=None, postgres_password="hunter2-not-real")

    assert "hunter2-not-real" not in repr(settings)
    assert "hunter2-not-real" not in str(settings.model_dump())
    assert "database_url" not in settings.model_dump()
    # It is still assembled correctly when actually asked for.
    assert "hunter2-not-real" in settings.database_url


def test_the_compose_file_passes_secrets_by_interpolation() -> None:
    """Every credential in compose is `${VAR}`, so the value lives in the
    environment rather than in a committed file."""
    compose = (REPO / "docker-compose.yml").read_text(encoding="utf-8")

    for line in compose.splitlines():
        if "API_KEY" in line and ":" in line:
            value = line.split(":", 1)[1].strip()
            assert value.startswith("${"), f"key not interpolated: {line.strip()}"


def test_the_image_bakes_in_no_credential() -> None:
    """A secret in an ENV layer survives in the image history."""
    dockerfile = (BACKEND / "Dockerfile").read_text(encoding="utf-8")

    for line in dockerfile.splitlines():
        if line.strip().startswith("ENV"):
            assert not SECRET_ASSIGNMENT.search(line), f"secret in image: {line}"


# --- 11.3: the CORS allowlist (SEC-3) ---------------------------------------


def cors_options(app) -> dict:
    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            return middleware.kwargs
    raise AssertionError("CORS middleware is not installed")


def test_cors_is_an_allowlist_and_never_a_wildcard(app) -> None:
    options = cors_options(app)

    assert options["allow_origins"] == ["http://localhost:5173"]
    assert "*" not in options["allow_origins"]


def test_cors_has_no_regex_backdoor(app) -> None:
    """`allow_origin_regex` would reopen what the allowlist closes."""
    assert cors_options(app).get("allow_origin_regex") is None


def test_credentialed_requests_cannot_come_from_anywhere(app) -> None:
    """`allow_credentials` with a wildcard origin is the combination browsers
    refuse and servers should never offer."""
    options = cors_options(app)

    if options.get("allow_credentials"):
        assert "*" not in options["allow_origins"]


def test_the_allowlist_comes_from_configuration_not_code() -> None:
    settings = Settings(
        _env_file=None,
        postgres_password="x",
        frontend_origins="https://a.example, https://b.example",
    )

    assert settings.cors_origins == ["https://a.example", "https://b.example"]


def test_methods_are_narrowed_to_what_the_api_uses(app) -> None:
    """The API reads and writes; it never DELETEs or PUTs, so those are not
    offered."""
    methods = set(cors_options(app)["allow_methods"])

    assert methods <= {"GET", "POST", "OPTIONS"}


def test_get_settings_is_the_only_reader_of_the_environment() -> None:
    """SEC-2: secrets are parsed once, in one place. A stray `os.environ` in a
    service would bypass the SecretStr handling entirely."""
    offenders = []
    for path in BACKEND.rglob("*.py"):
        if any(part in {".venv", "__pycache__", "tests"} for part in path.parts):
            continue
        if path.name in {"config.py", "conftest.py"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in ("os.environ", "os.getenv"):
            if marker in text:
                offenders.append(f"{path.relative_to(BACKEND)}: {marker}")

    assert offenders == [], f"environment read outside config.py: {offenders}"
