"""Export the release artefacts (task 11.9).

    python -m scripts.export_docs            # API reference only
    python -m scripts.export_docs --report   # and the EDA report (needs a DB)

Two documents, both **generated rather than written**:

* ``docs/openapi.json`` — the API reference, produced from the running
  application. A reference maintained by hand drifts the moment an endpoint
  changes; this one cannot describe a route that no longer exists.
* ``docs/eda-report.html`` — the automated EDA report (task 4.7), self-contained
  and printable to PDF.

Neither is committed. Both are reproducible from the code and the database, and
a generated file in version control is a merge conflict waiting to happen; the
command *is* the deliverable. ``tests/test_acceptance.py`` asserts the contract
behind the first one, so regenerating it is safe rather than hopeful.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.config import get_settings
from main import create_app

DOCS = Path(__file__).resolve().parent.parent.parent / "docs"


def export_openapi(destination: Path) -> Path:
    """Write the OpenAPI contract the application actually serves."""
    schema = create_app().openapi()

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return destination


def export_report(destination: Path) -> Path:
    """Render the EDA report over whatever is currently in the database."""
    from db.session import session_scope
    from services.eda import service

    settings = get_settings()
    with session_scope(settings) as session:
        result = service.generate_report(session, settings, persist=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.html, encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.export_docs",
        description="Generate the API reference and the EDA report (task 11.9).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also render the EDA report. Needs a reachable, seeded database.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DOCS,
        help="Directory to write into. Defaults to ./docs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    schema_path = export_openapi(args.out / "openapi.json")
    paths = [schema_path]

    if args.report:
        paths.append(export_report(args.out / "eda-report.html"))

    print("Wrote:")
    for path in paths:
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} kB)")

    if not args.report:
        print()
        print("  (--report also renders the EDA report; it needs a seeded database.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
