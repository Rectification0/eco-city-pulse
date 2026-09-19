"""Generate the EDA report (task 4.7).

    python -m scripts.build_report

The same work as ``GET /api/v1/eda/report``, for a cron job or a container
shell. Writes a self-contained HTML file to ``data/processed``.

Deliberately not wired into ``entrypoint.sh``: the endpoint generates the
report on demand and writes the same file, so building one at boot would only
delay the healthcheck for an artefact nobody has asked for yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import get_settings
from db.session import session_scope
from services.eda import report, service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.build_report",
        description="Render the automated EDA report (Module 5).",
    )
    parser.add_argument(
        "--no-decomposition",
        action="store_true",
        help="Skip the STL section, which is the slowest part.",
    )
    parser.add_argument(
        "--pdf",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Also write a PDF. Needs WeasyPrint; without it, open the HTML and "
            "use the browser's Print to PDF."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    with session_scope(settings) as session:
        result = service.generate_report(
            session, settings, include_decomposition=not args.no_decomposition
        )

    print("EDA report generated:")
    print(f"  sections  {', '.join(result.sections)}")
    print(f"  size      {len(result.html.encode('utf-8')):,} bytes")
    print(f"  html      {result.path}")

    if args.pdf is not None:
        destination = (
            Path(args.pdf) if args.pdf else settings.data_processed_path / "eda_report.pdf"
        )
        try:
            written = report.to_pdf(result.html, destination)
        except RuntimeError as exc:
            print(f"  pdf       not written: {exc}")
            return 1
        print(f"  pdf       {written}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
