"""Exit 0 when the observations table is empty, 1 otherwise.

A shell-friendly predicate for ``entrypoint.sh``: it decides whether the demo
seed should run without the entrypoint needing to know any SQL, and without a
psql client in the runtime image.
"""

from __future__ import annotations

import sys

from sqlalchemy import func, select

from db.models import Observation
from db.session import session_scope


def main() -> int:
    with session_scope() as session:
        count = session.scalar(select(func.count()).select_from(Observation)) or 0
    return 0 if count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
