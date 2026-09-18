"""Declarative base and shared column conventions.

The naming convention matters more than it looks: without it PostgreSQL
invents constraint names, Alembic autogenerate cannot match an existing
constraint to a model one, and every migration drifts. Fixing the pattern up
front keeps migrations deterministic for the rest of the project.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# DR-2: every persisted instant is timezone-aware and stored as UTC.
TIMESTAMPTZ = DateTime(timezone=True)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def normalize_utc(value: datetime | None, *, field: str) -> datetime | None:
    """Coerce an aware datetime to UTC; reject a naive one outright (DR-2).

    Silently assuming that a naive datetime is UTC is how a dataset ends up
    with a six-hour phase error in its lag features that nobody notices until
    the model underperforms. Fail at the boundary instead.
    """
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{field} must be timezone-aware; naive datetimes are rejected (DR-2)."
        )
    return value.astimezone(timezone.utc)
