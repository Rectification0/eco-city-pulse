"""Alembic environment.

Two deliberate choices:

* The URL comes from ``core.config`` rather than ``alembic.ini``, so migrations
  and the application can never disagree about which database they mean, and no
  credential is committed (SEC-2).
* ``include_object`` hides the PostGIS geometry column from autogenerate. That
  column is created by DDL in the initial migration (it is a generated column,
  which SQLAlchemy does not model here), so without the filter every future
  autogenerate would helpfully propose dropping it.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool

# db/migrations/env.py -> backend/. Needed because Alembic may be invoked from
# a different working directory than the one prepend_sys_path assumes.
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.config import get_settings  # noqa: E402
from db.base import Base  # noqa: E402
from db.models import (  # noqa: E402,F401  (import registers the tables)
    DataSource,
    MLModel,
    Observation,
    Prediction,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Objects created by raw DDL in a migration rather than by the ORM.
UNMANAGED_COLUMNS = {("observations", "geom")}
UNMANAGED_INDEXES = {"ix_observations_geom", "idx_observations_geom"}
UNMANAGED_TABLES = {"spatial_ref_sys"}  # shipped by PostGIS itself


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    if type_ == "table" and name in UNMANAGED_TABLES:
        return False
    if type_ == "column" and (obj.table.name, name) in UNMANAGED_COLUMNS:
        return False
    if type_ == "index" and name in UNMANAGED_INDEXES:
        return False
    return True


def get_url() -> str:
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (``alembic upgrade head --sql``)."""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
