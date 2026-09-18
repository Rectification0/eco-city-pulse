"""initial schema

Creates the four tables of specs §9 plus the spatial layer of design §6.1:
PostGIS, a generated ``observations.geom`` point derived from lat/lon, and the
composite index that serves both time-series windows and map queries (1.6).

The spatial parts are conditional. A stock local PostgreSQL has no PostGIS, and
the spec only needs it for district joins and map layers -- so a machine without
it still migrates cleanly and loses only those features, rather than failing.

Revision ID: 0001
Revises:
Create Date: 2026-09-18 13:54:16.717956
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _postgis_ready() -> bool:
    """Install PostGIS when the server offers it; report whether it is usable.

    ``CREATE EXTENSION`` needs superuser. The compose database runs as one, but
    a managed or shared server may not, so a refusal degrades to "no spatial
    column" instead of aborting the migration.
    """
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return False

    installed = bind.exec_driver_sql(
        "SELECT 1 FROM pg_extension WHERE extname = 'postgis'"
    ).scalar()
    if installed:
        return True

    available = bind.exec_driver_sql(
        "SELECT 1 FROM pg_available_extensions WHERE name = 'postgis'"
    ).scalar()
    if not available:
        return False

    try:
        op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    except Exception:  # pragma: no cover - depends on server privileges
        return False
    return True


def upgrade() -> None:
    spatial = _postgis_ready()

    op.create_table(
        "data_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("api_url", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "healthy",
                "degraded",
                "offline",
                name="source_status",
                native_enum=False,
            ),
            server_default="offline",
            nullable=False,
        ),
        sa.Column("last_run", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_data_sources")),
        sa.UniqueConstraint("name", name=op.f("uq_data_sources_name")),
    )

    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column(
            "features_used",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()), "postgresql"
            ),
            nullable=False,
        ),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("r2", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "mae IS NULL OR mae >= 0", name=op.f("ck_models_mae_non_negative")
        ),
        sa.CheckConstraint(
            "rmse IS NULL OR rmse >= 0", name=op.f("ck_models_rmse_non_negative")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_models")),
    )
    op.create_index(
        "ix_models_target_created_at",
        "models",
        ["target", sa.literal_column("created_at DESC")],
        unique=False,
    )

    op.create_table(
        "observations",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("pm25", sa.Float(), nullable=True),
        sa.Column("pm10", sa.Float(), nullable=True),
        sa.Column("temp", sa.Float(), nullable=True),
        sa.Column("humidity", sa.Float(), nullable=True),
        sa.Column("traffic_score", sa.Float(), nullable=True),
        sa.Column("is_anomaly", sa.Boolean(), server_default="false", nullable=False),
        sa.CheckConstraint(
            "humidity IS NULL OR humidity BETWEEN 0 AND 100",
            name=op.f("ck_observations_humidity_percent"),
        ),
        sa.CheckConstraint(
            "lat BETWEEN -90 AND 90", name=op.f("ck_observations_lat_decimal_degrees")
        ),
        sa.CheckConstraint(
            "lon BETWEEN -180 AND 180", name=op.f("ck_observations_lon_decimal_degrees")
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["data_sources.id"],
            name=op.f("fk_observations_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_observations")),
        sa.UniqueConstraint(
            "source_id",
            "timestamp",
            "lat",
            "lon",
            name="uq_observations_source_timestamp_location",
        ),
    )
    op.create_index(
        "ix_observations_timestamp_lat_lon",
        "observations",
        ["timestamp", "lat", "lon"],
        unique=False,
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("target_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_value", sa.Float(), nullable=False),
        sa.Column("actual_value", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["models.id"],
            name=op.f("fk_predictions_model_id_models"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
    )
    op.create_index(
        "ix_predictions_model_id_target_time",
        "predictions",
        ["model_id", "target_time"],
        unique=False,
    )
    op.create_index(
        "ix_predictions_target_time", "predictions", ["target_time"], unique=False
    )

    if spatial:
        # A generated column, not a trigger: the geometry can never drift from
        # the lat/lon it is derived from, and the ingestion path stays unaware
        # of PostGIS entirely. env.py excludes it from autogenerate.
        op.execute(
            "ALTER TABLE observations ADD COLUMN geom geometry(Point, 4326) "
            "GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(lon, lat), 4326)) STORED"
        )
        op.execute("CREATE INDEX ix_observations_geom ON observations USING GIST (geom)")


def downgrade() -> None:
    # The geometry column and its index disappear with the table.
    op.drop_index("ix_predictions_target_time", table_name="predictions")
    op.drop_index("ix_predictions_model_id_target_time", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_observations_timestamp_lat_lon", table_name="observations")
    op.drop_table("observations")
    op.drop_index("ix_models_target_created_at", table_name="models")
    op.drop_table("models")
    op.drop_table("data_sources")
    # PostGIS itself is left installed: other schemas may depend on it, and it
    # was not necessarily created by this migration.
