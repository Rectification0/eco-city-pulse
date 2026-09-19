"""prediction location

Adds ``lat``/``lon`` to ``predictions`` (task 9.5, 9.6).

specs §9 lists the table as ``id, model_id, target_time, predicted_value,
actual_value``. That is enough to *record* a forecast but not enough to score
one: design §6.1 requires ``actual_value`` to be backfilled from the observation
the forecast was about, and "the observation at that hour" is not a single row
-- every station reports that hour. Matching on time alone would pick an
arbitrary station's reading and call it the outcome, which would quietly
corrupt the drift-monitoring dataset the table exists to become.

So a prediction records where it was for. Nullable, because a future city-wide
aggregate forecast would legitimately have no single location, and because rows
written before this migration have none.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-19 18:30:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0003'
down_revision: str | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('predictions', sa.Column('lat', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('lon', sa.Float(), nullable=True))
    # The backfill scans for rows still missing an outcome whose target hour has
    # passed, then matches them to a station. This index serves exactly that
    # query; without it the scan grows with the whole prediction history.
    op.create_index(
        'ix_predictions_pending_backfill',
        'predictions',
        ['target_time', 'lat', 'lon'],
        unique=False,
        postgresql_where=sa.text('actual_value IS NULL'),
    )


def downgrade() -> None:
    op.drop_index('ix_predictions_pending_backfill', table_name='predictions')
    op.drop_column('predictions', 'lon')
    op.drop_column('predictions', 'lat')
