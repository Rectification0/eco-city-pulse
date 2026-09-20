"""source provenance

Adds ``data_sources.is_synthetic``, the flag that keeps modelled values out of
statistics presented as measurement.

``observations`` is deliberately one wide table across every source, which is
what makes the lag and rolling features well defined. The cost is that a demo
row and an AQICN row are indistinguishable once loaded: ``load_observations``
with no ``source_ids`` returned both, so the moment live ingestion started
alongside a seeded demo bundle, every profile, correlation and model was
computed over 31,680 synthetic points and a handful of real ones -- and
reported as though all of it had been observed. ETH-1 asks that caveats travel
with the payload; a number that cannot say which of the two it came from
cannot carry that caveat at all.

Provenance is a property of the *source*, so it belongs here rather than on
each observation: one flag per source, not one per row, and no backfill of
31,680 rows to add a fact that was already implied by ``source_id``.

A column rather than a list of known-synthetic names in code, because the
question "is this measured?" must survive a source being renamed, and because
``ensure_sources`` already reflects the adapter registry into this table --
``AdapterSpec.synthetic`` is the single place the answer is declared.

Existing rows are backfilled by name since that is the only evidence available
at migration time; from here on ``ensure_sources`` keeps the column in step
with the registry.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-20 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0004'
down_revision: str | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The two sources that generate rather than observe. Named here and nowhere
# else in the migration history; the application reads the column, not this
# list.
SYNTHETIC_SOURCE_NAMES = (
    'Demo Bundle (synthetic, offline)',
    'Synthetic Traffic (fallback)',
)


def upgrade() -> None:
    # server_default so the column is NOT NULL from the start: every existing
    # row gets false, and the two synthetic ones are corrected immediately
    # below. Adding it nullable and tightening later would leave a window in
    # which "unknown provenance" was representable, which is the state this
    # migration exists to abolish.
    op.add_column(
        'data_sources',
        sa.Column(
            'is_synthetic',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )

    op.execute(
        sa.text(
            'UPDATE data_sources SET is_synthetic = true WHERE name IN :names'
        ).bindparams(sa.bindparam('names', value=SYNTHETIC_SOURCE_NAMES, expanding=True))
    )


def downgrade() -> None:
    op.drop_column('data_sources', 'is_synthetic')
