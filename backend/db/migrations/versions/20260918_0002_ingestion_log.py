"""ingestion log

Adds the two tables behind FEAT-01's "Output: ingestion log": a run-level
record per ingestion attempt, and the individual records that failed their
source schema, kept with the reason (task 2.7, 2.8).

These sit outside the four tables of specs §9 because the specification
requires the log without saying where it lives.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-18 14:39:15.113324
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('ingestion_runs',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('source_id', sa.Integer(), nullable=False),
    sa.Column('mode', sa.Text(), nullable=False),
    sa.Column('status', sa.Enum('success', 'partial', 'failed', 'skipped', name='run_status', native_enum=False), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('records_fetched', sa.Integer(), server_default='0', nullable=False),
    sa.Column('records_valid', sa.Integer(), server_default='0', nullable=False),
    sa.Column('records_quarantined', sa.Integer(), server_default='0', nullable=False),
    sa.Column('records_written', sa.Integer(), server_default='0', nullable=False),
    sa.Column('message', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['source_id'], ['data_sources.id'], name=op.f('fk_ingestion_runs_source_id_data_sources'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ingestion_runs'))
    )
    op.create_index('ix_ingestion_runs_source_id_started_at', 'ingestion_runs', ['source_id', sa.literal_column('started_at DESC')], unique=False)
    op.create_table('quarantined_records',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('run_id', sa.BigInteger(), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('payload', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['ingestion_runs.id'], name=op.f('fk_quarantined_records_run_id_ingestion_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_quarantined_records'))
    )
    op.create_index('ix_quarantined_records_run_id', 'quarantined_records', ['run_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_quarantined_records_run_id', table_name='quarantined_records')
    op.drop_table('quarantined_records')
    op.drop_index('ix_ingestion_runs_source_id_started_at', table_name='ingestion_runs')
    op.drop_table('ingestion_runs')
