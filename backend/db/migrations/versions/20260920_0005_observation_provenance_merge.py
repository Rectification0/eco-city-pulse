"""observation provenance merge

Re-keys ``observations`` from ``(source_id, timestamp, lat, lon)`` to
``(provenance, timestamp, lat, lon)``, so that several feeds describing one
station-hour merge into one row.

The table has always described itself as holding "one harmonized hourly reading
at one location ... the joined environmental state at a point in space and
time". For the demo bundle that was true, because one source writes every
column. For live data it was not: AQICN supplied pm25/pm10, OpenWeather
temp/humidity and TomTom traffic_score, each as its *own* row, because
``source_id`` sat in the unique key and made a collision between them
impossible. So no live row ever carried both PM2.5 and traffic — the target and
its strongest explanatory variable could not appear together — and the map's
``DISTINCT ON (lat, lon)`` resolved each centroid to whichever feed wrote last,
which is usually one that has no PM2.5 at all.

Everything needed to merge was already present and simply never met: the
harmonizer's ``resample_hourly`` groups by ``(hour, lat, lon)`` with no source,
and the write path's upsert already coalesced NULLs. This migration removes the
one thing keeping them apart.

**Provenance rather than no key at all.** Dropping ``source_id`` outright would
merge the demo bundle into live rows -- they share the district centroids and
overlap in time -- and a statistic built from both is a claim about measurement
that measurement does not support (ETH-1). Two classes, mirroring
``data_sources.is_synthetic``, keep the scope a request resolves and the key a
row merges on from ever disagreeing.

**Existing rows.** Backfilled from the source's flag, then de-duplicated: rows
that now share a key are collapsed into the lowest id, taking the first
non-NULL value per column in id order, and ``is_anomaly`` by OR so a quality
judgement is never lost. That ordering is best-effort and deliberately *not*
the authority rule the write path applies -- the information needed to apply it
properly (which source contributed which column) was never stored per row. Rows
written before this migration therefore keep whichever value came first; a
re-ingest settles them correctly.

Rows ingested before the adapters were aligned also keep their original
coordinates, so historical AQICN rows stay at the monitor's position rather
than the district's and will not merge with anything. They are not wrong, only
unmergeable. Re-ingest for a clean grid.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-20 15:30:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0005'
down_revision: str | None = '0004'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MEASUREMENTS = ('pm25', 'pm10', 'temp', 'humidity', 'traffic_score')

OLD_CONSTRAINT = 'uq_observations_source_timestamp_location'
NEW_CONSTRAINT = 'uq_observations_provenance_timestamp_location'


def upgrade() -> None:
    # 'measured' as the server default: the overwhelming majority of rows in
    # any deployment are, and the synthetic ones are corrected immediately.
    op.add_column(
        'observations',
        sa.Column(
            'provenance',
            sa.String(length=16),
            nullable=False,
            server_default='measured',
        ),
    )
    op.create_check_constraint(
        'observation_provenance',
        'observations',
        "provenance IN ('measured', 'synthetic')",
    )
    op.execute(
        """
        UPDATE observations o
           SET provenance = 'synthetic'
          FROM data_sources d
         WHERE d.id = o.source_id
           AND d.is_synthetic
        """
    )

    # The old key has to go before the rows that only it kept apart can be
    # collapsed.
    op.drop_constraint(OLD_CONSTRAINT, 'observations', type_='unique')

    coalesced = ',\n               '.join(
        f'{name} = c.{name}' for name in MEASUREMENTS
    )
    firsts = ',\n                   '.join(
        f'(array_agg({name} ORDER BY id) '
        f'FILTER (WHERE {name} IS NOT NULL))[1] AS {name}'
        for name in MEASUREMENTS
    )
    op.execute(
        f"""
        WITH grouped AS (
            SELECT provenance, timestamp, lat, lon,
                   MIN(id) AS keep_id,
                   bool_or(is_anomaly) AS is_anomaly,
                   {firsts}
              FROM observations
             GROUP BY provenance, timestamp, lat, lon
            HAVING COUNT(*) > 1
        )
        UPDATE observations o
           SET {coalesced},
               is_anomaly = c.is_anomaly
          FROM grouped c
         WHERE o.id = c.keep_id
        """
    )
    op.execute(
        """
        DELETE FROM observations o
         USING (
            SELECT provenance, timestamp, lat, lon, MIN(id) AS keep_id
              FROM observations
             GROUP BY provenance, timestamp, lat, lon
         ) c
         WHERE o.provenance = c.provenance
           AND o.timestamp  = c.timestamp
           AND o.lat = c.lat
           AND o.lon = c.lon
           AND o.id <> c.keep_id
        """
    )

    op.create_unique_constraint(
        NEW_CONSTRAINT, 'observations', ['provenance', 'timestamp', 'lat', 'lon']
    )


def downgrade() -> None:
    # One-way in substance: the rows merged above cannot be split back into the
    # per-source rows they came from, because which source contributed which
    # column was never recorded. The schema reverts; the data stays merged, and
    # the old constraint holds only because one row per source-hour is a subset
    # of what it allowed.
    op.drop_constraint(NEW_CONSTRAINT, 'observations', type_='unique')
    op.create_unique_constraint(
        OLD_CONSTRAINT, 'observations', ['source_id', 'timestamp', 'lat', 'lon']
    )
    op.drop_constraint('observation_provenance', 'observations', type_='check')
    op.drop_column('observations', 'provenance')
