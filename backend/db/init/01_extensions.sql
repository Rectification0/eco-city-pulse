-- Task 1.1 — PostGIS, enabled before any migration runs.
--
-- docker-entrypoint-initdb.d scripts execute exactly once, on a *fresh* data
-- volume. On an existing volume this file is ignored, so the same statement is
-- repeated defensively (and conditionally) in the initial Alembic migration.
--
-- CREATE EXTENSION needs superuser, which POSTGRES_USER is inside this image.

CREATE EXTENSION IF NOT EXISTS postgis;

-- Spatial indexing helpers used by the district joins in design §6.1.
CREATE EXTENSION IF NOT EXISTS btree_gist;
