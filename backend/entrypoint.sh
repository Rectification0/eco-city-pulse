#!/bin/sh
# Container startup: migrate, seed if empty, then serve.
#
# This is what keeps the README's one-command promise true now that there is a
# schema to create. Compose already waits for the database healthcheck, so by
# the time this runs the server is accepting connections.
set -eu

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
    echo "[entrypoint] applying migrations..."
    alembic upgrade head
fi

# Demo mode only, and only into an empty table: re-seeding is idempotent, but
# skipping it entirely keeps restarts fast and leaves ingested data alone.
if [ "${INGESTION_MODE:-demo}" = "demo" ] && [ "${AUTO_SEED_DEMO:-true}" = "true" ]; then
    if python -m scripts.check_seed_needed; then
        echo "[entrypoint] seeding demo dataset (offline, no API keys)..."
        python -m scripts.seed_demo --days "${DEMO_SEED_DAYS:-120}"
    else
        echo "[entrypoint] observations already present; skipping demo seed."
    fi
fi

echo "[entrypoint] starting uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
