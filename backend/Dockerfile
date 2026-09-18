# --- Stage 1: build wheels ------------------------------------------------
# xgboost/prophet/shap may need a toolchain; keep it out of the final image.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# --- Stage 2: runtime -----------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser . .

RUN chmod +x /app/entrypoint.sh

USER appuser
EXPOSE 8000

# start-period covers migrations plus the first demo seed on a cold volume,
# which is the slowest path this container has.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# The entrypoint applies migrations and seeds demo data before serving, so
# `docker compose up` still needs no follow-up commands (task 1.6 / 1.9).
ENTRYPOINT ["/app/entrypoint.sh"]
