# Eco-City Pulse

Urban environmental intelligence platform: ingest heterogeneous city data, subject it to rigorous exploratory data analysis, and forecast **PM2.5** with an interpretable ML pipeline.

Built as a **modular monolith** — not "just an ML model", but the whole chain:

```
Data Collection → Data Engineering → EDA → Statistical Analysis → Visualization
  → Feature Engineering → ML → Evaluation → Explainability → Prediction → Decision Support
```

| Document | Contents |
|----------|----------|
| [`specs.md`](./specs.md) | Requirements, personas, API contract, acceptance criteria |
| [`design.md`](./design.md) | Architecture, schema, pipeline design, decision rationale |
| [`tasks.md`](./tasks.md) | 12-phase implementation plan with requirement traceability |
| [`RUNNING.md`](./RUNNING.md) | **Start here to run it** — both routes, verification, troubleshooting |
| [`CLAUDE.md`](./CLAUDE.md) | Conventions and per-phase workflow for contributors |
| [`docs/`](./docs) | Design notes: source merge, retention, the multi-city plan |

> Design rationale lives in `design.md` and in the module docstrings, not here.
> This file is for getting the thing running and finding your way around.

---

## Quick start

```bash
cp .env.example .env      # optional — defaults work as-is
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | <http://localhost:8080> |
| API health | <http://localhost:8000/api/v1/health> |
| API docs (Swagger) | <http://localhost:8000/docs> |
| PostgreSQL (container) | `localhost:5433` |

On first boot the backend applies migrations, loads the demo dataset, and runs
the data quality pipeline before serving — nothing to run afterwards. Cold start
takes a minute or two; `docker compose logs -f backend` shows progress.

> The database publishes on **5433** so it does not collide with a PostgreSQL
> already on the machine; inside the compose network it is still 5432.

**No API keys are required** — `INGESTION_MODE=demo` runs entirely offline, a
hard requirement (DR-1) rather than a convenience. Keys, live ingestion and the
analytics scope are covered in [`RUNNING.md`](./RUNNING.md).

---

## Architecture

Three containers (design §4):

| Container | Stack | Role |
|-----------|-------|------|
| `frontend` | React · TypeScript · Vite · Tailwind → **Nginx** | SPA; proxies `/api` so the browser sees one origin |
| `backend` | **FastAPI** (Uvicorn) | REST API, EDA engine, ML training and inference |
| `db` | **PostgreSQL 16 + PostGIS** | Observations, model registry, predictions, spatial queries |

```
frontend (nginx:80) ──/api/──► backend (uvicorn:8000) ──► db (postgres:5432)
```

**Layering rule:** routes validate input, call a service, shape the response —
no analytics logic. Analysis lives in `backend/services/`, persistence in
`backend/db/`.

---

## Pipeline

Each stage is a package under `backend/services/`. The guarantee column is what
the tests actually assert.

| Stage | Module | Guarantee |
|-------|--------|-----------|
| **Ingestion** (FEAT-01) | `adapters/`, `harmonizer.py`, `ingestion_service.py` | UTC · decimal degrees · hourly grid (DR-2…DR-4). A failed live fetch degrades that source to `offline` and still returns 200 |
| **Data quality** | `quality/` | MCAR/MAR characterisation, MICE + short-gap fill. Anomalies are **flagged, never deleted** (AC-4, AC-5) |
| **EDA** (FEAT-02) | `eda/` | Univariate · Pearson+Spearman · distribution profile, STL decomposition, self-contained HTML report (AC-3) |
| **Features** | `features/` | Lags/rollings on a reindexed hourly grid; **one transformer** for training and serving, byte-identical (AC-8) |
| **ESI** (FEAT-04) | `eda/reduction.py`, `eda/manifold.py` | PCA PC1 → **0–100**, oriented to PM2.5, loadings shipped with every score (AC-6). t-SNE is EDA-only |
| **ML** (FEAT-05) | `ml/` | 3 horizons × 4 models. Chronological split + embargo, scaler inside the model, expanding-window CV (AC-7, AC-8) |
| **Explainability** | `ml/explain.py` | `prediction = base_value + Σ contributions`, asserted for every model |
| **Prediction** (FEAT-06) | `ml/prediction.py`, `ml/intervals.py` | The specs §8 contract exactly; interval calibrated on the model's own held-out residuals (AC-9) |
| **Retention** | `retention.py` | Sweeps superseded and orphaned model artifacts; never deletes a `models` row or an observation (AC-4, AC-5) |

Two commitments run through all of it: **nothing is fitted on data that is later
scored** — not the scaler, not the feature selection, not the log-transform
decision, and each run reports the timestamps that prove it (AC-8) — and **no
causal claim anywhere** (ETH-1), with every correlation, loading and SHAP value
shipping its caveat inside the payload.

### CLI

```bash
cd backend
python -m scripts.seed_demo --days 120   # offline demo dataset
python -m scripts.run_quality            # impute, flag anomalies
python -m scripts.build_features         # engineered feature store
python -m scripts.train_models           # train + register the model ladder
python -m scripts.prune --dry-run        # retention sweep (drop the flag to apply)
python -m scripts.export_docs --report   # API reference + EDA report → docs/
```

---

## Local development

### Database

The compose database (`postgis/postgis:16-3.4`) includes PostGIS. A stock local
PostgreSQL usually does not, and without it everything works *except* the
spatial features — district joins and map layers.

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

alembic upgrade head                 # create the schema
python -m scripts.seed_demo          # load the offline demo dataset
uvicorn main:app --reload --port 8000
```

> Against the **compose** database from the host, use port 5433:
> `POSTGRES_HOST=localhost POSTGRES_PORT=5433 uvicorn main:app --reload`.

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api to localhost:8000
```

### Tests

```bash
cd backend
pytest                 # full suite; db tests skip if no database is reachable
pytest -m db           # only the tests needing PostgreSQL
pytest -m "not db"     # everything that runs offline
pytest tests/test_acceptance.py -v      # AC-1 … AC-11, one test each
pytest tests/test_audit_*.py            # the SEC/PRIV/ETH audits
ruff check --select F,I,E9 .
```

The audits are **executable**, not a checklist: an endpoint without a response
model, a credential in a compose file, `allow_origins=["*"]`, a PII-shaped
column or a caption claiming causation each fail the suite. Fixtures blank every
API key and roll back every `db` test, so no test can reach the outside world.

> Tests failing on a Windows host that blocks scikit-learn's compiled
> extensions? Run them in the container instead — see
> [`RUNNING.md`](./RUNNING.md).

---

## Project layout

`frontend/src` (charts · components · pages · `services/api.ts`) and `backend`
(`api/routes` · `core` · `db` · `services` · `scripts` · `tests`). The annotated
tree is [`design.md` §5](./design.md).

---

## Configuration

Environment-driven, parsed once in `backend/core/config.py`. **No secret has a
real default and none is ever committed** (SEC-2). Full list in
[`.env.example`](./.env.example).

| Variable | Default | Notes |
|----------|---------|-------|
| `INGESTION_MODE` | `demo` | `scheduled` · `manual` · `upload` · `demo` |
| `ANALYTICS_SOURCE_SCOPE` | `demo` | Which provenance EDA/training/serving read when a request names no `source_ids`: `demo` · `live` · `all` |
| `FRONTEND_ORIGINS` | `http://localhost:8080,http://localhost:5173` | CORS allowlist; never `*` (SEC-3) |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `db` / `5432` | What the backend connects to |
| `POSTGRES_HOST_PORT` | `5433` | What the container publishes on the host |
| `POSTGRES_PASSWORD` | — | Percent-encoded into the URL, so `@ : / #` are safe |
| `AQICN_API_KEY` · `OPENWEATHER_API_KEY` · `TOMTOM_API_KEY` | *(empty)* | Optional — blank means that source runs offline |
| `RUN_MIGRATIONS` · `AUTO_SEED_DEMO` · `AUTO_RUN_QUALITY` | `true` | Container start-up behaviour |
| `DEMO_SEED_DAYS` | `120` | History generated by the automatic seed |
| `INGESTION_INTERVAL_MINUTES` | `60` | Scheduled mode only |
| `UPLOAD_MAX_BYTES` | `10000000` | Ceiling for `POST /data/upload` |
| `MODEL_ARTIFACT_DIR` | `artifacts` | Backed by the `model_artifacts` volume so registry rows never dangle |
| `DB_ECHO` | `false` | Log every SQL statement; separate from `DEBUG` |
| `ARTIFACT_KEEP_PER_MODEL` · `RUN_LOG_RETENTION_DAYS` · `UNSCORED_PREDICTION_RETENTION_DAYS` | `3` · `90` · `30` | Retention windows; `observations` is never pruned |

---

## API

Base URL `/api/v1`. Every request and response is a Pydantic model — validation
is the security boundary (SEC-1). Full contract at
<http://localhost:8000/docs>.

| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| `GET` | `/health` | Liveness, mode, whether any key is configured | ✅ 0 |
| `GET` | `/data/districts` | District boundaries as GeoJSON | ✅ 1 |
| `GET` | `/data/sources` | Sources with ingestion health | ✅ 2 |
| `POST` | `/data/ingest` | Trigger a run (Manual / Demo) | ✅ 2 |
| `POST` | `/data/upload` | Ingest a CSV or JSON file | ✅ 2 |
| `GET` | `/data/ingestion/runs` | The ingestion log | ✅ 2 |
| `POST` | `/data/quality` | Impute, detect anomalies, flag | ✅ 3 |
| `POST` | `/eda/profile` | Univariate, bivariate, distribution statistics | ✅ 4 |
| `POST` | `/eda/decompose` | STL trend / seasonal / residual | ✅ 4 |
| `GET` | `/eda/report` | Self-contained HTML EDA report | ✅ 4 |
| `GET` | `/eda/cache` | Profile cache statistics | ✅ 4 |
| `POST` | `/eda/reduce` | PCA components, variance, loadings, ESI | ✅ 6 |
| `POST` | `/eda/tsne` | t-SNE 2D projection (EDA Studio only) | ✅ 6 |
| `POST` | `/eda/scatter` | Scatter points, OLS line, r / ρ / r², n (VIZ-1) | ✅ 12 |
| `POST` | `/eda/grouped` | Grouped means ± 95% CI and box summaries (VIZ-2, VIZ-3) | ✅ 12 |
| `POST` | `/eda/pairplot` | Sampled scatter matrix of every measurement (VIZ-4) | ✅ 12 |
| `POST` | `/eda/andrews` | Andrews curves per class, exact mean curves (VIZ-5) | ✅ 12 |
| `POST` | `/ml/train` | Train the ladder and register the results | ✅ 7 |
| `GET` | `/ml/models` | The model registry, newest first | ✅ 7 |
| `GET` | `/ml/models/{id}/importance` | SHAP feature importance for one model | ✅ 8 |
| `GET` | `/data/observations/latest` | Newest reading per station | ✅ 10 |
| `GET` | `/data/observations/series` | One station's recent hourly readings | ✅ 10 |
| `POST` | `/ml/predict` | PM2.5 forecast with interval and reasoning | ✅ 9 |
| `POST` | `/ml/predictions/backfill` | Fill in outcomes once the hour has passed | ✅ 9 |

**`POST /data/ingest` returns 200 even when a source fails** — DR-1 expressed in
the API. Read the per-source `status`, not just the HTTP code.

Errors share one envelope from `core/exceptions.py`:

```json
{ "error": { "code": "dataset_not_found", "message": "...", "details": {} } }
```

---

## Implementation status

| Phase | Scope | Status |
|-------|-------|--------|
| **0–4** | Foundation · data layer · ingestion · quality · EDA | ✅ Complete |
| **5** | Feature engineering — one shared transformer | ✅ Complete |
| **6** | Dimensionality reduction & ESI (FEAT-04) | ✅ Complete |
| **7** | ML pipeline — 3 horizons, 4 models, leakage audit (FEAT-05) | ✅ Complete |
| **8** | Explainability — SHAP, additive per model | ✅ Complete |
| **9** | Prediction service — conformal intervals, persisted forecasts (FEAT-06) | ✅ Complete |
| **10** | Frontend — Dashboard, EDA Studio, Model Lab, Admin | ✅ Complete |
| **11** | Security, quality & release — audits as tests, AC walkthrough | ✅ Complete |
| **Post-release** | Provenance scope · cross-source merge · retention | ✅ Complete |
| **12** | Visual EDA — scatter, grouped bars & boxplots, pair plot, Andrews curves | ✅ Complete |

Track detail in [`tasks.md`](./tasks.md).

---

## Ethics

Predictions rely on sensor placement, which may exhibit geographic and
socioeconomic bias. **Correlation shown does not equal causation** — causal
inference is out of scope and no causal claim is made anywhere in this platform.
This notice is surfaced in the UI itself (task 10.16).

Only public environmental data is used. Any text analysis aggregates metrics
only and stores no PII.

---

## Academic mapping (BACSE301)

| Module | Topic | Implemented in |
|--------|-------|----------------|
| Mod 1 | Data Collection & Structure | Multi-source API & CSV ingestion, JSON validation, DB storage (Phase 2) |
| Mod 2 | Data Preprocessing | MCAR/MAR analysis, MICE, IQR / Z-score / Isolation Forest vote (Phase 3) |
| Mod 3 | Descriptive Stats & Visualization | Univariate profile, Pearson + Spearman correlation (Phase 4) |
| Mod 4 | Dimensionality & Time-Series | STL (Phase 4) · PCA-based ESI and t-SNE (Phase 6) · ARIMA/Prophet (Phase 7) |
| Mod 5 | Advanced Visualization | Self-contained HTML report with inline SVG charts (Phase 4); parallel coordinates in Phase 10 |
