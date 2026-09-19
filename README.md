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
| [`CLAUDE.md`](./CLAUDE.md) | Conventions and per-phase workflow for contributors |

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

> The database publishes on **5433**, not 5432, so it does not collide with a
> PostgreSQL already on the machine. Inside the compose network the backend
> still reaches it on 5432.

**No API keys are required.** The platform defaults to `INGESTION_MODE=demo` and
runs entirely offline — a hard requirement (DR-1), not a convenience. A source
with no key reports as `offline` and is never contacted; traffic falls back to a
clearly-labelled synthetic series.

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

**Layering rule:** `backend/api/routes/*` contain no analytics logic. They
validate input, call a service, and shape the response. All pandas/scikit-learn
work lives in `backend/services/`, and all persistence in `backend/db/`.

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

Two commitments run through all of it:

- **Nothing is fitted on data that is later scored** — not the scaler, not the
  feature selection, not the log-transform decision. Each run reports the
  timestamps that prove it (AC-8).
- **No causal claim anywhere** (ETH-1). Correlation, PCA loadings and SHAP
  attributions all ship with caveats inside the payload.

### CLI

```bash
cd backend
python -m scripts.seed_demo --days 120   # offline demo dataset
python -m scripts.run_quality            # impute, flag anomalies
python -m scripts.build_features         # engineered feature store
python -m scripts.train_models           # train + register the model ladder
```

---

## Local development

### Database

| Setup | How | PostGIS |
|-------|-----|---------|
| **Container** (default) | `docker compose up` — `postgis/postgis:16-3.4` | ✅ included |
| **Existing local server** | Point `POSTGRES_HOST`/`POSTGRES_PORT` at it in `.env` | ⚠️ only if installed separately |

> **PostGIS caveat.** A stock PostgreSQL does not ship PostGIS. Without it
> everything works *except* the spatial features — district GeoJSON joins and
> map layers. Use the compose database for those.

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
ruff check --select F,I,E9 .
```

Fixtures strip every `Settings` variable from the environment and blank every
upstream API key, so no test can be influenced by — or reach — the outside
world. `db`-marked tests run inside a transaction that is rolled back.

> If the host blocks scikit-learn's compiled extensions (Windows Application
> Control), run the suite in the container instead — same code, Linux runtime:
> ```bash
> docker compose run --rm --no-deps --entrypoint pytest backend -q
> ```
> `tests/` is excluded from the production image, so mount it:
> `-v "$PWD/backend/tests:/app/tests"`.

---

## Project layout

```
eco-city-pulse/
├── frontend/src/{components,pages,services}   # SPA; services/ is the only place fetch appears
├── backend/
│   ├── api/routes/          # Thin HTTP layer — health, data, eda, ml
│   ├── core/                # config.py (env secrets), exceptions.py (error envelope)
│   ├── db/                  # models, session, migrations, init
│   ├── services/
│   │   ├── adapters/        # One module per source + the registry
│   │   ├── quality/         # Missingness, imputation, outliers, pipeline
│   │   ├── eda/             # Profile, STL, cache, report, PCA/ESI, t-SNE
│   │   ├── features/        # Spec, temporal, windows, transformer, store
│   │   └── ml/              # Targets, splitting, models, registry, training, SHAP
│   ├── scripts/             # seed_demo · run_quality · build_features · train_models
│   └── tests/
├── data/{raw,processed}/    # Landing zone · generated analysis output
├── artifacts/               # Model registry storage (generated)
└── docker-compose.yml
```

---

## Configuration

Environment-driven, parsed once in `backend/core/config.py`. **No secret has a
real default and none is ever committed** (SEC-2). Full list in
[`.env.example`](./.env.example).

| Variable | Default | Notes |
|----------|---------|-------|
| `INGESTION_MODE` | `demo` | `scheduled` · `manual` · `upload` · `demo` |
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
| `POST` | `/ml/train` | Train the ladder and register the results | ✅ 7 |
| `GET` | `/ml/models` | The model registry, newest first | ✅ 7 |
| `GET` | `/ml/models/{id}/importance` | SHAP feature importance for one model | ✅ 8 |
| `POST` | `/ml/predict` | PM2.5 forecast with reasoning | Phase 9 |

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
| 9 | Prediction service (FEAT-06) | ⬜ |
| 10 | Frontend — Dashboard, EDA Studio, Model Lab | ⬜ |
| 11 | Security, quality & release | ⬜ |

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
