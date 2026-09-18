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
| Source health | <http://localhost:8000/api/v1/data/sources> |
| District boundaries | <http://localhost:8000/api/v1/data/districts> |
| API docs (Swagger) | <http://localhost:8000/docs> |
| PostgreSQL (container) | `localhost:5433` |

On first boot the backend applies migrations and loads the demo dataset before
it starts serving, so there is nothing to run afterwards. Cold start takes
roughly a minute; `docker compose logs -f backend` shows the progress.

> The database container publishes on **5433**, not 5432, so it does not collide
> with a PostgreSQL already installed on the machine. Inside the compose network
> the backend still reaches it on 5432.

**No API keys are required.** The platform defaults to `INGESTION_MODE=demo` and
runs entirely offline — a hard requirement from the spec (DR-1), not a
convenience. A source with no key reports as `offline` and is never contacted;
traffic falls back to a clearly-labelled synthetic series. Nothing about the
platform degrades into an error because a third party is unavailable.

---

## Architecture

Three containers (design §4):

| Container | Stack | Role |
|-----------|-------|------|
| `frontend` | React · TypeScript · Vite · Tailwind → **Nginx** | SPA; proxies `/api` to the backend so the browser sees one origin |
| `backend` | **FastAPI** (Uvicorn) | REST API, EDA engine, ML training and inference |
| `db` | **PostgreSQL 16 + PostGIS** | Observations, model registry, predictions, spatial queries |

```
frontend (nginx:80) ──/api/──► backend (uvicorn:8000) ──► db (postgres:5432)
```

**Layering rule:** `backend/api/routes/*` contain no analytics logic. They validate input, call a service, and shape the response. All pandas/scikit-learn work lives in `backend/services/`, and all persistence in `backend/db/`.

---

## Data layer

### Schema

Four tables (specs §9, design §6.1). Every timestamp is `TIMESTAMPTZ` and every
coordinate is decimal degrees — naive datetimes are rejected at the ORM boundary
rather than silently assumed to be UTC (DR-2, DR-3).

| Table | Holds | Notes |
|-------|-------|-------|
| `data_sources` | One row per provider | `status` ∈ `healthy · degraded · offline`; a keyless live API is *offline*, which is a normal state, not an error |
| `observations` | Harmonized hourly readings | Wide, not key/value: one row is the joined environmental state at a point in space and time, which is what makes lag features well defined |
| `models` | Model registry | `features_used` is JSONB; metrics and `artifact_path` are written at registration |
| `predictions` | Issued forecasts | Append-only; `actual_value` is backfilled once the real observation lands, turning the table into a drift-monitoring dataset |
| `ingestion_runs` | One row per ingestion attempt | Beyond specs §9, because FEAT-01 requires a log without saying where it lives |
| `quarantined_records` | Records that failed their schema, with the reason | The evidence behind a `degraded` source |

Measurement columns are **nullable on purpose** — missingness is the signal the
Phase 3 quality engine analyses, so it has to survive the write path. Pollutant
values carry no range constraint for the same reason: a genuine spike must reach
the table to be *flagged*, never rejected (AC-5).

**Indexes**

- `ix_observations_timestamp_lat_lon` — the composite index serving both
  time-series windows and map queries (design §6.1).
- `uq_observations_source_timestamp_location` — the conflict target that makes
  re-ingestion idempotent.
- `ix_models_target_created_at` — resolves "the current model for this target".

**Spatial.** Where PostGIS is present, `observations.geom` is a *generated*
`geometry(Point, 4326)` column derived from `lon`/`lat`, with a GiST index. A
generated column rather than a trigger means the geometry can never drift from
the coordinates it comes from, and the ingestion path stays unaware of PostGIS
entirely.

### Migrations

Alembic, configured in `backend/alembic.ini`. The connection URL is **not** in
that file — it is assembled in `core/config.py` from environment variables, so
no credential is ever committed (SEC-2).

```bash
cd backend
alembic upgrade head          # apply
alembic downgrade base        # roll back
alembic check                 # assert the models and the database agree
alembic revision --autogenerate -m "what changed"
```

The initial migration installs PostGIS when the server offers it and skips the
spatial column when it does not, so a stock local PostgreSQL still migrates
cleanly and loses only the map features.

### Demo dataset

```bash
cd backend
python -m scripts.seed_demo --days 120        # load
python -m scripts.seed_demo --csv             # load, and dump the CSV too
```

Runs with **no network access at all** — the Phase 1 exit criterion and AC-2.
The data is synthetic, generated by `services/demo_data.py` using only the
standard library, and labelled as synthetic everywhere it is persisted. It is
shaped rather than random, because the later phases need structure to find:

| Property | Exists so that |
|----------|----------------|
| Seasonal · diurnal · weekly cycles | STL decomposition (4.5) has something to separate |
| Autocorrelated residuals (AR(1)) | Lag features (5.2) predict, and the naive baseline (7.5) is genuinely hard to beat |
| PM2.5 ↑ with traffic, ↓ with temperature | The correlation matrix (4.2) and PCA (6.2) have signal to report |
| Missingness of two mechanisms — flat dropout (MCAR) and PM10 dropping out when PM2.5 is high (MAR) | The MCAR/MAR/MNAR judgement (3.1) is real, and MICE (3.2) is the justified repair |
| Multi-hour station outages | Forward/backward fill (3.3) has contiguous gaps to fill |
| Rare extreme spikes, left unflagged | The outlier detectors (3.4–3.7) have something to detect |

Values are a function of the timestamp, not of when the generator runs, so
re-seeding is idempotent: the same hour always produces the same row, and the
upsert updates it in place rather than duplicating it.

Since Phase 2 the script owns no database logic of its own — it calls
`ingestion_service.run_demo`, so demo data enters through the same pipeline as
a live API.

### District boundaries

`data/raw/districts.geojson` holds 11 districts of Delhi and is served at
`GET /api/v1/data/districts`. The demo seed places one virtual station on each
district centroid, so seeded points always fall inside the polygons the map
draws.

> **These polygons are schematic**, not surveyed administrative boundaries —
> each district is a rectangular cell placed at its true relative position in
> the city. The file says so in its own `metadata.accuracy`, and that statement
> is served to the UI with the data. Replace it with an official boundary file
> before any operational use.

---

## Ingestion

One pipeline, four ways in (FEAT-01, design §6.3). Every mode runs the same
five stages, differing only in the adapter at the front:

```
fetch → validate (quarantine failures) → harmonize → upsert → log
```

Demo mode goes through it too. "Ingestion completes end-to-end in Demo mode
with every live API disabled" (AC-2) only means something if demo mode
exercises the real path rather than a private shortcut.

### Modes

| Mode | Trigger | Adapter |
|------|---------|---------|
| **Demo** | Container start, or `POST /data/ingest?mode=demo` | Offline generator |
| **Manual** | `POST /data/ingest` | Live HTTP clients |
| **Scheduled** | Background task, `INGESTION_MODE=scheduled` | Live HTTP clients |
| **Upload** | `POST /data/upload` | CSV / JSON parser |

Scheduling is an in-process asyncio task rather than a fourth container: it
needs no dependency, and an external cron can drive the same work by calling
`POST /data/ingest`. It starts **only** in `scheduled` mode, so the default
install makes no outbound request at all.

### Sources

| Source | Provides | Without a key |
|--------|----------|---------------|
| **AQICN** | PM2.5, PM10, temperature, humidity | `skipped` — never contacted |
| **OpenWeather** | Temperature, humidity | `skipped` — never contacted |
| **TomTom** | Congestion index | Replaced by the **synthetic fallback** |

Two conversions are worth knowing about, because without them the database
would hold the wrong quantity under the right column name:

- **AQICN reports an AQI index, not µg/m³.** It is inverted through the EPA
  breakpoint table (`services/adapters/aqi_scale.py`). An AQI of 155 and
  155 µg/m³ are wildly different amounts of pollution.
- **TomTom reports speeds, not congestion.** The stored `traffic_score` is
  `100 × (1 − current ÷ free-flow)`, the same 0–100 index the synthetic
  fallback produces, so the two are interchangeable downstream.

The synthetic fallback (task 2.2) exists because PM2.5 without traffic loses
the strongest explanatory variable in the dataset. It is registered under a
name containing "Synthetic", never claims to be measured, and reuses the demo
generator's traffic curve so the series stays continuous with seeded history.

### Harmonization (DR-2 … DR-4)

`services/harmonizer.py`, deliberately **without pandas** — ingestion handles a
stream of records, not a matrix, and the scientific stack belongs in the
analysis phases.

| Step | Rule |
|------|------|
| **Timestamps → UTC** | ISO strings, epoch seconds and datetimes all accepted. A naive value uses an assumption the *caller* states; nothing guesses. |
| **Coordinates → decimal degrees** | Floats, `28.61N`, `28°36'36"N` and DMS all parse. A latitude marked `E` is rejected as a transposition. Rounded to 5 dp (~1 m) so feed jitter cannot shard one sensor into several. |
| **Resample → hourly** | Grouped by `(UTC hour, lat, lon)`, each field averaged over the values actually present. Conversion happens *before* flooring — flooring first would truncate in the source's own zone and land off-grid. |

Locations are never averaged together: that would smear the city's spatial
gradient, which is the thing the map exists to show.

### Quarantine and the log

A record that fails its schema is **stored with the reason**, not dropped. A
provider changing its payload shape should surface as a visible pile of
quarantined records rather than an unexplained gap discovered weeks later. One
bad row never costs the batch — the other rows still load, and the source is
marked `degraded`.

```bash
curl localhost:8000/api/v1/data/ingestion/runs | jq
```

Storage is capped per run (500 records) so a file that is malformed from the
first byte cannot fill the table; the full count is still reported.

### Writes

`observations` has exactly one writer, `ingestion_service.write_observations`.
It upserts on `(source_id, timestamp, lat, lon)`:

- a re-ingested hour is **updated in place**, never duplicated;
- a new NULL never erases a known value — a partial mid-hour fetch must not
  delete what a complete fetch already wrote;
- `is_anomaly` is left untouched, because it belongs to the Phase 3 quality
  engine and re-ingestion must not silently unflag a record (AC-5).

Each source writes its own rows. AQICN rows carry air quality, OpenWeather rows
carry weather; they are joined at analysis time rather than merged on write, so
provenance survives.

### Uploading a file

```bash
curl -X POST localhost:8000/api/v1/data/upload   -F "file=@observations.csv"   -F "assume_timezone_offset_minutes=330"   # 0 = UTC; recorded in the log
```

Column aliases are accepted — `pm2.5`/`pm2_5`/`pm25`, `lon`/`lng`/`longitude`,
`rh`/`humidity` — because rejecting a file over header spelling pushes people
into hand-editing data before uploading it. Blank cells and `NA`/`null`/`-`
become missing values, which is what Phase 3 is built to handle. Format is
decided by content, not by the file extension.

---

## Local development

### Database

Two supported setups:

| Setup | How | PostGIS |
|-------|-----|---------|
| **Container** (default) | `docker compose up` — uses the `postgis/postgis:16-3.4` image | ✅ included |
| **Existing local server** | Point `POSTGRES_HOST`/`POSTGRES_PORT` at it in `.env` | ⚠️ only if installed separately |

```bash
cp .env.example .env      # then edit POSTGRES_* to taste
```

> **PostGIS caveat.** A stock PostgreSQL install does not ship PostGIS. Without
> it, everything works *except* the spatial features — district GeoJSON joins
> and the map layers (tasks 1.1, 1.8, 10.4). Use the compose database for those,
> or install PostGIS into the local server via Stack Builder.

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

alembic upgrade head                 # create the schema
python -m scripts.seed_demo          # load the offline demo dataset
uvicorn main:app --reload --port 8000
```

> Running against the **compose** database from the host means port 5433:
> `POSTGRES_HOST=localhost POSTGRES_PORT=5433 uvicorn main:app --reload`.

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api to localhost:8000
```

### Tests

```bash
cd backend && pytest                 # full suite; database tests skip if none is reachable
pytest -m db                         # only the tests that need PostgreSQL
pytest -m "not db"                   # everything that runs offline
```

Tests are hermetic in two senses. The fixtures strip every `Settings` variable
from the environment, so an exported `POSTGRES_USER` cannot change what the
assertions see, and they blank every upstream API key, so a developer who *has*
a real AQICN key cannot make the suite call it. No test touches the network.

`db`-marked tests run inside a transaction that is rolled back afterwards, so
they exercise the real schema — constraints, upsert semantics and all — without
leaving anything behind. They **skip** when no database answers, with a message
saying whether the server is unreachable or merely unmigrated.

---

## Project layout

```
eco-city-pulse/
├── frontend/
│   ├── src/
│   │   ├── components/          # Reusable chart/map/KPI components
│   │   ├── pages/               # Dashboard, EDA Studio, Model Lab
│   │   ├── services/            # Typed API client — the only place fetch appears
│   │   └── App.tsx
│   ├── nginx.conf               # SPA fallback + /api reverse proxy
│   ├── package.json
│   └── tailwind.config.js
├── backend/
│   ├── api/
│   │   ├── routes/              # Thin HTTP layer — health.py, data.py
│   │   └── dependencies.py      # Settings and DB session injection
│   ├── core/
│   │   ├── config.py            # Pydantic settings — the only reader of env secrets
│   │   └── exceptions.py        # Domain exceptions → HTTP envelope
│   ├── db/
│   │   ├── base.py              # Declarative base, naming convention, UTC coercion
│   │   ├── models.py            # The four tables of specs §9
│   │   ├── session.py           # Lazy engine + session scope
│   │   ├── init/                # Runs once on a fresh volume — enables PostGIS
│   │   └── migrations/          # Alembic env + versions
│   ├── services/
│   │   ├── adapters/            # One module per source + the registry
│   │   ├── harmonizer.py        # UTC, decimal degrees, hourly resample
│   │   ├── ingestion_service.py # The single write path
│   │   ├── scheduler.py         # Scheduled-mode background task
│   │   ├── demo_data.py         # Offline demo dataset generator
│   │   └── geo_service.py       # Districts and station locations
│   ├── scripts/
│   │   └── seed_demo.py         # python -m scripts.seed_demo
│   ├── tests/
│   ├── alembic.ini
│   ├── entrypoint.sh            # Migrate, seed if empty, then serve
│   ├── main.py                  # App factory: CORS, routers, handlers
│   └── requirements.txt
├── data/
│   ├── raw/                     # Immutable landing zone — districts.geojson
│   └── processed/               # Cleaned / feature-engineered output
├── docker-compose.yml
└── README.md
```

---

## Configuration

All configuration is environment-driven and parsed once in `backend/core/config.py`. **No secret has a real default and none is ever committed** (SEC-2). See [`.env.example`](./.env.example) for the full list.

| Variable | Default | Notes |
|----------|---------|-------|
| `INGESTION_MODE` | `demo` | `scheduled` · `manual` · `upload` · `demo` |
| `FRONTEND_ORIGINS` | `http://localhost:8080,http://localhost:5173` | CORS allowlist; never `*` (SEC-3) |
| `POSTGRES_HOST` | `db` | `localhost` to use a database already on the machine |
| `POSTGRES_PORT` | `5432` | Port the **backend connects to** |
| `POSTGRES_HOST_PORT` | `5433` | Port the **container publishes on** the host |
| `POSTGRES_PASSWORD` | — | Required in production; safe to contain `@`, `:`, `/` — it is percent-encoded into the connection URL |
| `AQICN_API_KEY` | *(empty)* | Optional — blank means that source runs offline |
| `OPENWEATHER_API_KEY` | *(empty)* | Optional |
| `TOMTOM_API_KEY` | *(empty)* | Optional; a synthetic traffic baseline substitutes |
| `DB_ECHO` | `false` | Log every SQL statement. Deliberately separate from `DEBUG` — it is deafening during a bulk load |
| `RUN_MIGRATIONS` | `true` | Apply migrations on container start |
| `AUTO_SEED_DEMO` | `true` | Seed the demo dataset on start, in demo mode, only when `observations` is empty |
| `DEMO_SEED_DAYS` | `120` | History generated by the automatic seed |
| `INGESTION_INTERVAL_MINUTES` | `60` | Scheduled mode only. Hourly, because DR-4 puts every source on an hourly grid |
| `INGESTION_STARTUP_DELAY_SECONDS` | `30` | Grace period before the first scheduled run |
| `UPLOAD_MAX_BYTES` | `10000000` | Ceiling for `POST /data/upload`, enforced on bytes actually read |

---

## API

Base URL `/api/v1`. Every request and response is a Pydantic model — validation
is the security boundary (SEC-1). Full contract at
<http://localhost:8000/docs>.

| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| `GET` | `/health` | Liveness, mode, whether any key is configured | ✅ Phase 0 |
| `GET` | `/data/districts` | District boundaries as GeoJSON | ✅ Phase 1 |
| `GET` | `/data/sources` | Sources with ingestion health | ✅ Phase 2 |
| `POST` | `/data/ingest` | Trigger a run (Manual / Demo) | ✅ Phase 2 |
| `POST` | `/data/upload` | Ingest a CSV or JSON file | ✅ Phase 2 |
| `GET` | `/data/ingestion/runs` | The ingestion log | ✅ Phase 2 |
| `POST` | `/eda/profile` | Univariate + bivariate statistics | Phase 4 |
| `POST` | `/eda/reduce` | PCA components, variance, loadings | Phase 6 |
| `POST` | `/ml/predict` | PM2.5 forecast with reasoning | Phase 9 |

**`POST /data/ingest` returns 200 even when a source fails.** That is DR-1
expressed in the API: an unreachable third party degrades that source to
`offline` and is reported in its outcome, rather than becoming a 5xx for the
caller. Read the per-source `status` field, not just the HTTP code.

Errors share one envelope, produced by the handlers in `core/exceptions.py`:

```json
{ "error": { "code": "dataset_not_found", "message": "...", "details": {} } }
```

---

## Implementation status

| Phase | Scope | Status |
|-------|-------|--------|
| **0** | Project foundation — containers, config, app factory, health, tests | ✅ Complete |
| **1** | Data layer — schema, migrations, PostGIS, demo seed, district boundaries | ✅ Complete |
| **2** | Ingestion engine — adapters, harmonizer, quarantine, four modes (FEAT-01) | ✅ Complete |
| 3 | Data quality engine — MICE, outliers | ⬜ |
| 4 | EDA & statistical engine (FEAT-02) | ⬜ |
| 5 | Feature engineering | ⬜ |
| 6 | Dimensionality reduction & ESI (FEAT-04) | ⬜ |
| 7 | ML pipeline (FEAT-05) | ⬜ |
| 8 | Explainability — SHAP | ⬜ |
| 9 | Prediction service (FEAT-06) | ⬜ |
| 10 | Frontend — Dashboard, EDA Studio, Model Lab | ⬜ |
| 11 | Security, quality & release | ⬜ |

Track detail in [`tasks.md`](./tasks.md).

---

## Ethics

Predictions rely on sensor placement, which may exhibit geographic and socioeconomic bias. **Correlation shown does not equal causation** — causal inference is explicitly out of scope, and no causal claim is made anywhere in this platform. This notice is surfaced in the UI itself (task 10.16).

Only public environmental data is used. Any text analysis aggregates metrics only and stores no PII.

---

## Academic mapping (BACSE301)

| Module | Topic | Implementation |
|--------|-------|----------------|
| Mod 1 | Data Collection & Structure | Multi-source API & CSV ingestion, JSON validation, DB storage |
| Mod 2 | Data Preprocessing | MICE imputation, Z-score / Isolation Forest anomaly detection, scaling |
| Mod 3 | Descriptive Stats & Visualization | Automated EDA dashboard — histograms, boxplots, correlation heatmaps |
| Mod 4 | Dimensionality & Time-Series | PCA-based Environmental Stress Index, STL decomposition |
| Mod 5 | Advanced Visualization | Parallel coordinates, missingness matrix, automated HTML/PDF reports |
