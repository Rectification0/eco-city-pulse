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
| API docs (Swagger) | <http://localhost:8000/docs> |
| PostgreSQL (container) | `localhost:5433` |

> The database container publishes on **5433**, not 5432, so it does not collide
> with a PostgreSQL already installed on the machine. Inside the compose network
> the backend still reaches it on 5432.

**No API keys are required.** The platform defaults to `INGESTION_MODE=demo` and runs entirely offline — a hard requirement from the spec (DR-1), not a convenience.

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

**Layering rule:** `backend/api/routes/*` contain no analytics logic. They validate input, call a service, and shape the response. All pandas/scikit-learn work lives in `backend/services/`.

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
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api to localhost:8000
```

### Tests

```bash
cd backend && pytest
```

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
│   │   ├── routes/              # Thin HTTP layer
│   │   └── dependencies.py      # Settings (and, from Phase 1, DB session) injection
│   ├── core/
│   │   ├── config.py            # Pydantic settings — the only reader of env secrets
│   │   └── exceptions.py        # Domain exceptions → HTTP envelope
│   ├── services/                # Ingestion, EDA, ML (Phase 2 onward)
│   ├── tests/
│   ├── main.py                  # App factory: CORS, routers, handlers
│   └── requirements.txt
├── data/
│   ├── raw/                     # Immutable landing zone
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

---

## Implementation status

| Phase | Scope | Status |
|-------|-------|--------|
| **0** | Project foundation — containers, config, app factory, health, tests | ✅ Complete |
| 1 | Data layer & persistence | ⬜ |
| 2 | Ingestion engine (FEAT-01) | ⬜ |
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
