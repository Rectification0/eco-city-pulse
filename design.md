# Eco-City Pulse — Technical Design (`design.md`)

> Architecture and technical design derived from **Eco_City_Pulse_Specification.pdf**.
> Requirements: [`specs.md`](./specs.md) · Execution plan: [`tasks.md`](./tasks.md)

---

## 1. Architectural Style

**Modular Monolith**, deployed as three Docker Compose containers. Chosen deliberately over microservices: the spec puts distributed streaming (Kafka) and Kubernetes **out of scope**, and a monolith keeps the data-science pipeline in one process boundary where pandas DataFrames can move between stages without serialization cost.

Internally the backend is split into **service modules** (`ingestion_service`, `eda_service`, `ml_service`) with thin API routers on top — so modules stay independently testable and could later be extracted if needed.

---

## 2. Functional Architecture — Five Tiers

| Tier | Responsibility | Components |
|------|----------------|------------|
| **1. Source** | Acquire raw data | Live APIs (AQICN, OpenWeather, TomTom), CSV/JSON uploads, offline fallback files |
| **2. Ingestion & Quality** | Make data trustworthy | Schema validation, normalization, missingness detection, outlier flagging |
| **3. Intelligence** | Analyze and model | EDA engine, Statistical engine, Feature store, ML pipeline |
| **4. Service** | Expose capability | FastAPI REST endpoints, caching, report generation |
| **5. Presentation** | Human interaction | React SPA — interactive charts, maps, prediction forms |

---

## 3. System Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Data Sources (Live APIs · File Uploads · Static Offline)    │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Backend Core — FastAPI                                      │
│  Ingestion Router · EDA Service · ML Service · Reporting     │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Intelligence Engine                                         │
│  Pandas Preprocessor · Scikit-learn · XGBoost · Statsmodels  │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Data Persistence — PostgreSQL · PostGIS · Object Storage    │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Frontend — React                                            │
│  Dashboard UI · EDA Studio · Prediction Engine               │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Deployment Design

| Container | Image / Stack | Role | Exposes |
|-----------|---------------|------|---------|
| **frontend** | Node build → **Nginx** static serve | Serves the React/Vite SPA bundle; proxies `/api` to backend | `80` |
| **backend** | Python + **FastAPI** (Uvicorn) | REST API, EDA engine, ML training & inference | `8000` |
| **db** | **PostgreSQL + PostGIS** | Observations, model registry, predictions, spatial queries | `5432` |

Orchestrated by `docker-compose.yml`. Model artifacts and generated reports live in a mounted volume (object-storage-style path) referenced by `models.artifact_path`.

**Configuration contract:** every secret (API keys, DB URL) arrives via environment variable and is parsed once in `backend/core/config.py` as a Pydantic settings object. No secret is ever committed.

---

## 5. Directory Structure

```
eco-city-pulse/
├── frontend/
│   ├── src/
│   │   ├── components/          # Reusable chart/map/KPI components
│   │   ├── pages/               # Dashboard, EDA Studio, Model Lab
│   │   ├── services/            # Typed API client
│   │   └── App.tsx
│   ├── package.json
│   └── tailwind.config.js
├── backend/
│   ├── api/
│   │   ├── routes/              # data.py, eda.py, ml.py
│   │   └── dependencies.py      # DB session, auth, settings injection
│   ├── core/
│   │   ├── config.py            # Pydantic settings from env
│   │   └── exceptions.py        # Domain exceptions → HTTP mapping
│   ├── services/
│   │   ├── eda_service.py
│   │   ├── ml_service.py
│   │   └── ingestion_service.py
│   ├── tests/
│   ├── main.py                  # App factory, CORS, router registration
│   └── requirements.txt
├── data/
│   ├── raw/                     # Immutable landing zone
│   └── processed/               # Cleaned / feature-engineered outputs
├── docker-compose.yml
└── README.md
```

**Design rule:** `api/routes/*` contain no analytics logic — they validate input, call a service, and shape the response. All pandas/sklearn work lives in `services/`.

---

## 6. Data Design

### 6.1 Schema

```sql
data_sources (
  id            SERIAL PRIMARY KEY,
  name          TEXT,
  api_url       TEXT,
  status        TEXT,          -- healthy | degraded | offline
  last_run      TIMESTAMPTZ
)

observations (
  id            BIGSERIAL PRIMARY KEY,
  source_id     INT REFERENCES data_sources(id),
  timestamp     TIMESTAMPTZ,   -- always UTC
  lat           DOUBLE PRECISION,   -- decimal degrees
  lon           DOUBLE PRECISION,
  pm25          DOUBLE PRECISION,
  pm10          DOUBLE PRECISION,
  temp          DOUBLE PRECISION,
  humidity      DOUBLE PRECISION,
  traffic_score DOUBLE PRECISION,
  is_anomaly    BOOLEAN DEFAULT FALSE
)

models (
  id             SERIAL PRIMARY KEY,
  name           TEXT,
  target         TEXT,         -- e.g. pm25_h1 | pm25_h6 | pm25_h24
  features_used  JSONB,
  mae            DOUBLE PRECISION,
  rmse           DOUBLE PRECISION,
  r2             DOUBLE PRECISION,
  created_at     TIMESTAMPTZ,
  artifact_path  TEXT
)

predictions (
  id              BIGSERIAL PRIMARY KEY,
  model_id        INT REFERENCES models(id),
  target_time     TIMESTAMPTZ,
  predicted_value DOUBLE PRECISION,
  actual_value    DOUBLE PRECISION   -- backfilled for drift monitoring
)
```

**Indexing:** composite index on `observations(timestamp, lat, lon)` drives both time-series windows and map queries. PostGIS geometry derived from `lat`/`lon` for district joins against the GeoJSON boundaries.

`predictions.actual_value` is deliberately nullable — it is backfilled once the real observation for `target_time` arrives, which turns the table into a live model-monitoring dataset.

### 6.2 Harmonization Design

Every source passes through a normalizer before hitting `observations`:

1. **Schema validation** — Pydantic model per source; reject/quarantine malformed records into the ingestion log.
2. **Timestamp → UTC** — no naive datetimes ever persisted.
3. **Location → decimal degrees**.
4. **Resample → hourly** — mean aggregation within the hour, giving all sources one shared time grid so joins and lag features are well-defined.

### 6.3 Ingestion Modes

The system **must survive with zero live APIs**. Four modes share one code path, differing only in the source adapter:

| Mode | Trigger | Adapter |
|------|---------|---------|
| Scheduled | Cron | HTTP client against live APIs |
| Manual | Admin action | HTTP client against live APIs |
| Upload | Analyst uploads CSV | File parser |
| Demo | Default at boot | Pre-loaded DB / static offline files |

A failed live fetch degrades the source to `status = offline` in `data_sources` and the platform continues on the most recent persisted data — availability is never coupled to a third party.

---

## 7. Data Quality Engine Design

**Missing values**

- Characterize the mechanism (**MCAR / MAR / MNAR**) and surface that judgement in the profile output — it is the academic justification for the imputation choice, not an internal detail.
- **Contiguous short gaps** → forward/backward fill (time-series locality is the better signal).
- **Complex multivariate gaps** → **MICE** via scikit-learn `IterativeImputer`.

**Outliers**

Three detectors run in parallel and vote into a single `is_anomaly` flag:

| Detector | Rule |
|----------|------|
| IQR | outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` |
| Z-Score | `|z| > 3` |
| Isolation Forest | multivariate anomaly score |

**Critical design decision:** anomalies are **flagged, never deleted**. A PM2.5 spike may be a genuine pollution event — the most informative record in the dataset. Deletion would destroy exactly the signal the platform exists to explain.

---

## 8. Feature Engineering Design

Computed by a single deterministic transformer so training and inference build features identically.

| Group | Features |
|-------|----------|
| Temporal | `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `season` |
| Lag | `PM2.5_lag_1h`, `PM2.5_lag_24h`, `Temp_lag_3h` |
| Rolling | `PM2.5_rolling_mean_24h`, `traffic_rolling_std_6h` |
| Transform | log transform on highly skewed pollutants (CO, SO2) |

Cyclical temporal features are the reason the hourly resample matters: without a uniform grid, `lag_1h` is not a well-defined shift.

---

## 9. Environmental Stress Index (ESI) Design

ESI compresses many correlated pollutant/weather signals into one interpretable number:

1. **Standardize** all continuous variables (zero mean, unit variance) — PCA is scale-sensitive.
2. **Fit PCA**, extract **PC1** (the dominant axis of environmental variation).
3. **Normalize PC1 to 0–100** using train-set min/max, so the score is human-readable.

The PC1 **loadings** are retained and shown in the UI — they explain *which* variables drive stress, keeping the index interpretable rather than a black box.

**t-SNE is strictly EDA-only.** It has no stable out-of-sample transform, so it never feeds a model or the ESI — it exists purely for 2D cluster visualization in the EDA Studio.

---

## 10. ML Pipeline Design

```
Raw Data
   ↓ Impute / Clean            (MICE, ffill/bfill, anomaly flags retained)
   ↓ Time-Aware Split          (chronological; no shuffling)
   ↓ Feature Selection
   ↓ Scale / Encode            (scaler fit on TRAIN ONLY)
   ↓ Train Models              (Naive Lag-1, Ridge, Random Forest, XGBoost)
   ↓ Cross-Validation          (TimeSeriesSplit / expanding window)
   ↓ Model Registry            (metrics + artifact path → models table)
```

**Targets:** PM2.5 at **+1h, +6h, +24h** — three separate regression models, one per horizon.

**Model ladder (deliberate):**

| Model | Purpose |
|-------|---------|
| Naive Lag-1 | Honest baseline. Any model that cannot beat "tomorrow ≈ today" adds nothing. |
| Ridge Regression | Linear reference, regularized against collinear pollutants. |
| Random Forest | Non-linear, low-tuning benchmark. |
| XGBoost | Primary production model. |

ARIMA / Prophet serve as classical time-series baselines for comparison against the tree models.

### 10.1 Leakage Safeguards

Leakage is the dominant failure mode in time-series ML, so it is designed against explicitly:

| Safeguard | Implementation |
|-----------|----------------|
| **Temporal split** | Train strictly precedes test in time. Never `train_test_split(shuffle=True)`. |
| **Scaler discipline** | `fit` on train, `transform` on test. Fitting on the full set leaks future distribution. |
| **Lag ordering** | Lag/rolling features computed **before** splitting (so they are causally valid), while the **target is shifted forward correctly** — the model may never see a value at or after the time it predicts. |
| **CV strategy** | `TimeSeriesSplit` / expanding window, never K-Fold. |

### 10.2 Explainability

**SHAP** values expose per-prediction feature attribution; the `/ml/predict` response carries `top_features` so every number the UI shows comes with its reasoning. This is the design answer to the problem statement's complaint about opaque models.

---

## 11. API Design

**Base URL:** `/api/v1`

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| `GET` | `/data/sources` | — | Source list with `status`, `last_run` (ingestion health) |
| `POST` | `/eda/profile` | `dataset_id` | Univariate + bivariate statistics as JSON |
| `POST` | `/eda/reduce` | dataset + method | PCA (or t-SNE) components, explained variance, loadings |
| `POST` | `/ml/predict` | location, time, horizon | Prediction + interval + top features |

```json
// POST /ml/predict → 200
{
  "prediction": 45.2,
  "unit": "ug/m3",
  "confidence_interval": [38.5, 51.9],
  "top_features": { "lag_24_pm25": 0.45, "wind_speed": 0.22 }
}
```

**Design conventions**

- Every request body and response is a **Pydantic** model — validation is the security boundary (SEC-1).
- Analytics endpoints are `POST` because they carry structured configuration payloads, not simple lookups.
- **Caching** sits in front of expensive EDA/profile computation; keys include the dataset version so results never go stale silently.
- Domain errors raised in `services/` map to HTTP status codes in `core/exceptions.py`, keeping routers free of try/except noise.
- **CORS** allows only the frontend origin (SEC-3).

---

## 12. Frontend Design

**Stack:** React · TypeScript · Vite · Tailwind CSS · Plotly.js · React-Leaflet

| Page | Composition |
|------|-------------|
| **Overview Dashboard** | Hero KPI tiles (current **ESI**, **PM2.5**) · React-Leaflet map rendering spatial pollution gradients over district GeoJSON · 24-hour prediction trendline |
| **EDA Studio** | Missingness matrix · Plotly histograms · interactive correlation heatmap · parallel coordinates · STL decomposition overlays (trend / seasonal / residual) |
| **Model Lab** | Trained-model table · hyperparameter configurations · MAE / RMSE / R² metrics · feature-importance bar charts |

**Design notes**

- A single typed API client in `src/services/` is the only place `fetch` appears — pages consume typed hooks, so an API change surfaces as a compile error rather than a runtime blank chart.
- Plotly handles statistical charts (histograms, heatmaps, parallel coordinates); Leaflet handles geospatial. No overlap.
- The **ethics disclaimer is a persistent UI element**, not a footnote:
  > "Predictions rely on sensor placement which may exhibit geographic/socioeconomic bias. Correlation shown does not equal causation."

---

## 13. Security, Privacy & Ethics Design

| Concern | Design |
|---------|--------|
| Input validation | Pydantic schemas on every endpoint |
| Secrets | Environment variables → `core/config.py`; never in source or images |
| CORS | Restricted to the frontend origin |
| Data sensitivity | Public environmental data only |
| PII | Social/text analysis emits **aggregate metrics only** |
| Honesty | Bias + correlation-vs-causation disclaimer surfaced in the UI; **causal inference is out of scope** and no causal claim is made anywhere |

---

## 14. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| Modular monolith over microservices | Kafka/K8s are out of scope; DataFrames pass in-process with no serialization tax. |
| Flag anomalies, never delete | A pollution spike is signal, not noise — deleting it removes the very event worth explaining. |
| PCA PC1 for ESI (not a hand-weighted formula) | Weights are learned from data and the loadings remain inspectable. |
| Separate model per horizon (1h/6h/24h) | Drivers differ by horizon; one multi-output model would blur them. |
| Naive Lag-1 as mandatory baseline | Forces an honest claim of value from the ML stack. |
| t-SNE confined to EDA | No stable out-of-sample transform; unsafe in a production inference path. |
| Hourly resampling as a hard contract | Lag and rolling features are only well-defined on a uniform time grid. |
| Demo mode as the default | Guarantees the platform demonstrates end-to-end with zero API keys and zero network. |

---

## 15. Syllabus Traceability (BACSE301)

| Module | Topic | Where it lives in this design |
|--------|-------|-------------------------------|
| Mod 1 | Data Collection & Structure | §6.2–6.3 ingestion, harmonization, validation, DB storage |
| Mod 2 | Data Preprocessing | §7 MICE, Z-score, Isolation Forest, scaling |
| Mod 3 | Descriptive Stats & Visualization | §11 `/eda/profile`, §12 histograms, boxplots, correlation heatmaps |
| Mod 4 | Dimensionality & Time-Series | §9 PCA/ESI, §12 STL decomposition, §10 time-aware modelling |
| Mod 5 | Advanced Visualization | §12 parallel coordinates, missingness matrix, automated HTML/PDF reports |
