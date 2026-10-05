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
  last_run      TIMESTAMPTZ,
  is_synthetic  BOOLEAN        -- generated, not measured (migration 0004)
)

observations (
  id            BIGSERIAL PRIMARY KEY,
  source_id     INT REFERENCES data_sources(id),   -- who wrote last
  provenance    TEXT,          -- measured | synthetic (migration 0005)
  timestamp     TIMESTAMPTZ,   -- always UTC
  lat           DOUBLE PRECISION,   -- decimal degrees
  lon           DOUBLE PRECISION,
  pm25          DOUBLE PRECISION,
  pm10          DOUBLE PRECISION,
  temp          DOUBLE PRECISION,
  humidity      DOUBLE PRECISION,
  traffic_score DOUBLE PRECISION,
  is_anomaly    BOOLEAN DEFAULT FALSE,
  UNIQUE (provenance, timestamp, lat, lon)
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
  actual_value    DOUBLE PRECISION,  -- backfilled for drift monitoring
  lat             DOUBLE PRECISION,  -- which station it was for (task 9.6)
  lon             DOUBLE PRECISION
)
```

**Indexing:** composite index on `observations(timestamp, lat, lon)` drives both time-series windows and map queries. PostGIS geometry derived from `lat`/`lon` for district joins against the GeoJSON boundaries.

**The unique key is provenance, not source.** Three live feeds describe one
station-hour and each fills a different part of it, so keying on `source_id`
gave three rows of one column each and no row that was the joined state the
table exists to hold. Keying on provenance lets them merge while keeping the
demo bundle — same centroids, same hours — out of rows presented as
measurement. See [`docs/observation-merge.md`](./docs/observation-merge.md).

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
| `POST` | `/eda/scatter` | `x`, `y`, optional `color_by` | Sampled points, OLS line, r / ρ / r², n — §12.1 |
| `POST` | `/eda/grouped` | `measure`, `group_by`, optional `split_by` | Per-cell mean ± 95% CI and box summaries — §12.1 |
| `POST` | `/eda/pairplot` | optional `color_by` | Sampled rows of every measurement, band per row — §12.1 |
| `POST` | `/eda/andrews` | `class_by` | `t` grid, sampled curves, exact per-class mean curves — §12.1 |
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
| **EDA Studio** | Missingness matrix · Plotly histograms · interactive correlation heatmap · parallel coordinates · STL decomposition overlays (trend / seasonal / residual) · *Phase 12:* scatter plot · grouped bar chart · grouped boxplot · pair plot · Andrews curves (§12.1) |
| **Model Lab** | Trained-model table · hyperparameter configurations · MAE / RMSE / R² metrics · feature-importance bar charts |

**Design notes**

- A single typed API client in `src/services/` is the only place `fetch` appears — pages consume typed hooks, so an API change surfaces as a compile error rather than a runtime blank chart.
- Plotly handles statistical charts (histograms, heatmaps, parallel coordinates); Leaflet handles geospatial. No overlap.
- The **ethics disclaimer is a persistent UI element**, not a footnote:
  > "Predictions rely on sensor placement which may exhibit geographic/socioeconomic bias. Correlation shown does not equal causation."

### 12.1 Visual EDA — Bivariate & Multivariate Plots *(extension, Phase 12)*

Implements specs §6.4 (VIZ-1 … VIZ-6). The current EDA Studio describes pairs
of columns with **coefficients** (the heatmap) and many columns at once with
**projections** (parallel coordinates, t-SNE). It never shows a raw two-column
relationship or a comparison between groups. These five charts add that.

**Where the work lives.** This follows the existing EDA layering, nothing new:

| Layer | File | Responsibility |
|-------|------|----------------|
| Computation | `services/eda/visual.py` | Pure functions: frame in, frozen dataclass with `as_dict()` out. No I/O. |
| Service | `services/eda/service.py` | Resolve scope, load the frame, derive groupings, cache, call `visual.py` |
| Route | `api/routes/eda.py` | Pydantic request/response, one service call, no analytics |
| UI | `frontend/src/pages/EdaStudio.tsx` | Draws the payload with Plotly, computes nothing |

The browser could draw a scatter from raw rows, but then the statistics
would be computed in two places, and the browser has a 31,680-row frame
to handle. Computing them on the server means each number has one
definition, can be tested offline, and can be reused by the HTML report
(task 12.10).

**Shared mechanics**

- **Scope.** Each service function calls `datasets.resolve_source_ids` first
  and passes the result to the query, the cache key and `describe_window`
  (see `docs/observation-merge.md`). This is the rule the two dashboard
  endpoints broke before.
- **Statistics on every row, points from a sample.** r, OLS coefficients,
  group means, CIs and quartiles use every complete row. Only the points sent
  to the browser are thinned, using `manifold.subsample`, the same even,
  deterministic stride t-SNE uses. A random sample would redraw the chart on
  every refresh. Each payload reports `rows_used`, `points_returned` and
  `sampled`.
- **Groupings use the Phase 5 transformer.** `hour_of_day`, `day_of_week`,
  `is_weekend` and `month` come from `services/features/temporal.py`, so
  "09:00" means 09:00 IST everywhere in the system. A separate `.dt.hour` on
  UTC timestamps would shift every diurnal peak by 5½ hours.
  `time_of_day` buckets `hour_of_day` into four six-hour bands.
- **PM2.5 bands defined once.** The CPCB bands (Good ≤ 30, Satisfactory ≤ 60,
  Moderate ≤ 90, Poor ≤ 120, Very poor ≤ 250, Severe) currently exist only in
  `frontend/src/charts/theme.ts`. Grouping and colouring need them on the
  server, so they move to `services/eda/bands.py`. A test parses `theme.ts`
  and fails if the two copies disagree; that is cheaper than generating the
  TypeScript, and the edges almost never change.
- **Anomalies drawn, not dropped.** Each point carries `is_anomaly`, and the
  UI gives flagged points a distinct marker. Removing them would hide the
  outliers that Slide 7 is supposed to show (AC-5).
- **Caching** goes through the existing profile cache, keyed on dataset
  version, scope and the request parameters.
- **Caveats in the payload.** Each response has a `caveats` list (association
  ≠ causation; plus the chart-specific ones below).

**VIZ-1 — Scatter plot (`/eda/scatter`)**

- `x`, `y` ∈ measurement columns, `x ≠ y`. Optional `color_by` ∈ groupings.
- Computed over all complete (x, y) rows: OLS slope and intercept
  (`numpy.polyfit`, degree 1), Pearson r, Spearman ρ, r², n.
  Pearson and Spearman come from the `profile.py` functions, so a scatter's
  r always equals the heatmap cell for the same pair.
- Points: at most **2,000**, drawn as `scattergl`. The trend line is sent as
  its two end points, not as a fitted value per point.
- Caveat: r measures straight-line association only. A curve or a cluster
  can give a small r alongside a strong relationship, which is why the
  points are shown, not just r.

**VIZ-2 — Grouped bar chart and VIZ-3 — Grouped boxplot (`/eda/grouped`)**

Both charts group by the same columns, so one endpoint computes both from a
single `groupby`.

- `measure` ∈ measurement columns; `group_by` ∈ groupings; optional
  `split_by` ∈ groupings, `≠ group_by` (the bar chart's second level, e.g.
  station × time of day).
- **Bars:** per cell, mean, standard deviation, n, and a 95% CI from the
  t-distribution (`scipy.stats.t.ppf(0.975, n − 1) · s/√n`). Cells with
  n < 30 are flagged `thin` and the UI fades them: their interval is too
  wide to compare against.
- **Boxes** (by `group_by` only): q1, median, q3, whisker fences (the most
  extreme values within 1.5·IQR), the outlier count, and at most 50 of the
  most extreme outlier values per group. Plotly's `box` trace takes
  precomputed `q1`/`median`/`q3`/`lowerfence`/`upperfence`, so the full
  distribution is never sent.
- Caveat: these whiskers are **per group and for display only**. The quality
  engine's `is_anomaly` flag is per station and needs two of three
  detectors to agree, so a point can sit past a whisker without being
  flagged. The payload reports both counts so the two are not confused.

**VIZ-4 — Pair plot (`/eda/pairplot`)**

- Every measurement against every other: a 5 × 5 matrix, drawn with Plotly
  `splom`, lower triangle only (the upper triangle repeats it).
- At most **1,500** rows (`manifold.DEFAULT_MAX_POINTS`). That is 15,000
  marks across ten panels, near the limit of what `splom` redraws smoothly.
- Coloured by `pm25_band` by default; `color_by` accepts any grouping.
- The diagonal is hidden. Each column's histogram is already on the
  Distribution card. Each panel's tooltip shows the r for its pair from
  `/eda/profile`, so the pair plot and heatmap match cell for cell.
- Caveat: the pair plot is a sample; the coefficients are not.

**VIZ-5 — Andrews curves (`/eda/andrews`)**

Each row *x* (standardised) becomes a curve over *t* ∈ [−π, π]:

```
f_x(t) = x₁/√2 + x₂·sin t + x₃·cos t + x₄·sin 2t + x₅·cos 2t
```

- **Standardise first** (z-score on every row of the scope). Unstandardised,
  PM10 has the largest numbers and would set every curve's shape.
- **Column order is fixed and returned**: `pm25, pm10, traffic_score, temp,
  humidity`. Andrews curves depend on the order: early columns get the
  low-frequency terms, which shape the curve most. Ordering by PC1 loading
  would change the picture every time the ESI model is refitted, with no
  sign of why.
- `class_by` ∈ {`time_of_day` (default), `pm25_band`, `is_weekend`, `month`}.
  The default is not `pm25_band`: classes defined by PM2.5 would separate
  on the PM2.5 term by construction, so separation would show nothing new.
- **Curves:** `t` is a 128-point grid. At most **60 curves per class**, sampled
  evenly within each class, so a rare class such as *Severe* is not hidden
  under the common ones.
- **Mean curves are exact.** *f* is linear in *x*, so the mean curve of a
  class equals the curve of its mean vector. Computing it from every row
  costs one `groupby().mean()`, and it is the statement the slide relies
  on, not the sample.
- Caveat: separation depends on column order, and the curves show
  similarity between whole rows, not the effect of any one variable.

**Rendering**

- New EDA Studio cards: **Bivariate** (scatter; grouped bars; grouped
  boxplot) and, beside parallel coordinates, **Pair plot** and **Andrews
  curves**. Each has a column/grouping selector.
- Each `<Plot>` takes the required `ariaLabel`. Band colours come from
  `AQI_BANDS`, and the other series use the validated palette, so no meaning
  is carried by colour alone: bars and boxes are labelled with their group,
  and flagged points use a different marker shape as well as a different
  colour.

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
| One provenance per request, chosen explicitly | Pooling a seeded demo bundle with live readings produces statistics that describe neither, and the alternative — switching automatically once real data appears — silently redraws every chart the instant the first ingest lands. |
| Merge a station-hour across feeds, keyed on provenance | The table claims to hold the joined environmental state; keyed on `source_id` it held a third of it, and no live row carried both PM2.5 and traffic. |
| Per-field source authority | `temp` arrives from two feeds that disagree. Declaring who owns a field makes the merged value independent of ingestion order, rather than decided by tuple position in the adapter registry. |
| Visual EDA statistics computed on the server, drawn in the browser | One definition for each number: a scatter's r equals the heatmap cell, it is testable offline, and the report reuses it. The browser receives a sample of points, never the 31,680-row frame. |
| Andrews curves on a fixed column order, classed by time of day | Order changes the picture, so it must not move when the ESI refits; classing by PM2.5 band would separate on the PM2.5 term by construction. |
| Retention deletes artifacts, never rows or measurements | `predictions.model_id` is `ON DELETE CASCADE`, so reclaiming a file by dropping its registry row would take the drift dataset with it. Observations cost under 100 MB a year and are flag-never-delete by policy (AC-4, AC-5). |

---

## 15. Syllabus Traceability (BACSE301)

| Module | Topic | Where it lives in this design |
|--------|-------|-------------------------------|
| Mod 1 | Data Collection & Structure | §6.2–6.3 ingestion, harmonization, validation, DB storage |
| Mod 2 | Data Preprocessing | §7 MICE, Z-score, Isolation Forest, scaling |
| Mod 3 | Descriptive Stats & Visualization | §11 `/eda/profile`, §12 histograms, boxplots, correlation heatmaps; §12.1 scatter, grouped bars, grouped boxplots |
| Mod 4 | Dimensionality & Time-Series | §9 PCA/ESI, §12 STL decomposition, §10 time-aware modelling |
| Mod 5 | Advanced Visualization | §12 parallel coordinates, missingness matrix, automated HTML/PDF reports; §12.1 pair plot, Andrews curves |
