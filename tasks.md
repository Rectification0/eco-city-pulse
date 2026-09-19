# Eco-City Pulse — Implementation Tasks (`tasks.md`)

> Execution plan derived from **Eco_City_Pulse_Specification.pdf**.
> Requirements: [`specs.md`](./specs.md) · Design: [`design.md`](./design.md)
>
> Every task carries the requirement (`FEAT-*`, `DR-*`, `SEC-*`, `AC-*`, `OBJ-*`) it satisfies.

**Legend:** `[ ]` not started · `[~]` in progress · `[x]` done
**Priority:** **M** = Must Have · **S** = Should Have

---

## Phase 0 — Project Foundation

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 0.1 | Create repository skeleton exactly per the specified directory structure (`frontend/`, `backend/`, `data/raw`, `data/processed`) | design §5 | M | [x] |
| 0.2 | Write `docker-compose.yml` with three services: `frontend` (Nginx), `backend` (FastAPI), `db` (PostgreSQL + PostGIS) | AC-1 | M | [x] |
| 0.3 | Backend `Dockerfile` + `requirements.txt` (fastapi, uvicorn, pydantic, sqlalchemy, psycopg, pandas, numpy, scikit-learn, xgboost, statsmodels, shap, prophet) | AC-1 | M | [x] |
| 0.4 | Frontend scaffold — Vite + React + TypeScript + Tailwind; multi-stage Dockerfile building to Nginx | AC-1 | M | [x] |
| 0.5 | `backend/core/config.py` — Pydantic settings reading all secrets from env; `.env.example` committed, `.env` gitignored | SEC-2 | M | [x] |
| 0.6 | `backend/main.py` app factory — CORS restricted to frontend origin, router registration, `/health` endpoint | SEC-3 | M | [x] |
| 0.7 | `backend/core/exceptions.py` — domain exceptions and handlers mapping them to HTTP responses | design §11 | M | [x] |
| 0.8 | `README.md` — one-command startup instructions and architecture summary | — | M | [x] |
| 0.9 | Pytest harness + CI-ready test command; `backend/tests/` bootstrapped | OBJ-5 | S | [x] |

**Exit criteria:** `docker-compose up` brings all three containers healthy; frontend reaches `/api/v1/health`. *(AC-1)*

---

## Phase 1 — Data Layer & Persistence

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 1.1 | Enable PostGIS extension in DB init script | design §6.1 | M | [x] |
| 1.2 | Define `data_sources` table/model — `id, name, api_url, status, last_run` | specs §9 | M | [x] |
| 1.3 | Define `observations` table/model — `id, source_id, timestamp, lat, lon, pm25, pm10, temp, humidity, traffic_score, is_anomaly` | specs §9 | M | [x] |
| 1.4 | Define `models` table/model — `id, name, target, features_used, mae, rmse, r2, created_at, artifact_path` | specs §9 | M | [x] |
| 1.5 | Define `predictions` table/model — `id, model_id, target_time, predicted_value, actual_value` | specs §9 | M | [x] |
| 1.6 | Migrations (Alembic) + composite index on `observations(timestamp, lat, lon)` | design §6.1 | M | [x] |
| 1.7 | `api/dependencies.py` — DB session dependency and settings injection | design §5 | M | [x] |
| 1.8 | Load city-district **GeoJSON** into `data/raw/` and expose it to the API | specs §5.1 | S | [x] |
| 1.9 | Seed script producing the **Demo mode** dataset (offline historical air quality + weather + synthetic traffic) | DR-1, AC-2 | M | [x] |

**Exit criteria:** all four tables exist with constraints and indexes; demo seed loads without a network connection.

---

## Phase 2 — Ingestion Engine (FEAT-01)

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 2.1 | Source adapter interface; concrete adapters: **AQICN** (air quality), **OpenWeather** (weather), **TomTom** (traffic) | specs §5.1 | M | [x] |
| 2.2 | Synthetic traffic-score generator as the fallback when no traffic API key is configured | specs §5.1 | M | [x] |
| 2.3 | CSV / JSON upload parser with per-source Pydantic schema validation | FEAT-01, SEC-1 | M | [x] |
| 2.4 | Harmonizer — all timestamps → **UTC** | DR-2 | M | [x] |
| 2.5 | Harmonizer — all coordinates → **decimal degrees** | DR-3 | M | [x] |
| 2.6 | Harmonizer — resample all sources to **hourly** granularity (mean within hour) | DR-4 | M | [x] |
| 2.7 | `ingestion_service.py` — single write path to `observations`, with quarantine for malformed records | FEAT-01 | M | [x] |
| 2.8 | Ingestion log + `data_sources.status` / `last_run` updates; failed live fetch degrades source to `offline` without failing the request | FEAT-01, DR-1 | M | [x] |
| 2.9 | Implement all four modes: **Scheduled (cron)**, **Manual (admin trigger)**, **Upload (CSV)**, **Demo (pre-loaded)** | DR-1 | M | [x] |
| 2.10 | `GET /api/v1/data/sources` — sources with ingestion health | specs §8 | M | [x] |
| 2.11 | Tests: schema rejection, UTC conversion, hourly resample correctness, offline degradation | AC-2 | M | [x] |

**Exit criteria:** ingestion completes end-to-end in Demo mode with **every live API disabled**. *(AC-2)*

---

## Phase 3 — Data Quality Engine (Syllabus Module 2)

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 3.1 | Missingness analyzer — per-column counts/percentages plus **MCAR / MAR / MNAR** characterization | specs §5.3 | M | [x] |
| 3.2 | **MICE** imputation via scikit-learn `IterativeImputer` for complex multivariate gaps | FEAT-03, AC-4 | M | [x] |
| 3.3 | Forward / backward fill for contiguous short gaps | specs §5.3 | M | [x] |
| 3.4 | **IQR** outlier detector (boxplot bounds `Q1 − 1.5·IQR`, `Q3 + 1.5·IQR`) | specs §5.3 | M | [x] |
| 3.5 | **Z-Score** outlier detector (`|z| > 3`) | specs §5.3 | M | [x] |
| 3.6 | **Isolation Forest** multivariate anomaly detector | specs §5.3 | M | [x] |
| 3.7 | Combine detectors into the `is_anomaly` flag — **flag only, never delete** | AC-5 | M | [x] |
| 3.8 | Persist cleaned output to `data/processed/` | design §5 | S | [x] |
| 3.9 | Tests: zero nulls after MICE on the modelled feature set; anomalous rows **retained** and flagged | AC-4, AC-5 | M | [x] |

**Exit criteria:** cleaning pipeline yields a null-free feature set while preserving every anomalous record. *(AC-4, AC-5)*

---

## Phase 4 — EDA & Statistical Engine (FEAT-02)

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 4.1 | `eda_service.py` — univariate profile: mean, median, IQR, std, skewness, kurtosis, missingness | FEAT-02, AC-3 | M | [x] |
| 4.2 | Bivariate profile — correlation matrix (Pearson + Spearman) | FEAT-02 | M | [x] |
| 4.3 | `POST /api/v1/eda/profile` returning the full JSON stats payload | specs §8, AC-3 | M | [x] |
| 4.4 | Distribution/skewness analysis identifying pollutants needing log transform | specs §6.1 | M | [x] |
| 4.5 | **STL decomposition** (statsmodels) — trend / seasonal / residual series for PM2.5 | Mod 4 | M | [x] |
| 4.6 | Caching layer in front of profile computation, keyed by dataset version | design §11 | S | [x] |
| 4.7 | Automated **HTML / PDF** EDA report generation | Mod 5 | S | [x] |
| 4.8 | Tests: profile covers every numeric column; statistics verified against known fixtures | AC-3 | M | [x] |

**Exit criteria:** `/eda/profile` returns mean, median, IQR, and missingness for every numeric column. *(AC-3)*

---

## Phase 5 — Feature Engineering

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 5.1 | Temporal features — `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `season` | specs §6.1 | M | [x] |
| 5.2 | Lag features — `PM2.5_lag_1h`, `PM2.5_lag_24h`, `Temp_lag_3h` | specs §6.1 | M | [x] |
| 5.3 | Rolling statistics — `PM2.5_rolling_mean_24h`, `traffic_rolling_std_6h` | specs §6.1 | M | [x] |
| 5.4 | Log transformation on highly skewed pollutants (CO, SO2) | specs §6.1 | M | [x] |
| 5.5 | Single deterministic transformer shared by training **and** inference (no drift between paths) | design §8 | M | [x] |
| 5.6 | Feature-store persistence of engineered features | design §2 | S | [x] |
| 5.7 | Tests: lag correctness on a known series; no forward-looking leakage in rolling windows | AC-8 | M | [x] |

**Exit criteria:** feature set reproducible and byte-identical between training and inference paths.

---

## Phase 6 — Dimensionality Reduction & ESI (FEAT-04)

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 6.1 | Standardize continuous variables (`StandardScaler`) ahead of PCA | specs §6.2 | M | [x] |
| 6.2 | Fit PCA; persist explained variance ratios and component **loadings** | FEAT-04 | S | [x] |
| 6.3 | Extract **PC1** and normalize to a **0–100** ESI score | FEAT-04, AC-6 | S | [x] |
| 6.4 | `POST /api/v1/eda/reduce` — PCA components, explained variance, loadings | specs §8 | M | [x] |
| 6.5 | **t-SNE** 2D projection — **EDA Studio only**, never in an inference path | specs §6.2 | S | [x] |
| 6.6 | Expose PC1 loadings to the UI so ESI stays interpretable | design §9 | S | [x] |
| 6.7 | Tests: ESI bounded within 0–100; PCA reproducible under a fixed seed | AC-6 | S | [x] |

**Exit criteria:** ESI returned on a 0–100 scale derived from PCA PC1, with loadings inspectable. *(AC-6)*

---

## Phase 7 — ML Pipeline (FEAT-05)

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 7.1 | Define the three targets — PM2.5 at **+1h, +6h, +24h**; correct forward target shift | specs §6.3 | M | [x] |
| 7.2 | **Time-aware split** — strictly chronological, no shuffling | AC-8 | M | [x] |
| 7.3 | Scale/encode with scaler **fit on train only** | AC-8 | M | [x] |
| 7.4 | Feature selection stage | specs §6.3 | M | [x] |
| 7.5 | **Naive Lag-1** baseline model | AC-7 | M | [x] |
| 7.6 | **Ridge Regression** model | specs §6.3 | M | [x] |
| 7.7 | **Random Forest** model | specs §6.3 | M | [x] |
| 7.8 | **XGBoost** model — primary production model | FEAT-05 | M | [x] |
| 7.9 | **ARIMA / Prophet** classical time-series baselines | specs §3.1 | S | [x] |
| 7.10 | Cross-validation with `TimeSeriesSplit` / expanding window (never K-Fold) | AC-8 | M | [x] |
| 7.11 | Evaluation — MAE, RMSE, R² per model per horizon | FEAT-05 | M | [x] |
| 7.12 | Model registry — persist artifact to storage, write metrics + `artifact_path` to `models` | FEAT-05 | M | [x] |
| 7.13 | **Leakage audit test** — assert every training timestamp precedes every test timestamp; assert scaler never saw test data | AC-8, OBJ-5 | M | [x] |
| 7.14 | Baseline-beating test — XGBoost MAE < Naive Lag-1 MAE at the 1h horizon | AC-7 | M | [x] |

**Exit criteria:** models trained and registered; leakage audit passes; XGBoost beats the naive baseline. *(AC-7, AC-8)*

---

## Phase 8 — Explainability

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 8.1 | **SHAP** explainer over the trained XGBoost model | design §10.2 | M | [x] |
| 8.2 | Global feature importance surfaced for the Model Lab | specs §10 | M | [x] |
| 8.3 | Per-prediction `top_features` attribution for the predict response | FEAT-06 | M | [x] |
| 8.4 | SHAP configuration options exposed to the Analyst role | specs §4 | S | [x] |

**Exit criteria:** every prediction ships with its reasoning; no number appears in the UI unexplained.

---

## Phase 9 — Prediction Service (FEAT-06)

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 9.1 | Inference lag-feature fetcher — assemble features for a given location + time from `observations` | FEAT-06 | M | [x] |
| 9.2 | Load the registered model artifact and run inference | FEAT-06 | M | [x] |
| 9.3 | Confidence-interval estimation | specs §8 | M | [x] |
| 9.4 | `POST /api/v1/ml/predict` returning `prediction`, `unit`, `confidence_interval`, `top_features` | AC-9 | M | [x] |
| 9.5 | Persist each prediction to `predictions` | specs §9 | M | [x] |
| 9.6 | Backfill `predictions.actual_value` once the real observation arrives (drift monitoring input) | design §6.1 | S | [x] |
| 9.7 | Contract test on the exact response shape from the spec | AC-9 | M | [x] |

**Exit criteria:** predict endpoint matches the specified response contract exactly. *(AC-9)*

---

## Phase 10 — Frontend

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 10.1 | Typed API client in `src/services/` — the only place `fetch` appears | design §12 | M | [x] |
| 10.2 | App shell, routing, Tailwind theme | specs §10 | M | [x] |
| 10.3 | **Dashboard** — hero KPI tiles for current **ESI** and **PM2.5** | specs §10 | M | [x] |
| 10.4 | **Dashboard** — React-Leaflet map with spatial pollution gradients over district GeoJSON | specs §10 | M | [x] |
| 10.5 | **Dashboard** — 24-hour prediction trendline | specs §10 | M | [x] |
| 10.6 | **EDA Studio** — missingness matrix | Mod 5 | M | [x] |
| 10.7 | **EDA Studio** — Plotly histograms and boxplots | Mod 3 | M | [x] |
| 10.8 | **EDA Studio** — interactive correlation heatmap | Mod 3 | M | [x] |
| 10.9 | **EDA Studio** — parallel coordinates plot | Mod 5 | S | [x] |
| 10.10 | **EDA Studio** — STL decomposition overlays | Mod 4 | M | [x] |
| 10.11 | **EDA Studio** — t-SNE 2D cluster scatter | specs §6.2 | S | [x] |
| 10.12 | **Model Lab** — trained-model table with hyperparameters and MAE/RMSE/R² | specs §10 | M | [x] |
| 10.13 | **Model Lab** — feature-importance bar charts | specs §10 | M | [x] |
| 10.14 | Prediction form (location + time + horizon) wired to `/ml/predict` | FEAT-06 | M | [x] |
| 10.15 | Admin view — ingestion logs, source health, threshold configuration | specs §4 | S | [x] |
| 10.16 | Persistent ethics disclaimer: sensor-placement bias + correlation ≠ causation | ETH-1, AC-11 | M | [x] |
| 10.17 | Loading, empty, and error states for every data-backed view | — | S | [x] |

**Exit criteria:** all three screens render against real backend data with the disclaimer visible. *(AC-10, AC-11)*

---

## Phase 11 — Security, Quality & Release

| # | Task | Traces | Pri | Status |
|---|------|--------|-----|--------|
| 11.1 | Audit: every endpoint request/response is a Pydantic model | SEC-1 | M | [ ] |
| 11.2 | Audit: no secret in source, image, or compose file — env vars only | SEC-2 | M | [ ] |
| 11.3 | Audit: CORS allowlist limited to the frontend origin | SEC-3 | M | [ ] |
| 11.4 | Audit: only public environmental data ingested; text analysis aggregates only, **no PII** | PRIV-1, PRIV-2 | M | [ ] |
| 11.5 | Audit: no causal claim anywhere in copy, labels, or reports | ETH-1 | M | [ ] |
| 11.6 | Integration test — full path from ingestion through prediction in Demo mode | AC-1…AC-11 | M | [ ] |
| 11.7 | Syllabus traceability review — confirm Modules 1–5 are each demonstrably implemented | specs §13 | M | [ ] |
| 11.8 | Acceptance walkthrough against all of AC-1 … AC-11 | specs §14 | M | [ ] |
| 11.9 | Final documentation — README, API reference, EDA report artifacts | — | M | [ ] |

---

## Dependency Order

```
Phase 0 ─► Phase 1 ─► Phase 2 ─► Phase 3 ─► Phase 4
                                     │
                                     └─► Phase 5 ─► Phase 6 (ESI)
                                                └─► Phase 7 (ML) ─► Phase 8 ─► Phase 9
                                                                                 │
Phase 10 (frontend) ◄────── needs Phase 2/4/6/7/9 endpoints ─────────────────────┘
                                                                                 ▼
                                                                          Phase 11
```

Phase 10 can begin against mocked API responses as soon as the Phase 0 scaffold exists, then switch to live endpoints as each backend phase lands.

---

## Requirement Coverage Check

| Requirement | Covered by |
|-------------|-----------|
| FEAT-01 Ingestion Engine | Phase 2 |
| FEAT-02 Automated Profile | Phase 4 |
| FEAT-03 MICE Imputation | Phase 3 |
| FEAT-04 ESI Generator | Phase 6 |
| FEAT-05 Model Trainer | Phase 7 |
| FEAT-06 Predict Endpoint | Phase 9 |
| DR-1 … DR-4 (harmonization) | Phase 2 |
| SEC-1 … SEC-3, PRIV, ETH | Phase 0.5–0.6, Phase 10.16, Phase 11 |
| OBJ-5 Leakage prevention / MLOps | Phase 7.2, 7.3, 7.10, 7.13 |
| BACSE301 Mod 1–5 | Phases 2, 3, 4, 6, 10 |
| AC-1 … AC-11 | Phase exit criteria + Phase 11.8 |
