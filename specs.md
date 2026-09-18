# Eco-City Pulse — Requirements Specification (`specs.md`)

> Derived from **Eco_City_Pulse_Specification.pdf** — Master Functionality & Requirements Specification.
> Companion documents: [`design.md`](./design.md) (architecture & technical design), [`tasks.md`](./tasks.md) (execution plan).

---

## 1. Overview

**Eco-City Pulse** is an end-to-end urban environmental intelligence platform that ingests, processes, analyzes, and predicts urban environmental conditions. It serves as a B.Tech CS/Data Science capstone demonstrating mastery of the **BACSE301 Exploratory Data Analysis** syllabus while extending into a full-stack ML application.

The platform is explicitly **not "just an ML model."** It is a unified pipeline:

```
Data Collection → Data Engineering → EDA → Statistical Analysis → Visualization
  → Feature Engineering → ML → Evaluation → Explainability → Prediction → Decision Support
```

### 1.1 Problem Statement

Urban environments suffer from rapidly fluctuating pollution levels, traffic congestion, and micro-climatic changes. Existing tools either provide **raw, uncontextualized data** (simple API dashboards) or **opaque predictions** (black-box ML models). There is a critical need for an interpretable platform that not only forecasts environmental conditions but strictly applies Exploratory Data Analysis to uncover the *why* behind the predictions.

### 1.2 Motivation

To transition academic data science concepts — missing value imputation, outlier detection, statistical distributions, dimensionality reduction, and predictive modeling — into a robust, interactive, real-world software engineering product.

### 1.3 Vision

Build the most academically rigorous and technically robust version of a **modular monolith** data science platform. Eco-City Pulse ingests heterogeneous data, harmonizes it, exposes it to automated and manual EDA, and powers an interpretable ML engine to predict **PM2.5** levels and generate an **Environmental Stress Index (ESI)**.

---

## 2. Objectives

| # | Objective |
|---|-----------|
| OBJ-1 | Implement a resilient data ingestion and harmonization engine (API + CSV fallback). |
| OBJ-2 | Build an automated and interactive EDA studio demonstrating syllabus-prescribed statistical techniques. |
| OBJ-3 | Develop an interpretable ML pipeline predicting future PM2.5 concentrations. |
| OBJ-4 | Deliver a production-quality frontend for exploratory visualization and decision support. |
| OBJ-5 | Ensure strict data leakage prevention and robust MLOps practices. |

---

## 3. Scope

### 3.1 In Scope

- Multi-source data ingestion (live APIs + CSV/JSON uploads + offline fallback)
- Automated data cleaning — MICE imputation, IQR / Z-score outlier detection
- Feature engineering (temporal, lag, rolling, transformations)
- Automated EDA reports
- Dimensionality reduction (PCA, t-SNE)
- Time-series analysis — ARIMA/Prophet baseline + XGBoost / Random Forest
- Spatial mapping
- REST API backend
- Interactive frontend

### 3.2 Out of Scope

- Distributed streaming architecture (Kafka)
- Kubernetes clusters
- Real-time millisecond latency requirements
- Native mobile applications
- **Causal inference** — only correlation will be claimed

---

## 4. Target Users & Personas

| Role | Capabilities | Primary Persona |
|------|--------------|-----------------|
| **General User** | View dashboards, check predictions, explore basic maps, read reports. | **Alex**, a concerned citizen planning outdoor activities based on air quality. |
| **Analyst / Data Scientist** | Run deep EDA, trigger preprocessing pipelines, evaluate metrics, configure SHAP. | **Dr. Lin**, an academic researching traffic impact on PM2.5. |
| **Administrator** | Manage API credentials, monitor ingestion logs, configure thresholds. | **Sam**, the system owner maintaining data freshness and backend health. |

---

## 5. Data Requirements

### 5.1 Data Sources

| Domain | Fields | Source |
|--------|--------|--------|
| **Air Quality** | AQI, PM2.5, PM10, NO2, SO2, CO, O3 | AQICN API / Kaggle Historical |
| **Weather** | Temperature, Humidity, Wind Speed, Wind Direction, Precipitation | OpenWeather API |
| **Traffic (Proxy)** | Traffic density score / congestion indices | TomTom API or synthetic baseline |
| **Spatial Boundaries** | GeoJSON of city districts | Static GeoJSON |

### 5.2 Ingestion & Harmonization Requirements

- **DR-1:** The system **MUST** operate without live APIs. Supported modes:
  - **Scheduled** (Cron)
  - **Manual** (Admin trigger)
  - **Upload** (CSV)
  - **Demo** (pre-loaded DB)
- **DR-2:** All timestamps converted universally to **UTC**.
- **DR-3:** All locations standardized to **Decimal Degrees**.
- **DR-4:** Data resampled to **hourly granularity** for time-series alignment.

### 5.3 Data Quality Engine (Syllabus Module 2)

**Missing Values**

- Detect missingness mechanism: **MCAR, MAR, MNAR**.
- Resolve using **MICE** (Multivariate Imputation by Chained Equations) for complex features.
- Use **forward / backward fill** for contiguous gaps.

**Outliers**

- Detect using **IQR** (boxplot bounds), **Z-Score (> 3)**, and **Isolation Forest**.
- **Action:** flag as `is_anomaly`. **Do NOT auto-delete** — an outlier could be a valid pollution spike.

---

## 6. Intelligence & Analysis Requirements

### 6.1 Feature Engineering

| Category | Features |
|----------|----------|
| **Temporal** | `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `season` |
| **Lag** | `PM2.5_lag_1h`, `PM2.5_lag_24h`, `Temp_lag_3h` |
| **Rolling Stats** | `PM2.5_rolling_mean_24h`, `traffic_rolling_std_6h` |
| **Transformations** | Log transformation on highly skewed pollutants (e.g., CO, SO2) |

### 6.2 Dimensionality Reduction & ESI

- **PCA** constructs the **Environmental Stress Index (ESI)**:
  1. Standardize continuous variables
  2. Extract **Principal Component 1 (PC1)**
  3. Normalize the score to **0–100**
- **t-SNE** used **exclusively** in the EDA Studio for 2D clustering visualization.

### 6.3 Machine Learning Pipeline

- **Target:** predict continuous PM2.5 levels **1, 6, and 24 hours** into the future (regression).
- **Models evaluated:** Naive Lag-1 (baseline), Ridge Regression, Random Forest, XGBoost.
- **Pipeline:**

```
Raw Data → Impute/Clean → Time-Aware Split → Feature Selection → Scale/Encode
  → Train Models → Cross-Validation → Model Registry
```

**Leakage Safeguards (mandatory):**

- Strict **temporal** splitting (no random shuffle).
- Scalers fit **only** on the training set.
- Lag features computed **before** splitting; target variable shifted correctly.

---

## 7. Core Feature Specifications

| Feature ID | Feature Name | Inputs | Processing / Outputs | Priority |
|-----------|--------------|--------|----------------------|----------|
| **FEAT-01** | Ingestion Engine | API configs, CSV | Validate schema, normalize, write to DB. **Output:** ingestion log. | Must Have |
| **FEAT-02** | Automated Profile | Dataset ID | Calculates mean, median, IQR, missingness. **Output:** JSON stats. | Must Have |
| **FEAT-03** | MICE Imputation | Raw DataFrame | Applies `IterativeImputer`. **Output:** clean DataFrame. | Must Have |
| **FEAT-04** | ESI Generator | Scaled Data | Applies PCA, extracts PC1. **Output:** 0–100 score. | Should Have |
| **FEAT-05** | Model Trainer | Target, Features | Time-split, train XGBoost, log metrics. **Output:** model artifact. | Must Have |
| **FEAT-06** | Predict Endpoint | Location, Time | Fetches lag features, runs inference. **Output:** predicted PM2.5. | Must Have |

---

## 8. API Specification

**Base URL:** `/api/v1`

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/data/sources` | List data sources and ingestion health. |
| `POST` | `/eda/profile` | Generate univariate and bivariate statistics. |
| `POST` | `/eda/reduce` | Perform dimensionality reduction (PCA). |
| `POST` | `/ml/predict` | Request a PM2.5 prediction. |

**Example `POST /ml/predict` response:**

```json
{
  "prediction": 45.2,
  "unit": "ug/m3",
  "confidence_interval": [38.5, 51.9],
  "top_features": { "lag_24_pm25": 0.45, "wind_speed": 0.22 }
}
```

---

## 9. Database Requirements (PostgreSQL)

| Table | Columns |
|-------|---------|
| `data_sources` | `id`, `name`, `api_url`, `status`, `last_run` |
| `observations` | `id`, `source_id`, `timestamp`, `lat`, `lon`, `pm25`, `pm10`, `temp`, `humidity`, `traffic_score`, `is_anomaly` |
| `models` | `id`, `name`, `target`, `features_used`, `mae`, `rmse`, `r2`, `created_at`, `artifact_path` |
| `predictions` | `id`, `model_id`, `target_time`, `predicted_value`, `actual_value` |

PostGIS extension required for spatial queries.

---

## 10. Frontend Requirements

**Stack:** React, TypeScript, Vite, Tailwind CSS, Plotly.js, React-Leaflet.

| Screen | Requirements |
|--------|--------------|
| **Overview Dashboard** | Hero KPIs (current ESI, PM2.5); Leaflet map showing spatial pollution gradients; 24-hour prediction trendline. |
| **EDA Studio** | Missingness matrix; Plotly histograms; interactive correlation heatmap; parallel coordinates; STL decomposition overlays. |
| **Model Lab** | Table of trained models; hyperparameter configurations; metrics; feature importance bar charts. |

---

## 11. Security, Privacy & Ethics Requirements

- **SEC-1:** All API requests validated via **Pydantic**.
- **SEC-2:** Secrets supplied through **environment variables** only.
- **SEC-3:** **CORS restricted** to the frontend origin.
- **PRIV-1:** Only **public** environmental data utilized.
- **PRIV-2:** Text analysis on social data strictly aggregates metrics — **no PII**.
- **ETH-1:** The UI **must explicitly state**:
  > "Predictions rely on sensor placement which may exhibit geographic/socioeconomic bias. Correlation shown does not equal causation."

---

## 12. Deployment Requirements

**Modular Monolith via Docker Compose.**

| Container | Contents |
|-----------|----------|
| Container 1 | React/Vite frontend served by **Nginx** |
| Container 2 | **FastAPI** Python backend |
| Container 3 | **PostgreSQL + PostGIS** extension |

---

## 13. Academic / Syllabus Mapping (BACSE301)

| Module | Topic | Eco-City Pulse Implementation |
|--------|-------|-------------------------------|
| **Mod 1** | Data Collection & Structure | Multi-source API & CSV ingestion, JSON validation, DB storage. |
| **Mod 2** | Data Preprocessing | MICE imputation, Z-score / Isolation Forest anomaly detection, scaling. |
| **Mod 3** | Descriptive Stats & Visualization | Automated EDA dashboard (histograms, boxplots, correlation heatmaps). |
| **Mod 4** | Dimensionality & Time-Series | PCA for Environmental Stress Index, STL decomposition for temporal trends. |
| **Mod 5** | Advanced Visualization | Parallel coordinates, missingness matrix, automated HTML/PDF reports. |

---

## 14. Acceptance Criteria

| ID | Criterion |
|----|-----------|
| AC-1 | The full stack starts with a single `docker-compose up` and the frontend reaches the backend. |
| AC-2 | Ingestion succeeds in **Demo mode with all live APIs disabled**. |
| AC-3 | `POST /eda/profile` returns mean, median, IQR, and missingness for every numeric column. |
| AC-4 | MICE imputation produces a DataFrame with **zero** nulls in the modelled feature set. |
| AC-5 | Outliers are **flagged** (`is_anomaly = true`), never deleted. |
| AC-6 | ESI is returned on a **0–100** scale derived from PCA PC1. |
| AC-7 | Trained XGBoost model beats the **Naive Lag-1 baseline** on MAE for the 1h horizon. |
| AC-8 | Temporal split verified — no training timestamp is later than any test timestamp. |
| AC-9 | `POST /ml/predict` returns prediction, unit, confidence interval, and top features. |
| AC-10 | All three frontend screens render with real backend data. |
| AC-11 | The bias/causation disclaimer is visible in the UI. |
