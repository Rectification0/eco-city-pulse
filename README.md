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

On first boot the backend applies migrations, loads the demo dataset, and runs
the data quality pipeline over it before it starts serving — so there is nothing
to run afterwards and the demo opens on cleaned, anomaly-flagged data. Cold
start takes a minute or two; `docker compose logs -f backend` shows the
progress.

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

## Data quality

BACSE301 Module 2 (specs §5.3, design §7). Repairs values and flags records --
and **never removes one**.

```
analyse missingness → fill short gaps → MICE → three detectors vote → flag → persist
```

```bash
curl -X POST localhost:8000/api/v1/data/quality | jq     # or:
cd backend && python -m scripts.run_quality
```

### What can and cannot be inferred about missingness

The requirement says "detect MCAR, MAR, MNAR". Two of those three are testable
and one is not, so the engine is explicit about which is which:

| Mechanism | How it is decided |
|-----------|-------------------|
| **MAR** | A rank test finds the column's missingness predictable from *observed* values of another variable (or from the hour of day). That is the condition under which MICE is the right repair. |
| **MCAR** | No such association survives a Bonferroni-corrected threshold and an effect-size floor. |
| **MNAR** | **Never inferred.** Missingness that depends on the *unobserved* value is not identifiable from the data — the evidence that would separate it from MCAR is precisely what is missing. It is accepted only as a declared domain rule with a stated justification. |

Every incomplete column carries the caveat that MNAR cannot be excluded. This
is the same discipline ETH-1 imposes on the UI, applied to the statistics.

The test is **Mann-Whitney U with an AUC effect size**, not a correlation. That
choice is load-bearing: missingness concentrated in a predictor's tail — a
sensor that saturates above some concentration — is a 20× effect that linear
correlation reads as r ≈ 0.01. An earlier point-biserial implementation
reported MCAR for exactly that mechanism.

Two things are reported alongside the verdict:

- **Grid completeness** — hours absent from the series entirely, as opposed to
  rows present with a null. Only the second shows up in a column null count, so
  reporting both prevents an "only 3% missing" claim about a feed that skipped
  a fortnight.
- **Co-missingness** — which columns go missing together. Four sensors dropping
  out in the same hours is one station outage, not four independent quirks.

### Imputation

| Gap | Repair | Why |
|-----|--------|-----|
| Contiguous, ≤ `max_gap_hours` (default 3) | Forward/backward fill | For an hour or two of a smooth physical series the neighbouring value beats any model |
| Everything else | **MICE** (`IterativeImputer`) | Conditions each column on the others — exactly right for a MAR mechanism |

Only *whole* short runs are filled. `ffill(limit=n)` would fill the first n
hours of a week-long outage, fabricating the start of a flat line; a run is
filled entirely or left to MICE.

Both stages run **per station**. A fill that crossed stations would carry one
sensor's reading into another's gap; a model fitted across the city's pollution
gradient would regress every station toward the mean.

Imputed values are bounded by the range actually observed — an unconstrained
linear model will happily predict a negative concentration — and the run is
deterministic, because AC-4's guarantee is worthless if the numbers change
between runs. Stations where the chained equations had not converged within the
iteration budget are **counted in the report** rather than warned about.

### Outliers

| Detector | Rule | Sees |
|----------|------|------|
| IQR | outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` | one column |
| Z-score | `\|z\| > 3` | one column |
| Isolation Forest | multivariate anomaly score | every column |

A row is flagged when **at least two of the three agree** (configurable). Three
choices keep the flag meaningful:

- **Bounds are fitted per station.** A city-wide IQR on a real pollution
  gradient flags the dirtiest district wholesale rather than any anomaly.
- **An imputed value is never flagged.** The detectors need a complete matrix,
  so they run after imputation — but a column only votes where the original
  reading was actually observed. Otherwise the engine would flag its own
  invention.
- **Isolation Forest uses an explicit 2% contamination**, not scikit-learn's
  `"auto"`, which flagged 24% of this dataset. A detector that calls a quarter
  of the data anomalous contributes nothing to a vote.

> **A flag is not a diagnosis.** These are statistical outliers. They cannot
> distinguish a faulty sensor from a genuine pollution episode — a
> stubble-burning night, a festival, a still winter inversion look identical in
> the numbers. That is exactly why the record is kept and the judgement left to
> a human (AC-5).

### What gets written

`observations.is_anomaly`, and nothing else. Phase 2's ingestion owns the
measurement columns and never touches this one, so the two write paths never
fight: ingestion records what arrived, the quality engine records what it thinks
of it. Imputed values are **not** written back — `observations` holds what was
measured, and the repaired frame is a derived artefact.

That artefact goes to `data/processed/`:

| File | Contents |
|------|----------|
| `observations_cleaned.csv` | The null-free feature set |
| `quality_report.json` | Counts, mechanisms, justifications, and every caveat |

CSV rather than Parquet: the dataset is small, the file stays readable without
a toolchain, and it adds no dependency. The report travels beside it so a
cleaned file is never separated from the account of how it was produced.

Flags are written in both directions — a row that no longer meets the threshold
is unflagged — so the column reflects the current detectors rather than the
union of every run ever made. Unflagging changes a judgement, never the data.

---

## EDA & statistics

BACSE301 Modules 3–5 (FEAT-02). Everything here describes the data; nothing
here claims to explain it.

```bash
curl -X POST localhost:8000/api/v1/eda/profile   | jq   # statistics
curl -X POST localhost:8000/api/v1/eda/decompose | jq   # STL
curl localhost:8000/api/v1/eda/report -o report.html    # the document
```

### The profile

| Layer | What it answers |
|-------|-----------------|
| **Univariate** | Centre, spread, shape and completeness per column: mean, median, std, IQR, 5th/95th percentile, skew, excess kurtosis, missingness (AC-3) |
| **Bivariate** | Pearson **and** Spearman, plus the overlap each pair was computed on |
| **Distribution** | Whether a column is skewed enough to want a log transform, and whether the transform actually helps |

**Both correlation methods, always.** They answer different questions and
disagree informatively: Pearson measures linear association and is pulled hard
by outliers; Spearman measures monotone association on ranks and is not. A
large Spearman beside a small Pearson means the relationship is monotone but
curved — which a linear model will under-fit. Each pair reports its
`divergence` so that case is easy to spot.

**Every coefficient carries its sample size.** A correlation from 40
overlapping rows and one from 30,000 are different claims, and a bare number
hides which you have.

**The log-transform recommendation is verified, not assumed.** The transform is
applied, the skew recomputed, and it is recommended only when the magnitude
actually falls materially. A log transform costs interpretability —
coefficients stop being in µg/m³ — so it has to buy something. On the current
demo window nothing qualifies: PM2.5 and PM10 sit near 0.8 skew, below the 1.0
threshold. specs §6.1 names CO and SO2 as the motivating case, and neither is
in the specs §9 schema.

### Time-series decomposition

STL on one station's series, separating **trend / seasonal / residual** — which
is what turns "PM2.5 was 180 last night" into how much was the seasonal
baseline, how much the ordinary evening peak, and how much was genuinely
unusual.

Three things STL needs that raw observations do not provide, each handled
explicitly rather than assumed:

- **One series.** A decomposition of stacked stations is meaningless, so a
  station is chosen (the one with the most data) or named.
- **A gap-free grid.** Absent hours are reindexed in and interpolated, and the
  **count of interpolated points is reported** — a decomposition resting on 30%
  invented data says so, and carries a caveat above 10%.
- **Enough history.** Fewer than three full cycles is refused rather than
  fitted; a confident-looking trend from two days is worse than an error.

Both **strength measures** (Hyndman–Athanasopoulos) come back on 0–1: the share
of variation each component explains beyond the noise. They make two series
comparable in a way raw component amplitudes do not.

> statsmodels ships STL as a compiled extension, so it is imported lazily. On a
> machine where the OS refuses to load it, the endpoint returns 503 and the rest
> of the engine keeps working rather than the application failing at startup.
> This is not hypothetical — it happened on the development machine mid-phase.

### Caching

design §11 asks for a cache whose keys include the dataset version "so results
never go stale silently". The **dataset version** is a fingerprint of the slice:
row count, max id, newest timestamp, and flagged-row count. One aggregate query,
and it moves whenever the data does — an ingestion run changes the count and the
max id, a quality run changes the flag count. A cached entry is therefore
*unreachable* once its data has changed, not merely unlikely to be served.

Measured on the demo dataset: **0.94 s cold, 0.006 s warm.** It is an in-process
LRU with a TTL, not Redis — the deployment is three containers (specs §12), and
a fourth for a cache in front of a second-scale computation would be poor value.
The honest consequence: with several workers each holds its own cache, so the
hit rate falls but correctness does not, because the key still pins the data.

### The report

`GET /eda/report` renders one **self-contained HTML file** — inline CSS, inline
SVG, no scripts, no CDN, nothing to fetch. It opens from a USB stick and
survives being emailed, which is the same offline promise DR-1 makes about the
data itself.

Charts follow a validated palette: one hue for magnitude, a **blue↔red diverging
ramp with a grey midpoint** for correlation (which has a sign, so a one-hue ramp
would hide the difference between −0.8 and +0.8), thin marks, recessive grid,
and **every heatmap cell labelled** — a heatmap read by colour alone is
unreadable in greyscale, in print, and for a colour-blind reader. The STL panels
share one scale per panel, because two lines each normalised to their own range
is a dual-axis chart in disguise. Light and dark are each stepped for their own
surface.

**PDF** comes from the browser: the report carries a print stylesheet with
page-break rules, so Print → Save as PDF produces a clean paginated document. A
server-side renderer (WeasyPrint) would add ~100 MB of system libraries to the
image for a Should-Have feature; `report.to_pdf()` uses it if it happens to be
installed and says exactly that when it is not.

The report closes with the ETH-1 disclaimer, verbatim, from the same constant
the UI will use (task 10.16) — an exported document travels further than the
dashboard does.

---

## Feature engineering

specs §6.1 / design §8. Twelve columns, built by **one** transformer that both
training and serving call.

```bash
cd backend && python -m scripts.build_features        # writes data/processed/features.csv
```

| Group | Features |
|-------|----------|
| Temporal | `hour_of_day` · `day_of_week` · `is_weekend` · `month` · `season` |
| Lag | `pm25_lag_1h` · `pm25_lag_24h` · `temp_lag_3h` |
| Rolling | `pm25_rolling_mean_24h` · `traffic_score_rolling_std_6h` |
| Transform | `pm25_log` · `pm10_log` — added only when the skew test says they earn their place |

Named after the schema (`pm25`) rather than the requirement's prose
(`PM2.5_lag_1h`): a feature name that does not match its source column is a
rename waiting to be got wrong.

### The hourly grid

design §8: "without a uniform grid, `lag_1h` is not a well-defined shift".
Ingestion puts every source on an hourly grid (DR-4), but a grid can still have
**holes** — an hour with no reading produces no row. `shift(1)` moves by one
*row*, so across a six-hour outage it would label a reading from six hours ago
as `pm25_lag_1h`: a wrong number with a right-looking name, which no downstream
assertion would catch.

So each station's series is reindexed onto a complete hourly range first. Absent
hours become NaN, the shift becomes a true time shift, and a lag that reaches
over a gap comes back **NaN** — the honest answer, rather than whatever the
sensor last reported.

### Local time, not UTC

Timestamps stay UTC in the database (DR-2), but `hour_of_day` is a claim about
human activity. In IST the evening peak sits near 19:00 local — 13:30 UTC — so a
UTC-derived hour puts rush hour mid-afternoon and splits every weekday across two
dates. The offset (+05:30, no DST) lives *inside* the feature spec, so the value
used in training is the value replayed at inference.

Seasons follow the regional calendar rather than meteorological quarters:
**post-monsoon (Oct–Nov)** is the stubble-burning, low-inversion window that
dominates North Indian PM2.5, and a Sep–Nov "autumn" would split it in two.

### One transformer, two paths

The phase exists to make one guarantee: *every feature at time t is a function of
that station's observations in `[t − history, t]` and of t itself, and nothing
else.* Three decisions follow from it, and each one costs something:

| Decision | Why | Cost |
|----------|-----|------|
| **No MICE in the feature path** | MICE fits across a whole slice, so the same hour gets different values from a year of training data than from a two-day serving window. Only window-local forward fill is used. | A gap too long to repair locally surfaces as NaN instead of a number |
| **Windows reduced from their own contents** | pandas' rolling aggregates run an incremental accumulator whose rounding error depends on how many rows preceded the window — so the same hour can differ in its last bits between the two paths | ~2 orders of magnitude on that step; `exact_windows=False` buys the speed back |
| **The log decision is frozen at fit time** | Recomputing skew on a serving window would answer differently than the training set did | The spec must travel with the model artefact |

The payoff is a claim that can be checked, so it is checked literally: build the
features over 400 hours, build them again over the trailing window a prediction
request would load, and compare the two rows **after serialisation**
(`test_the_two_paths_serialise_to_identical_bytes`).

Nothing reads forward, and that is asserted as a property rather than by
inspection: rewrite every observation after hour 60 and every feature at or
before hour 60 must be bit-for-bit unchanged (AC-8).

### The feature store

Three files in `data/processed/`, written together on purpose — a feature file
without the definition that built it is a set of unlabelled numbers:

| File | Contents |
|------|----------|
| `features.csv` | The engineered frame |
| `feature_spec.json` | The fitted transformer: spec, fingerprint, fit window, and the skew evidence behind every log decision |
| `feature_manifest.json` | Rows, completeness, per-feature null counts, dataset fingerprint |

Warm-up rows (the first 24 hours of each station) keep their NaNs. Which rows a
model may use is Phase 7's decision, not this layer's — the transform returns
every row it was given.

---

## Dimensionality reduction & ESI

FEAT-04 (specs §6.2, design §9). A dashboard cannot show five correlated numbers
and call it a summary, so PCA compresses them into one.

```bash
curl -X POST localhost:8000/api/v1/eda/reduce | jq   # components, loadings, ESI
curl -X POST localhost:8000/api/v1/eda/tsne   | jq   # 2D scatter, EDA Studio only
```

1. **Standardize** every continuous variable. PCA maximises variance and
   variance is scale-dependent — left in raw units, PM10 (tens to hundreds)
   would dominate humidity (a percentage) for no reason but its units.
2. **Fit PCA**, keep the explained variance ratios and the **loadings**.
3. **Normalize PC1 to 0–100** against the fit window.

### The sign is chosen, not accepted

A principal component is defined only up to sign: the same data yields PC1 or
its exact negative depending on the LAPACK build, and an index that silently
inverted between runs would be worse than no index. PC1 is oriented to increase
with PM2.5, so **high ESI means dirtier air by construction** rather than by
luck. The response states the orientation and whether the axis had to be
flipped.

On the demo dataset PC1 loads positively on PM2.5, PM10 and traffic and
negatively on temperature — a winter-inversion axis, readable directly from the
numbers.

### What ESI is not

A **relative** position within the window it was fitted on. 80 means "high for
this city in this period", never "unsafe": the index is not health-calibrated,
and it makes no claim about what caused the reading (ETH-1). Both disclaimers
ship inside the payload rather than being left to whoever writes the UI.

Scores are clipped to 0–100, so a reading more extreme than anything in the fit
window saturates rather than escaping the scale (AC-6). Rows with a missing
measurement are excluded from the projection and **counted** — PCA has no notion
of a missing value, and dropping them silently would let an ESI computed from a
tenth of the window look exactly like one computed from all of it.

The loadings travel with every score (task 6.6), which is what keeps the index
interpretable rather than a black box; the EDA Studio renders them in Phase 10.
The fitted model — scaler parameters, components, variance ratios, calibration —
is written to `data/processed/pca_model.json`, so a stored ESI is still readable
months later.

### t-SNE is strictly a picture

`POST /eda/tsne` embeds the readings in two dimensions for cluster inspection,
and **never feeds a model or the ESI** (design §9). That is not a policy choice:
`TSNE` has `fit_transform` and no `transform`, because a new point has no defined
position in an embedding optimised for other points.

Local neighbourhoods are meaningful; the distance between clusters, their sizes
and the orientation of the axes are artefacts of the optimisation. The caveat
ships in the payload. Points are capped and sampled **evenly across the window**
— deterministic, so the scatter does not redraw itself on every refresh, and
bounded, because t-SNE is quadratic.

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
│   │   ├── quality/             # Missingness, imputation, outliers, pipeline
│   │   ├── eda/                 # Profile, STL, cache, report, PCA/ESI, t-SNE
│   │   ├── features/            # Spec, temporal, windows, transformer, store
│   │   ├── harmonizer.py        # UTC, decimal degrees, hourly resample
│   │   ├── ingestion_service.py # The single write path
│   │   ├── datasets.py          # The pandas boundary
│   │   ├── scheduler.py         # Scheduled-mode background task
│   │   ├── demo_data.py         # Offline demo dataset generator
│   │   └── geo_service.py       # Districts and station locations
│   ├── scripts/
│   │   ├── seed_demo.py         # python -m scripts.seed_demo
│   │   ├── run_quality.py       # python -m scripts.run_quality
│   │   └── build_features.py    # python -m scripts.build_features
│   ├── tests/
│   ├── alembic.ini
│   ├── entrypoint.sh            # Migrate, seed if empty, then serve
│   ├── main.py                  # App factory: CORS, routers, handlers
│   └── requirements.txt
├── data/
│   ├── raw/                     # Immutable landing zone — districts.geojson
│   └── processed/               # Cleaned frame, reports, feature store, PCA model (generated)
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
| `AUTO_RUN_QUALITY` | `true` | Run the quality pipeline once, right after a fresh demo seed |

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
| `POST` | `/data/quality` | Impute, detect anomalies, flag | ✅ Phase 3 |
| `POST` | `/eda/profile` | Univariate, bivariate and distribution statistics | ✅ Phase 4 |
| `POST` | `/eda/decompose` | STL trend / seasonal / residual | ✅ Phase 4 |
| `GET` | `/eda/report` | Self-contained HTML EDA report | ✅ Phase 4 |
| `GET` | `/eda/cache` | Profile cache statistics | ✅ Phase 4 |
| `POST` | `/eda/reduce` | PCA components, variance, loadings, ESI | ✅ Phase 6 |
| `POST` | `/eda/tsne` | t-SNE 2D projection (EDA Studio only) | ✅ Phase 6 |
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
| **3** | Data quality engine — missingness, MICE, outlier vote | ✅ Complete |
| **4** | EDA & statistical engine — profile, STL, cache, report (FEAT-02) | ✅ Complete |
| **5** | Feature engineering — temporal, lag, rolling, log; one shared transformer | ✅ Complete |
| **6** | Dimensionality reduction & ESI — PCA, loadings, 0–100 index, t-SNE (FEAT-04) | ✅ Complete |
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
| Mod 2 | Data Preprocessing | MCAR/MAR analysis, MICE imputation, IQR / Z-score / Isolation Forest anomaly vote — **implemented** (Phase 3) |
| Mod 3 | Descriptive Stats & Visualization | Univariate profile, Pearson + Spearman correlation, histograms — **implemented** (Phase 4) |
| Mod 4 | Dimensionality & Time-Series | STL decomposition with strength measures — **implemented** (Phase 4); PCA-based ESI and t-SNE — **implemented** (Phase 6) |
| Mod 5 | Advanced Visualization | Self-contained HTML report with inline SVG charts — **implemented** (Phase 4); parallel coordinates in Phase 10 |
