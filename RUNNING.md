# Running Eco-City Pulse end to end

A guide for getting from a fresh clone to **every screen populated** — real
observations, a cleaned dataset, trained models, live forecasts with their
reasoning, and an exported report.

No API keys are needed at any point. The platform runs entirely offline by
design (DR-1), on a synthetic dataset that is labelled as synthetic everywhere
it is stored.

---

## The one thing that surprises everyone

`docker compose up` gives you a working system, but **not a complete one**. The
container startup migrates the schema, seeds 120 days of demo data, and runs the
data-quality pipeline — and then stops. It does **not** train any models,
because training takes minutes and would hold the healthcheck open.

So on a first run you will see:

| Screen | After `compose up` | After you train |
|--------|--------------------|-----------------|
| Dashboard — tiles, map | ✅ populated | ✅ |
| Dashboard — forecast points | ❌ absent | ✅ |
| EDA Studio | ✅ all panels | ✅ |
| Model Lab | ❌ "No models registered yet" | ✅ |
| Admin | ✅ populated | ✅ |

One command fixes it, and it is step 4 below. If the Model Lab is empty, you
have not missed a bug — you have not trained yet.

---

## API keys — you need none

**Zero keys are required.** Every screen you are about to bring up — the map,
the models, the forecasts, the report — runs on a synthetic dataset generated
locally from the standard library. Nothing is downloaded.

This is a requirement (DR-1), not a convenience, and the acceptance suite
enforces it: `test_ac2_demo_mode_needs_no_key_and_contacts_nobody` asserts that
demo mode is the default, that no key is configured, and that the generator
imports no HTTP client at all.

Skip the rest of this section unless you specifically want live data.

### The three optional keys

| Env var | Provider | Free tier | Where |
|---------|----------|-----------|-------|
| `AQICN_API_KEY` | AQICN / World Air Quality Index — air quality | Free for non-commercial use; email address only | [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) |
| `OPENWEATHER_API_KEY` | OpenWeather — weather | 60 calls/min, 1M/month, no card | [openweathermap.org/api](https://openweathermap.org/api) |
| `TOMTOM_API_KEY` | TomTom — traffic flow | 2,500 requests/day, 5 calls/sec | [developer.tomtom.com](https://developer.tomtom.com/) |

They are independent — set one, two, or all three.

**Two things that catch people out:**

- **An OpenWeather key does not work immediately.** It is issued instantly but
  takes roughly 10 minutes to 2 hours to activate on their side. A fresh key
  returning 401 usually means you were simply too quick, not that you did
  anything wrong.
- **AQICN requires attribution** and forbids commercial use or redistribution
  of the data. Fine for this project; check it before you build on it.

### Where to put them

One file, `.env` at the repository root. It serves both routes — `docker
compose` reads it directly, and the native backend reads it through
`core/config.py`, which is the only place in the codebase that touches the
environment.

```bash
cp .env.example .env
```

```ini
# --- Upstream API keys (all optional) ---
AQICN_API_KEY=your-aqicn-token
OPENWEATHER_API_KEY=
TOMTOM_API_KEY=
```

`.env` is gitignored. `.env.example` is committed and must stay empty of real
values — an audit test fails the build if a key is ever pasted into it.

Using Docker? Restart so the new environment is picked up:

```bash
docker compose up -d --force-recreate backend
```

### What actually changes

A key on its own fetches nothing. Something has to trigger a run:

| You want | Do this |
|----------|---------|
| One live fetch, now | `curl -X POST localhost:8000/api/v1/data/ingest` |
| Polling on a timer | `INGESTION_MODE=scheduled` in `.env`, then restart |

> `POST /data/ingest` defaults to **manual** mode, which attempts every live
> source — it does not consult `INGESTION_MODE`. That setting controls the
> *background scheduler*, not the manual trigger.

Without a key, that same call still succeeds: the keyless source degrades to
`offline` and says so in its own outcome. The endpoint returns **200 even when a
source fails**, which is DR-1 expressed in the API — read the per-source
`status`, not the HTTP code.

### Did it work?

Open **Admin**, or:

```bash
curl -s localhost:8000/api/v1/data/sources | jq '.sources[] | {name, status, credentials_configured}'
```

A configured source flips from `offline` to `healthy` once it has fetched
successfully. Without a key, `offline` is the **correct** state — the source is
configured and deliberately never contacted, not broken.

### Traffic is the special case

If `TOMTOM_API_KEY` is absent, traffic does not simply go missing — a synthetic
traffic series stands in, labelled as synthetic everywhere it is stored. That is
why the models still train and the correlations still have something to report
with no keys at all.

---

## Which route to take

| | **Docker** | **Native** |
|---|---|---|
| Setup effort | One command | Python + Node + PostgreSQL/PostGIS |
| Good for | Seeing it work, demos, review | Developing, fast iteration |
| First run | ~5–10 min (image build) | ~10–15 min (installs) |
| Rebuild after a code change | Slow (rebuild image) | Instant (hot reload) |
| Memory | ~2 GB held by the WSL VM | Just the processes |

**Choose Docker if you want to look at it. Choose native if you want to change
it.** Both end at the same place.

---

## Route A — Docker

### Prerequisites

Docker Desktop, running. Nothing else.

### Steps

```bash
git clone <this repo> && cd EcoCityEDA
docker compose up --build          # first build takes a few minutes
```

Watch it come up:

```bash
docker compose logs -f backend
```

You are looking for, in order:

```
[entrypoint] applying migrations...
[entrypoint] seeding demo dataset (offline, no API keys)...
[entrypoint] running the data quality pipeline...
[entrypoint] starting uvicorn...
```

Then open <http://localhost:8080>.

### Train the models (the step above)

```bash
docker compose exec backend python -m scripts.train_models --no-classical
```

This trains all three horizons (+1h, +6h, +24h) × four models, so the Model Lab
ends up with twelve rows. `--no-classical` skips the ARIMA and Prophet
baselines, which dominate the runtime — drop it if you want them in the
comparison. To go faster while you are just looking around:

```bash
docker compose exec backend python -m scripts.train_models --horizon 1 --no-classical
```

That registers four models and is enough for the Model Lab and the +1h forecast;
the Dashboard's +24h point stays absent until you train that horizon too.

### When you are done

Stop the stack **and reclaim the memory** — WSL2 does not hand it back on its
own, and will sit on a couple of gigabytes indefinitely:

```bash
docker compose down
wsl --shutdown          # or close vmmemWSL in Task Manager, as administrator
```

---

## Route B — Native

### Prerequisites

| | Version | Notes |
|---|---|---|
| Python | 3.12 | 3.11 will probably work; 3.12 is what the image uses |
| Node | 18+ | 20 or 24 recommended |
| PostgreSQL | 16+ | **with PostGIS** — see below |

**PostGIS matters.** Without it the migration still succeeds and everything
works *except* the spatial column and the district map joins. On Windows,
install it through Stack Builder alongside PostgreSQL; on macOS
`brew install postgis`; on Debian/Ubuntu `apt install postgresql-16-postgis-3`.

Check before you start:

```bash
psql -U postgres -c "SELECT * FROM pg_available_extensions WHERE name='postgis';"
```

### Steps

```bash
# 1. Configuration
cp .env.example .env
# Edit POSTGRES_USER / POSTGRES_PASSWORD to match your local server.
# POSTGRES_PORT stays 5432 for a local install.

# 2. Create the database
psql -U postgres -c "CREATE DATABASE ecocitypulse;"

# 3. Backend dependencies
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # a few minutes: xgboost, prophet, shap are large
```

### The pipeline, in order

Each step consumes what the previous one produced. Run them from `backend/`
with the virtualenv active.

```bash
alembic upgrade head                     # 1. schema          (seconds)
python -m scripts.seed_demo --days 120   # 2. ~31,700 rows    (seconds)
python -m scripts.run_quality            # 3. impute + flag   (under a minute)
python -m scripts.train_models --no-classical   # 4. train + register  (a few minutes)
```

| Step | Produces | Skip it and… |
|------|----------|--------------|
| 1 `alembic upgrade head` | The schema | Nothing else runs at all |
| 2 `seed_demo` | Rows in `observations` | Every screen is empty |
| 3 `run_quality` | `is_anomaly` flags, `data/processed/` | Anomaly counts read 0; models train on unrepaired gaps |
| 4 `train_models` | Artifacts + rows in `models` | Model Lab empty, no forecasts |

Step 4 is the one the container does not do for you.

> **`build_features` is optional.** `python -m scripts.build_features` writes the
> feature store to `data/processed/` for inspection. Training builds its own
> features in memory, so skipping it changes nothing about the running system.

### Serve it

Two terminals:

```bash
# backend
cd backend && uvicorn main:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173>.

> Pointing the native backend at the **compose** database instead? That one
> publishes on **5433**, not 5432:
> `POSTGRES_HOST=localhost POSTGRES_PORT=5433 uvicorn main:app --reload`

---

## Check that it actually worked

### By eye

| Screen | What you should see |
|--------|---------------------|
| **Dashboard** | Four tiles with numbers; a dark map with coloured districts; a 48-hour line with orange diamond forecast points and error bars |
| **EDA Studio** | Ten charts — missingness, histogram + boxplot, correlation heatmap, four STL panels, parallel coordinates, t-SNE |
| **Model Lab** | Twelve rows (four models × three horizons), a skill column, a SHAP bar chart, and a working forecast form |
| **Admin** | Six sources (five `offline`, one `healthy`), one ingestion run, empty quarantine, a threshold form |
| **Every screen** | The amber disclaimer in the footer |

Some sources reading `offline` is **correct**, not a failure: a live API with no
key is configured and deliberately never contacted.

### By command

```bash
cd backend
pytest tests/test_acceptance.py -v     # AC-1 … AC-11, one test each
pytest tests/test_audit_*.py           # the security, privacy and ethics audits
```

Or check the API directly:

```bash
curl localhost:8000/api/v1/data/observations/latest | jq '.readings | length'   # 11
curl localhost:8000/api/v1/ml/models | jq 'length'                             # 12
curl -X POST localhost:8000/api/v1/ml/predict \
  -H 'Content-Type: application/json' \
  -d '{"lat":28.66,"lon":77.13,"horizon":1}' | jq
```

That last one should return a prediction, a unit of `ug/m3`, a two-element
confidence interval, and `top_features` — the specs §8 contract exactly.

### Export the artefacts

```bash
python -m scripts.export_docs --report   # → docs/openapi.json, docs/eda-report.html
```

The report is a single self-contained HTML file: no scripts, no CDN, nothing to
fetch. It opens from a USB stick and prints to PDF.

---

## When something goes wrong

### "No models registered yet" in the Model Lab

You have not run step 4. See above — this is the expected first-run state, not a
bug.

### The Dashboard has no forecast points

Same cause. The line is observed history, which needs only the seed; the
diamonds need trained models.

### `connection refused` / `no database at localhost:5432`

Three usual causes, in order of likelihood:

1. PostgreSQL is not running.
2. You are pointing at the wrong port — a **local** install is 5432, the
   **compose** database publishes on **5433**.
3. `.env` has credentials that do not match your server.

### `db` tests all skip

That is by design when no database answers, but it also happens when the
database is reachable and **not migrated**. The skip message says which. Run
`alembic upgrade head`.

### The map has no street background

The basemap is fetched from OpenStreetMap, so it needs internet. Everything else
— polygons, colours, tooltips, legend — works offline; only the tiles are
external.

### `ImportError: DLL load failed ... Application Control policy has blocked this file`

Windows Application Control blocking one of scikit-learn's compiled extensions.
This breaks any import of `sklearn.metrics`, and therefore the whole backend.

First try reinstalling — the freshly written file gets re-evaluated, and this
cleared it here:

```bash
pip install --force-reinstall --no-deps scikit-learn==1.9.1
```

If it persists, run in the container instead. Do not weaken the policy.

### `ModuleNotFoundError: xgboost` (or prophet, or shap)

They are in `requirements.txt` but are large and sometimes fail quietly during a
bulk install. Install the missing one directly. Each is optional in the sense
that the rest of the app keeps working — these libraries are imported lazily, so
you lose one estimator or one explainer rather than the whole application.

### Tests are very slow in the container

The tree models use `n_jobs=-1` and oversubscribe the VM badly — roughly 15×
slower. Cap the threads:

```bash
docker compose run --rm --no-deps -e OMP_NUM_THREADS=4 \
  -v "$PWD/backend/tests:/app/tests" \
  --entrypoint pytest backend -m "not db" -q
```

`tests/` is excluded from the production image, hence the mount.

### `vmmemWSL` is eating gigabytes

WSL2 keeps memory it has already allocated even after every container exits.
`wsl --shutdown`, or close `vmmemWSL` from Task Manager as administrator.
Docker restarts it on demand.

### Training says it cannot beat the baseline

Check the window. Persistence — "next hour looks like this hour" — is a strong
forecast for hourly PM2.5, and on a very short seed there may be too little
history for anything to improve on it. `--days 120` is the tested default.

---

## Time budget

| | Docker | Native |
|---|---|---|
| Install / build | 5–10 min | 10–15 min |
| Migrate + seed + clean | automatic | ~1 min |
| Train | 1–2 min | 1–2 min |
| **Total to a full system** | **~10 min** | **~20 min** |

Add several minutes if you train with `--classical`, which fits ARIMA at
sampled origins and Prophet on each horizon.

---

## A note on the interface itself

The screens were built with the non-visual reader in mind, not only the
sighted one:

- every chart carries an `aria-label` describing what it plots — and this is
  enforced rather than remembered: the shared `<Plot>` component takes
  `ariaLabel` as a *required* prop, so a chart without one does not compile.
  The map is labelled the same way;
- colour never carries meaning alone — air-quality bands always appear with
  their name, correlation cells are labelled with their coefficient, and the
  map has a legend naming every band;
- the palette was validated for colour-vision deficiency against the chart
  surface rather than chosen by eye;
- the numbers behind each chart are available as JSON from the same endpoint the
  chart calls, so a screen reader user can reach the data without the picture.

The one place this is incomplete is keyboard interaction inside the Plotly and
Leaflet widgets, which is bounded by those libraries.

---

## Where to look next

| Document | For |
|----------|-----|
| [`README.md`](./README.md) | What it is, architecture, API surface |
| [`design.md`](./design.md) | Why it is built this way |
| [`specs.md`](./specs.md) | Requirements and acceptance criteria |
| [`CLAUDE.md`](./CLAUDE.md) | Conventions if you are going to change it |

The reasoning behind any given decision lives in that module's docstring. They
are written to be read.
