# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Eco-City Pulse — a modular monolith that ingests urban environmental data,
subjects it to rigorous EDA, and forecasts PM2.5 with an interpretable ML
pipeline. A BACSE301 capstone, so the *reasoning* is as much the deliverable as
the code.

Requirements live in [`specs.md`](./specs.md), design rationale in
[`design.md`](./design.md), and the phased plan in [`tasks.md`](./tasks.md).
Every task traces to a requirement id (`FEAT-*`, `DR-*`, `SEC-*`, `AC-*`,
`OBJ-*`); keep those references in docstrings and commit messages.

## Commands

```bash
cd backend
pytest                              # full suite; db tests skip without PostgreSQL
pytest -m "not db"                  # offline only
pytest tests/test_ml_explain.py -q  # one file
ruff check --select F,I,E9 .        # the lint gate this repo uses
alembic upgrade head                # apply migrations

python -m scripts.seed_demo         # offline demo dataset
python -m scripts.run_quality       # impute + flag anomalies
python -m scripts.build_features    # engineered feature store
python -m scripts.train_models      # train + register the model ladder
python -m scripts.prune             # retention sweep; --dry-run first
```

Run the app with `docker compose up --build` (three containers; DB publishes on
host port **5433**).

## Architecture rules

- `api/routes/*` validate input, call a service, shape the response. **No
  analytics logic in a route.** All pandas/scikit-learn work lives in
  `services/`, all persistence in `db/`.
- Services raise domain exceptions from `core/exceptions.py`; handlers map them
  to one error envelope. Routes carry no `try/except`.
- Every request and response is a Pydantic model (SEC-1).
- Secrets are read only in `core/config.py`, only from the environment (SEC-2).
- Optional heavy libraries (statsmodels, xgboost, prophet, shap) are imported
  **lazily** so one unavailable extra cannot take down the whole app at startup.

## Conventions

- **Docstrings carry the reasoning.** Say why a decision was made and what the
  alternative would have cost, with the requirement id. That prose is the
  deliverable, not decoration.
- **Deterministic by default.** Fixed seeds, sorted `__all__`, frozen
  dataclasses with `as_dict()` / `from_dict()` where a result is serialised.
- **Caveats travel in the payload**, not in the UI's head: no causal claim
  anywhere (ETH-1).
- **Test names are sentences** describing the property under test, and the
  docstring says why that property matters.
- Acceptance-criteria tests (AC-*) must run **offline**. A criterion that skips
  when no database is reachable is a criterion that is not checked.
- `db`-marked tests use a rolled-back transaction against real PostgreSQL.

## After implementing a phase

Do all of these before committing:

1. **`tasks.md`** — tick the phase's checkboxes (`[ ]` → `[x]`). Do not touch
   other phases.
2. **`README.md`** — **keep it minimal.** Normally this is only:
   - the Implementation status row for the phase, and
   - any new endpoint rows in the API table.

   Add at most one short row to the Pipeline table if the phase introduced a new
   `services/` package. Detailed rationale belongs in module docstrings and
   `design.md` — **do not** add a per-phase prose section to the README.
3. **`.gitignore`** — check whether the phase generates new artifacts (files
   under `data/processed/`, `artifacts/`, reports, caches). Add a pattern if
   anything new is written to disk, and keep the explanatory comment accurate.
4. **Verify** — `pytest` and `ruff check --select F,I,E9 .` both clean. If a
   path only runs with a database, exercise it some other way and say so.
5. **Commit and push** — conventional commit (`feat(phase-N): …`), body
   explaining the decisions and any findings fixed along the way.
6. **If the phase needed Docker Desktop / WSL** — stop the containers, then
   **remind the user to close `vmmemWSL` from Task Manager as administrator**.
   WSL2 does not hand freed memory back to Windows on its own: the VM keeps
   pages it has already allocated even with every container exited, so the
   reminder is the difference between a gigabyte or two reclaimed and a
   gigabyte or two idle. (`wsl --shutdown` does the same thing from a shell.)

## Environment notes

- Host Windows **Application Control** has been seen to block scikit-learn's
  compiled extensions (`_argkmin_classmode.pyd`), which breaks any import of
  `sklearn.metrics` and so the whole app. It cleared on its own a while after
  `pip install --force-reinstall --no-deps scikit-learn==<pinned>`, so try that
  first — the freshly written file gets re-evaluated. If it persists, run the
  suite in the Linux container instead, and **do not weaken the policy**:
  ```bash
  docker compose build backend          # the image must carry current source
  docker compose run --rm --no-deps -e OMP_NUM_THREADS=4 \
    -v "$PWD/backend/tests:/app/tests" \
    --entrypoint pytest backend -m "not db" -q
  ```
  `tests/` is excluded from the production image by `.dockerignore`, hence the
  mount. Cap the BLAS threads: the tree models use `n_jobs=-1` and oversubscribe
  the VM badly (~15× slower without it). The `-m db` tests are slower still,
  because several retrain the whole ladder per test.
- Use the scratchpad directory for throwaway verification scripts; never commit
  them.
