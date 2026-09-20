# Retention

Status: **built**. `services/retention.py`, `python -m scripts.prune`.

There was no retention of any kind before this — nothing purged, expired or
pruned, and neither `specs.md` nor `design.md` mentioned the subject. This note
records what was measured, why most of it turned out not to need a policy, and
what the one real problem was.

---

## 1. What was measured

On a live database, after `VACUUM` so the numbers are not bloat:

| | Size | Growth |
|---|------|--------|
| `observations` heap | 8.8 MB / 31,701 rows — **290 b/row** | 264 rows/day |
| `observations` indexes | 20.1 MB — **663 b/row** | — |
| `artifacts/` | **45 MB from 10 files** | ~46 MB per training run |
| `models`, `predictions`, `ingestion_runs` | < 350 kB combined | small, unbounded |

Two things stand out.

**Observations are not the problem.** Eleven stations on an hourly grid is 264
rows a day, about 96,000 a year, under 100 MB a year including indexes. A
decade of Delhi is roughly a gigabyte. Note also that indexes are **2.3× the
data** — four of them, including a PostGIS GiST index on a generated `geom`
column — so if observation storage ever did matter, that is where 70% of it is.

**Artifacts are the problem.** A random forest pickles to ~22 MB, and the
ladder trains one per horizon, so a single training run writes about 46 MB and
nothing ever removed it. Retrained daily that is 17 GB a year; hourly, 400 GB.
Two orders of magnitude worse than the data anyone would think to worry about.

## 2. The two rules

### Retention deletes artifacts, never `models` rows

`predictions.model_id` is `ON DELETE CASCADE`. Deleting a registry row to
reclaim its file would take that model's forecast history with it — the drift
dataset design §6.1 asks for, and the only record of how a deployed model
behaved on data it never trained on. An audit trail destroyed as a side effect
of tidying up is the worst possible version of this feature.

So the row stays. It costs a few hundred bytes and holds the part worth
keeping: the metrics, the feature list, the date. The file is the weights.
`registry.load_artifact` already answers a missing file with
`ModelNotFoundError`, so a pruned artifact has a defined failure mode rather
than a stack trace — and it can only be reached by asking for an old model by
name, because the newest is always kept.

### Nothing here deletes a measurement

The quality engine's rule is flag, never delete (AC-4, AC-5). A retention job
holding database credentials is exactly the place that rule gets quietly
broken, so `test_a_full_pass_never_removes_an_observation` asserts it rather
than trusting it. `observations` also has a `RESTRICT` foreign key to its
source specifically so history outlives a source's removal; at under 100 MB a
year there is no reason to weaken any of that.

If it ever does matter, the answer is to downsample old hours into daily
aggregates, not to delete them.

## 3. What the sweeps do

### `sweep_artifacts`

Keeps the newest `artifact_keep_per_model` (default **3**) per
`(target, model name)`. Grouping by both rather than by target alone is what
guarantees the production model survives: `serving.load_latest` resolves a
target to its newest row of any name, and that row is necessarily the newest of
its own name too, so keeping at least one per group keeps it. A misconfigured
`0` is floored to `1` for the same reason — and that floor has its own test.

Two kinds of file go:

- **superseded** — referenced by a row that has fallen out of the newest few;
- **orphaned** — in the directory, referenced by no row at all. These are
  normal, not corruption: every `db`-marked test that trains a ladder writes
  artifacts and then rolls its rows back. The first dry run on the development
  machine found two.

Only `*.joblib` is ever globbed, because the directory is a mounted volume in
the container and a sweep that took everything would take whatever else was
mounted beside it.

This one runs **automatically after each registration**
(`prune_artifacts_on_register`, default on). Registration is the only moment an
artifact can become superseded, and a retention job that has to be remembered
is a retention job that runs once.

### `sweep_run_log`

Drops `ingestion_runs` older than `run_log_retention_days` (default **90**).
`quarantined_records` cascades from it deliberately: a rejected payload is
evidence for the run that rejected it, and keeping it after the run is gone
leaves a rejection nobody can trace back to anything.

### `sweep_unscored_predictions`

Drops forecasts still missing an `actual_value` more than
`unscored_prediction_retention_days` (default **30**) after their target hour.
Such a row is not waiting for anything — the observation that would have scored
it never arrived — and it is re-scanned by every future backfill.

**Scored rows are never removed, at any age.** Age is what makes them valuable,
so age must not be what removes them.

## 4. Using it

```bash
python -m scripts.prune --dry-run --verbose   # report only, delete nothing
python -m scripts.prune                       # do it
```

The dry run is worth the habit, and it is itself tested: it reports the same
counts and bytes the real run would and leaves every file in place.

Settings live in `core/config.py` and can be overridden per deployment:

| Setting | Default | |
|---------|---------|---|
| `artifact_keep_per_model` | 3 | newest N per (target, name); floored at 1 |
| `prune_artifacts_on_register` | true | sweep automatically after training |
| `run_log_retention_days` | 90 | |
| `unscored_prediction_retention_days` | 30 | scored rows are exempt |

## 5. What is deliberately not covered

- **`observations`** — see §2. No policy, by design.
- **`models` rows** — see §2. They only ever accumulate, at a few hundred bytes
  each, and they are the metrics history.
- **`data/processed`** — the quality and feature outputs are overwritten in
  place rather than versioned, so they do not grow.
- **Automatic scheduling.** The artifact sweep is automatic because it is tied
  to an event; the other two are age-based and belong in whatever already runs
  cron for the deployment. Adding a scheduler thread for something that needs
  to run once a week would be more machinery than the problem deserves.
