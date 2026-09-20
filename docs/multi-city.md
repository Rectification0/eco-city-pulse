# Multi-city — what it would take

Status: **plan only, nothing built.** Written 2026-09-20, after live ingestion
was switched on for Delhi.

The question that prompted this was "why only Delhi?". The short answer is that
the adapters are already geography-agnostic — they take a latitude and a
longitude and return readings — and every Delhi-specific fact in the system
lives in data or in one hardcoded constant. The long answer is that "several
cities at once" is a different change from "a different city", and one of them
is much cheaper than the other.

This document records what binds the platform to Delhi today, what has to be
fixed first regardless of which option is chosen, and what the options cost.

---

## 1. What is actually Delhi-specific

Five things, in descending order of difficulty.

### 1.1 `data/raw/districts.geojson` — the single source of geography

Eleven districts. Every location in the system derives from this one file:
`geo_service.load_districts` reads it, `district_centroids` derives the
coordinates, and `stations_from_districts` turns those into the stations that
the demo generator writes to, that the three adapters are queried at, and that
the map draws.

The file's own metadata is candid about what it is:

> "Schematic approximation. Each district is a rectangular cell placed at the
> district's true relative position within the NCT bounding box; these are not
> surveyed administrative boundaries. **Replace with an official boundary file
> before any operational or published use.**"

So even the Delhi geometry is a placeholder. Any real multi-city work needs
licensed boundary data per city — GADM, Natural Earth, OSM extracts or a city's
own open-data portal — and those have materially different licence terms. This
is a procurement question before it is an engineering one.

### 1.2 A station is a coordinate, not an entity

`datasets.station_key` is `f"{lat:.5f},{lon:.5f}"`, and `observations` is
unique on `(source_id, timestamp, lat, lon)`. There is no station table, no
district table and no city column anywhere in the schema.

This is the load-bearing constraint. Two cities can coexist in one
`observations` table today only by accident of their coordinates differing, and
nothing can answer "which city is this row in?" without a spatial join against
whichever GeoJSON happens to be on disk.

### 1.3 IST is hardcoded

`services/features/temporal.py` shifts UTC onto local time with a **fixed
+05:30 offset**, and says why:

> "An offset rather than a named zone, deliberately. A zone would apply
> daylight-saving transitions, which is correct in the abstract and wrong here:
> the region modelled has none."

That reasoning is sound for Delhi and wrong for any city that observes DST.
`hour_of_day` is a claim about human activity — rush hour, the working day — so
it has to be local to the city the row belongs to. Multi-city turns this
constant into a per-city property, and re-opens the DST question the comment
deliberately closed.

### 1.4 The demo generator is calibrated to Delhi's climate

`services/demo_data.py` models monsoon humidity peaking around day 210, a
hot-summer/mild-winter temperature curve, and episodic spikes described as
"stubble burning, a festival, a still winter night". Point it at Wellington or
Lima and it produces confident fiction about that city's air.

Either the generator gains a per-city climate profile, or demo mode stays
Delhi-only and other cities are live-data-only.

### 1.5 The model registry has no city dimension

`models` is keyed by `(name, target)` and `serving.load_latest(session,
target)` takes the newest row for a target. Train a second city and it
overwrites the first city's production model for that horizon. Feature
transformers are stored per model artifact, so they inherit the same problem.

---

## 2. The prerequisite: live rows are not joined rows

> **Update, 2026-09-20: the merge described here is now built** — migration
> `0005`, written up in [`observation-merge.md`](./observation-merge.md). What
> follows is the analysis that led to it, kept because the *station entity* it
> argues for is still the open piece and is still what multi-city needs. The
> merge was done with a provenance key instead, which was the smaller change;
> the station table remains the right home for a monitor's true position.

This blocks multi-city, and it already blocked single-city live mode. It should
be fixed first whichever option is chosen, because every option multiplies it.

`Observation` describes itself as "one harmonized hourly reading at one
location ... a single row is the joined environmental state at a point in space
and time". For the demo bundle that is true — one source writes every column.
For live data it is not. Measured from the first real ingest:

| Source | Rows | Fills | At |
|--------|------|-------|-----|
| AQICN | 10 | pm25, pm10, temp, humidity | its own station coordinates |
| OpenWeather | 11 | temp, humidity | the district centroid |
| TomTom | 11 | traffic_score | the district centroid |

Nothing merges the three. The unique constraint includes `source_id`, so the
schema actively keeps them apart. Consequences:

- No live row has both `pm25` and `traffic_score`. The feature the model leans
  on hardest never co-occurs with the target.
- AQICN resolves each request to its *nearest station*, whose coordinates match
  no district centroid, so its PM2.5 belongs to no polygon. Two districts can
  also resolve to the same station — `north-east-delhi` and `shahdara` do — so
  eleven requests produce ten rows.
- `DISTINCT ON (lat, lon)` at a centroid therefore returns an OpenWeather or
  TomTom row, both of which have `pm25` of null.

**The fix is a station entity.** A `stations` table — `(id, city_id,
district_id, lat, lon, name)` — with `observations.station_id` replacing the
coordinate pair as the identity, and each adapter's reading snapped to the
station it was requested for rather than the coordinate the upstream chose to
answer with. Merging then means grouping on `(station_id, timestamp)` and
coalescing columns, which is a query rather than an architecture.

That same table is exactly what multi-city needs for §1.2. The two pieces of
work are one piece of work, which is the main argument for doing it properly
rather than patching the map.

---

## 3. Options

### Option A — one city per deployment

Swap `districts.geojson`, set a timezone offset and a climate profile in
config, run a separate instance per city.

- **Schema change:** none.
- **Cost:** small — config plumbing plus the §1.3 and §1.4 constants.
- **Gets you:** Bengaluru instead of Delhi, cleanly.
- **Does not get you:** two cities in one dashboard, cross-city comparison, or
  a shared model. Operationally it is N databases and N deployments.

### Option B — multi-city in one deployment

`cities` and `stations` tables, `observations.station_id`, city-keyed model
registry, a `city` parameter through the API, a city selector in the frontend.

- **Schema change:** substantial — a migration that backfills every existing
  row to a Delhi station.
- **Cost:** phase-sized. Touches ingestion, datasets, features, training,
  serving, every route and the whole frontend.
- **Gets you:** the real thing, including cross-city EDA, which is the more
  interesting analysis anyway.

### Option C — station entity now, one city for now

Do §2 — `cities`/`stations` tables, `station_id` on observations, per-city
timezone — and leave the API and frontend single-city. Multi-city becomes a
later, much smaller change.

- **Schema change:** the same migration as B, without the surface area.
- **Cost:** moderate, and it is cost that B would pay anyway.
- **Gets you:** live mode that actually works, which is the immediate problem.

**Recommendation: C, then B if it is still wanted.** C is the intersection of
"fix live data" and "prepare for multi-city", so none of it is throwaway. It
also keeps the capstone's demo path untouched while the live path is repaired.

---

## 4. Sketch of the work in C

Not a task list yet — sizes are guesses until the migration is written.

1. **Migration.** `cities (id, name, country, tz_offset_minutes, geojson_path)`;
   `stations (id, city_id, district_id, lat, lon, name, source_hint)`;
   `observations.station_id` FK, backfilled by matching existing `(lat, lon)`
   to Delhi's centroids; new unique constraint `(source_id, station_id,
   timestamp)`. The old coordinate columns stay — they are the measurement's
   own record of where it was taken, which is not always the station's nominal
   position.
2. **Ingestion.** Each adapter records the station it was *asked* about
   alongside the coordinates the upstream *answered* with. AQICN's nearest
   station becomes a property of the reading, not its identity.
3. **A merged read.** One row per `(station_id, hour)`, coalescing columns
   across sources with a documented precedence — a measured value always beats
   a modelled one, which is what `is_synthetic` already records.
4. **Temporal features.** `tz_offset_minutes` comes from the row's city rather
   than the module constant, and the DST decision in §1.3 gets re-taken
   explicitly rather than inherited.
5. **Geography service.** `load_districts(city)` instead of one cached file,
   and a real boundary file for Delhi while we are there.
6. **Tests.** The merge precedence, the backfill, and a test that two stations
   at the same coordinates in different cities do not collide.

---

## 5. Open questions

These need answers before task 1, and most are not engineering questions.

1. **Which cities, and why?** Comparison across climates is a different
   analysis from coverage of one region, and it changes whether a shared model
   is meaningful at all.
2. **Where does boundary data come from, and under what licence?** Delhi's
   current file is self-generated and schematic. A published capstone probably
   should not ship one.
3. **Do the upstreams cover the candidate cities?** AQICN's station density
   varies enormously; TomTom's traffic coverage more so. A city with no AQICN
   station nearby gets no PM2.5 at all, which is the target variable.
4. **One model per city, or one model with city as a feature?** The second is
   more interesting and needs far more data. The first is honest and boring.
5. **Does demo mode need to cover the new cities?** If yes, §1.4 becomes a
   modelling exercise per city rather than a config change.
6. **Is cross-city comparison in scope for BACSE301?** It would be the
   strongest EDA in the project, and it is also the largest piece of work here.
