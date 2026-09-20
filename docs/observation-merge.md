# Merging a station-hour across sources

Status: **built**, migration `0005`. Written 2026-09-20.

`observations` has always described itself as holding "one harmonized hourly
reading at one location ... the joined environmental state at a point in space
and time". For the demo bundle that was true, because one source writes every
column. For live data it was not, and this note records why, what changed, and
which decisions inside the change were judgement calls rather than
consequences.

---

## 1. The problem

Measured from the first real ingest, before the fix:

| Source | Rows | Filled | At |
|--------|------|--------|-----|
| AQICN | 10 | pm25, pm10, temp, humidity | its own station coordinates |
| OpenWeather | 11 | temp, humidity | the district centroid |
| TomTom | 11 | traffic_score | the district centroid |

Three feeds, three rows, no joined state anywhere. Concretely:

- **No live row held both `pm25` and `traffic_score`** — the target and the
  variable the model leans on hardest could not co-occur, so nothing trained on
  live data could use traffic at all.
- **AQICN's readings belonged to no district.** It resolves a coordinate to its
  *nearest* monitor, which matches no centroid. Two districts could also
  resolve to the same monitor — `north-east-delhi` and `shahdara` do — so
  eleven requests produced ten rows and the map drew twenty-two stations of
  which eleven were unplaceable.
- **`DISTINCT ON (lat, lon)`** at a centroid therefore returned an OpenWeather
  or TomTom row, neither of which has `pm25`. The dashboard's PM2.5 map was
  blank exactly where it mattered.

## 2. What was already there

Two mechanisms, both correct, neither able to fire.

`harmonizer.resample_hourly` groups by `(hour, lat, lon)` with **no source**,
coalescing each field independently. It was always capable of merging across
feeds; it is just called once per source, inside `ingest_source`, so it never
sees more than one at a time.

`write_observations` upserts with `COALESCE(new, existing)` — documented as "a
fresh value wins, but a NULL leaves what is already stored alone". That is
partial-row merge semantics, written and tested.

The only thing keeping them apart was `source_id` in the unique key
`(source_id, timestamp, lat, lon)`, which made a collision between two feeds
impossible by construction. The fix is therefore mostly subtraction.

## 3. What changed

### 3.1 Every adapter keys on the station it asked about

TomTom already did this — `harmonize_coordinates(*parsed.requested_point)`.
AQICN used the monitor's own position and OpenWeather used `coord`, which
matched the centroid only because OpenWeather echoes lat/lon queries and is
free not to. Both now carry `_requested_point` through the payload, as TomTom
does.

Nothing merges until the three feeds agree on where they are. This step is the
prerequisite; the key change below is what lets the agreement matter.

The monitor's true position is real information and is **deliberately dropped**
rather than half-recorded: it is a property of a station entity this schema does
not have. See [`multi-city.md`](./multi-city.md), where that entity is the
central proposal.

### 3.2 The key is provenance, not source

`(provenance, timestamp, lat, lon)`, where provenance is `measured` or
`synthetic`, mirroring `data_sources.is_synthetic`.

Dropping `source_id` without replacing it would have merged the demo bundle
into live rows — they share the centroids and overlap in time — producing a
statistic built from measurement and generation together and presented as
measurement. That is the one blend no caveat repairs after the fact (ETH-1).
Two classes keep the scope a request resolves and the key a row merges on from
ever disagreeing.

An analyst upload counts as `measured`. A third class would put two rows on one
station-hour inside a single scope, which is the duplicate the column exists to
remove.

### 3.3 Field authority

`temp` and `humidity` arrive from AQICN *and* OpenWeather, and they disagree —
33.8 against 32.9 on the same hour in the first live run. Under plain
`COALESCE(new, existing)` the winner is whoever wrote last, which is decided by
the order of `LIVE_ADAPTER_TYPES`: deterministic, but for no reason anyone
could point at, and different if that tuple were ever reordered.

`AdapterSpec.authoritative_for` declares it instead:

| Source | Owns |
|--------|------|
| AQICN | pm25, pm10 |
| OpenWeather | temp, humidity |
| TomTom | traffic_score |
| Analyst Upload | everything — a person correcting the record outranks a feed |
| Demo Bundle | everything |
| Synthetic Traffic (fallback) | nothing — may only supply what no measurement did |

Two rules, per field:

- an **authoritative** writer emits `COALESCE(new, existing)` — its value
  replaces what is there;
- every other writer emits `COALESCE(existing, new)` — it may fill a NULL,
  never overwrite.

Which makes the result **independent of ingestion order**, and that is the
property worth having: whichever of AQICN and OpenWeather runs first, the
weather source's temperature survives, because an air-quality station's
thermometer is incidental to what it is there to measure.

A consequence worth stating plainly: *a source that claims nothing can only
fill gaps.* That is exactly right for the traffic fallback and exactly wrong
for anything standing in for a real feed, which is why the upload adapter
claims everything.

## 4. What this did not fix, by design

**The feeds do not agree on what hour it is, and they are each right.**
OpenWeather and TomTom report conditions at the moment of the request; AQICN
reports when its monitor last published, typically an hour earlier. After the
fix, a single ingest produced:

```
13:30 IST   9 rows   pm25=9   traffic=0     <- AQICN's stations
14:30 IST  11 rows   pm25=1   traffic=11    <- weather + traffic, and one
                                               station that had just updated
```

So the *newest* row for a station reliably has temperature and traffic and no
PM2.5 — permanently, not while ingestion catches up, because each poll writes a
new weather row at the same time as it fills the previous hour's pollution.

Stamping AQICN with the fetch time would make the rows merge and would be a lie
about when the measurement happened (DR-2). Two things absorb the offset
instead:

1. **`/data/observations/latest` takes the newest row that carries PM2.5**,
   not simply the newest row. One row, one honest timestamp, and
   `stale_minutes` says how far back it is. Taking each column from whichever
   row last had it would manufacture a reading no hour ever produced, on the
   view whose entire job is PM2.5 by district.
2. **Short-gap imputation** already carries a station's values forward within
   its own series, which is what the quality engine is for.

**Per-row source attribution is gone.** `source_id` now means "which feed wrote
last", not "where this row came from". What is no longer answerable from a row
is which feed supplied a *particular column* of it. Recovering that would need
a per-field provenance map, which is a lot of machinery for a question nothing
currently asks — and for the five measurement columns §3.3 already fixes the
answer in all but the contested two.

**This leaves one visible wart.** `/data/sources` derives `observation_count`
by grouping `observations.source_id`, which now counts rows a feed happened to
write *last* rather than rows it contributed to. On a merged station-hour the
losers of that race count zero: after a live ingest the Admin table reads

```
AQICN            healthy   obs=10
OpenWeather      healthy   obs=0     <- contributed to 11 rows
TomTom Traffic   healthy   obs=11
```

which is not a broken feed and does not look like anything else. The honest
number is the ingestion log's `records_written` per source, which survives
merging because it records what a run did rather than what the table retained.
`last_observation_at` has the same flaw for the same reason. Neither is fixed
yet; `data_sources.last_run` and the ingestion log are meanwhile the
trustworthy view of whether a feed is working.

**Rows ingested before §3.1** keep their original coordinates, so historical
AQICN rows stay at the monitor's position and merge with nothing. They are not
wrong, only unmergeable. Re-ingest for a clean grid; migration `0005` says so.

## 5. Results

Same live data, before and after:

| | Before | After |
|---|--------|-------|
| Stations on the map | 22 | 11 |
| Mapped to a district | 11 | 11 |
| Carrying PM2.5 | 10 | 11 |
| Carrying PM2.5 *and* traffic | 0 | 1, rising as hours fill |

## 6. Tests that hold it in place

In `test_ingestion.py`, all `db`-marked:

- three feeds describing one hour become one row;
- the authority decides a contested field **whatever the order** —
  parametrised both ways, and it is the test that fails if the authority rule
  is removed;
- a feed may fill a gap in a field it does not own;
- a generated row never merges into a measured one.

And in `test_models.py`, the unique constraint is asserted column by column, so
putting `source_id` back is a failing test rather than a silent regression.
