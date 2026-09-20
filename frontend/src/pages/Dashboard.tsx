/**
 * Dashboard — tasks 10.3, 10.4, 10.5.
 *
 * Three questions, in the order someone actually asks them: what is the air
 * like now, where is it worse, and what happens next.
 *
 * **On the trendline (10.5).** The platform trains three horizons (+1h, +6h,
 * +24h), so the honest forecast is three points with their intervals, not a
 * smooth 24-point curve. Drawing a continuous line through hours no model was
 * trained for would invent twenty-one predictions. The observed history is the
 * line; the forecast is marked points, visibly separated from it.
 */

import { useMemo, useState } from 'react'
import { Async, useAsync } from '../components/Async'
import Card from '../components/Card'
import DistrictMap from '../components/DistrictMap'
import StatTile from '../components/StatTile'
import Plot from '../charts/Plot'
import { CATEGORICAL, INK, band, baseLayout } from '../charts/theme'
import {
  getDistricts,
  getLatestObservations,
  getReduction,
  getStationSeries,
  predict,
  type PredictionResponse,
  type StationReading,
} from '../services/api'

const HORIZONS = [1, 24]

/**
 * How often the observed panels re-fetch.
 *
 * DR-4 puts every source on an hourly grid and the scheduler polls hourly, so
 * nothing here changes faster than that — a tighter interval would re-request
 * the same hour. Five minutes exists to catch a *manual* ingest, which can
 * land at any moment, and to bound how long the page can sit on a reading that
 * has since been superseded. The forecast panels are left alone: they are
 * derived from this data, and re-running three model inferences every five
 * minutes to redraw the same three points is work nobody asked for.
 */
const OBSERVED_REFRESH_MS = 5 * 60 * 1000

function freshness(observedAt: string | null, staleMinutes: number | null) {
  if (!observedAt) return 'No readings yet.'
  const clock = new Date(observedAt).toLocaleString()
  if (staleMinutes === null) return `Newest reading ${clock}.`
  const age =
    staleMinutes < 90
      ? `${Math.round(staleMinutes)} min`
      : `${(staleMinutes / 60).toFixed(1)} h`
  return `Newest reading ${clock} — ${age} old.`
}

export default function Dashboard() {
  const [selected, setSelected] = useState<StationReading | null>(null)

  const latest = useAsync(getLatestObservations, [], {
    refreshMs: OBSERVED_REFRESH_MS,
  })
  const districts = useAsync(getDistricts, [])
  const esi = useAsync(() => getReduction(false), [])

  const station = useMemo(() => {
    if (latest.state.status !== 'ready') return null
    if (selected) return selected
    // Default to the dirtiest station: the one a reader most wants to see.
    const ranked = [...latest.state.data.readings].sort(
      (a, b) => (b.pm25 ?? -1) - (a.pm25 ?? -1),
    )
    return ranked[0] ?? null
  }, [latest.state, selected])

  const cityPm25 = useMemo(() => {
    if (latest.state.status !== 'ready') return null
    const values = latest.state.data.readings
      .map((r) => r.pm25)
      .filter((v): v is number => v !== null)
    return values.length
      ? values.reduce((sum, v) => sum + v, 0) / values.length
      : null
  }, [latest.state])

  return (
    <div className="space-y-6">
      {/* --- 10.3: hero tiles ------------------------------------------- */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="City PM2.5 (mean)"
          value={cityPm25}
          unit="µg/m³"
          band={band(cityPm25).label}
          color={band(cityPm25).color}
          caption="Mean of the newest reading from every station."
        />

        <Async state={esi.state} height="h-32">
          {(data) => (
            <StatTile
              label="Environmental Stress Index"
              value={data.esi.latest}
              unit="/ 100"
              caption={`PCA PC1, oriented by ${data.pc1_oriented_by}. Relative to this window — not a health threshold.`}
              digits={0}
            />
          )}
        </Async>

        <Async state={latest.state} height="h-32">
          {(data) => {
            const worst = [...data.readings].sort(
              (a, b) => (b.pm25 ?? -1) - (a.pm25 ?? -1),
            )[0] as StationReading | undefined
            return (
              <StatTile
                label="Worst district now"
                value={worst?.pm25 ?? null}
                unit="µg/m³"
                band={worst?.district_name ?? worst?.station}
                color={band(worst?.pm25).color}
                caption="Highest current PM2.5 across the network."
              />
            )
          }}
        </Async>

        <Async state={latest.state} height="h-32">
          {(data) => (
            <StatTile
              label="Flagged anomalies"
              value={data.readings.filter((r) => r.is_anomaly).length}
              digits={0}
              caption="Stations whose current reading the detectors flagged. Flagged, never removed."
            />
          )}
        </Async>
      </section>

      {/* --- 10.4: the map ---------------------------------------------- */}
      <Card
        title="Where it is worse"
        subtitle={
          latest.state.status === 'ready'
            ? `Current PM2.5 by district. Click a station to retarget the forecast. ${freshness(
                latest.state.data.observed_at,
                latest.state.data.stale_minutes,
              )}`
            : 'Current PM2.5 by district. Click a station to retarget the forecast.'
        }
        note="District polygons are schematic, not surveyed administrative boundaries. A district with no monitor is shown unfilled — that is missing data, not clean air."
      >
        <Async state={districts.state} retry={districts.retry} height="h-[26rem]">
          {(districtData) => (
            <Async state={latest.state} retry={latest.retry} height="h-[26rem]">
              {(observationData) => (
                <DistrictMap
                  districts={districtData}
                  readings={observationData.readings}
                  selected={station?.station}
                  onSelect={setSelected}
                />
              )}
            </Async>
          )}
        </Async>
      </Card>

      {/* --- 10.5: the trendline ---------------------------------------- */}
      {station && <Trendline station={station} />}
    </div>
  )
}

function Trendline({ station }: { station: StationReading }) {
  const history = useAsync(
    () => getStationSeries(station.lat, station.lon, 48),
    [station.station],
    { refreshMs: OBSERVED_REFRESH_MS },
  )

  const forecasts = useAsync(
    () =>
      Promise.all(
        HORIZONS.map((horizon) =>
          predict({
            lat: station.lat,
            lon: station.lon,
            horizon,
            persist: false,
          }).catch(() => null),
        ),
      ),
    [station.station],
  )

  return (
    <Card
      title={`Recent history and forecast — ${station.district_name ?? station.station}`}
      subtitle="48 hours observed, then the horizons the platform actually trains."
      note="Forecast points are shown only at the horizons a model exists for (+1h, +24h). A continuous curve between them would draw predictions no model made. Bars are the 80% interval, calibrated on each model's own held-out errors."
    >
      <Async state={history.state} retry={history.retry}>
        {(series) => (
          <Async state={forecasts.state} retry={forecasts.retry}>
            {(predictions) => (
              <TrendPlot series={series.points} predictions={predictions} />
            )}
          </Async>
        )}
      </Async>
    </Card>
  )
}

function TrendPlot({
  series,
  predictions,
}: {
  series: { timestamp: string; pm25: number | null }[]
  predictions: (PredictionResponse | null)[]
}) {
  const made = predictions.filter((p): p is PredictionResponse => p !== null)

  const data: Plotly.Data[] = [
    {
      type: 'scatter',
      mode: 'lines',
      name: 'Observed',
      x: series.map((point) => point.timestamp),
      y: series.map((point) => point.pm25),
      line: { color: CATEGORICAL[0], width: 2 },
      hovertemplate: '%{x|%d %b %H:%M}<br>%{y:.1f} µg/m³<extra>Observed</extra>',
    },
  ]

  if (made.length) {
    data.push({
      type: 'scatter',
      mode: 'markers',
      name: 'Forecast',
      x: made.map((p) => p.target_time),
      y: made.map((p) => p.prediction),
      marker: { color: CATEGORICAL[1], size: 11, symbol: 'diamond' },
      error_y: {
        type: 'data',
        symmetric: false,
        array: made.map((p) => p.confidence_interval[1] - p.prediction),
        arrayminus: made.map((p) => p.prediction - p.confidence_interval[0]),
        color: CATEGORICAL[1],
        thickness: 2,
        width: 6,
      },
      customdata: made.map((p) => [p.horizon_hours, ...p.confidence_interval]),
      hovertemplate:
        '+%{customdata[0]}h<br>%{y:.1f} µg/m³<br>' +
        '80%: %{customdata[1]:.1f}–%{customdata[2]:.1f}<extra>Forecast</extra>',
    })
  }

  return (
    <Plot
      ariaLabel="Observed PM2.5 over the last 48 hours with forecast points"
      className="h-80 w-full"
      data={data}
      layout={baseLayout({
        hovermode: 'x unified',
        yaxis: {
          title: { text: 'PM2.5 (µg/m³)', font: { color: INK.muted } },
          gridcolor: INK.grid,
          zerolinecolor: INK.grid,
          tickfont: { color: INK.muted },
          rangemode: 'tozero',
        },
      })}
    />
  )
}
