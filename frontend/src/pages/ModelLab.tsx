/**
 * Model Lab — tasks 10.12, 10.13, 10.14.
 *
 * The table lists the **whole ladder**, baseline included. That is the point of
 * the comparison: an MAE of 6.5 µg/m³ means nothing on its own, and a table
 * showing only the winner would hide the floor it had to clear (AC-7). The
 * skill column is the honest summary — how much better than "next hour looks
 * like this hour".
 *
 * The importance chart is SHAP, so the bars are in µg/m³ and signed: they say
 * how much a feature moved *this model's* output, which is a statement about
 * the model and not about the atmosphere.
 */

import { useMemo, useState } from 'react'
import { Async, useAsync } from '../components/Async'
import Card from '../components/Card'
import Plot from '../charts/Plot'
import { CATEGORICAL, INK, SEQUENTIAL, baseLayout, band } from '../charts/theme'
import {
  getImportance,
  getLatestObservations,
  getModels,
  predict,
  type RegisteredModel,
  type StationReading,
} from '../services/api'

export default function ModelLab() {
  const models = useAsync(() => getModels(50), [])
  const [selected, setSelected] = useState<number | null>(null)

  return (
    <div className="space-y-6">
      <Async state={models.state} retry={models.retry} height="h-64" empty={(rows) => rows.length === 0} emptyMessage="No models registered yet. Run: python -m scripts.train_models">
        {(rows) => (
          <>
            <Registry rows={rows} selected={selected} onSelect={setSelected} />
            <Importance modelId={selected ?? (rows[0]?.id as number)} />
          </>
        )}
      </Async>

      <Forecast />
    </div>
  )
}

/* --- 10.12: the trained-model table ---------------------------------------- */

function Registry({
  rows,
  selected,
  onSelect,
}: {
  rows: RegisteredModel[]
  selected: number | null
  onSelect: (id: number) => void
}) {
  const baselineByTarget = useMemo(() => {
    const map = new Map<string, number>()
    for (const row of rows) {
      if (row.name === 'naive_lag1' && row.mae !== null && !map.has(row.target)) {
        map.set(row.target, row.mae)
      }
    }
    return map
  }, [rows])

  const active = selected ?? rows[0]?.id

  return (
    <Card
      title="Registered models"
      subtitle="Every rung of the ladder, newest first"
      note="R² flatters an autocorrelated series — persistence alone scores above 0.7 at one hour. Read the skill column, which is the fractional MAE improvement over that baseline."
    >
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500">
            <tr>
              {['Model', 'Target', 'MAE', 'RMSE', 'R²', 'Skill', 'Trained'].map((head) => (
                <th key={head} className="whitespace-nowrap px-3 py-2 font-medium">
                  {head}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const baseline = baselineByTarget.get(row.target)
              const skill =
                baseline && row.mae !== null && baseline > 0
                  ? 1 - row.mae / baseline
                  : null
              return (
                <tr
                  key={row.id}
                  onClick={() => onSelect(row.id)}
                  className={`cursor-pointer border-t border-slate-800 transition hover:bg-slate-800/40 ${
                    row.id === active ? 'bg-slate-800/60' : ''
                  }`}
                >
                  <td className="whitespace-nowrap px-3 py-2">
                    <span className="text-slate-200">{row.name}</span>
                    {row.name === 'naive_lag1' && (
                      <span className="ml-2 rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-400">
                        baseline
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-400">{row.target}</td>
                  <td className="px-3 py-2 tabular-nums text-slate-300">
                    {row.mae?.toFixed(3) ?? '—'}
                  </td>
                  <td className="px-3 py-2 tabular-nums text-slate-400">
                    {row.rmse?.toFixed(3) ?? '—'}
                  </td>
                  <td className="px-3 py-2 tabular-nums text-slate-400">
                    {row.r2?.toFixed(3) ?? '—'}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {skill === null ? (
                      <span className="text-slate-600">—</span>
                    ) : (
                      <span
                        className={
                          // The baseline's skill against itself is zero by
                          // definition — neutral, not a failure.
                          row.name === 'naive_lag1'
                            ? 'text-slate-500'
                            : skill > 0
                              ? 'text-emerald-400'
                              : 'text-red-400'
                        }
                      >
                        {skill > 0 ? '+' : ''}
                        {(skill * 100).toFixed(1)}%
                      </span>
                    )}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-xs text-slate-500">
                    {new Date(row.created_at).toLocaleString()}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Select a row to explain it below.
      </p>
    </Card>
  )
}

/* --- 10.13: feature importance --------------------------------------------- */

function Importance({ modelId }: { modelId: number }) {
  const importance = useAsync(() => getImportance(modelId, 300), [modelId])

  return (
    <Card
      title="What the model relies on"
      subtitle="Mean |SHAP| per feature, in µg/m³"
      note="SHAP describes how each feature moved this model's output. It is a statement about the model, not about the atmosphere: a large attribution is not evidence that the feature caused the pollution."
    >
      <Async state={importance.state} retry={importance.retry} height="h-80">
        {(data) => {
          const top = [...data.features].slice(0, 12).reverse()
          return (
            <>
              <Plot
                ariaLabel="Feature importance by mean absolute SHAP value"
                className="h-80 w-full"
                data={[
                  {
                    type: 'bar',
                    orientation: 'h',
                    x: top.map((feature) => feature.mean_abs_shap),
                    y: top.map((feature) => feature.feature),
                    marker: { color: SEQUENTIAL[3] },
                    customdata: top.map((feature) => [feature.direction, feature.share * 100]),
                    hovertemplate:
                      '%{y}<br>%{x:.3f} µg/m³ · %{customdata[1]:.1f}% of total' +
                      '<br>usually %{customdata[0]}<extra></extra>',
                  },
                ]}
                layout={baseLayout({
                  margin: { l: 180, r: 24, t: 8, b: 40 },
                  showlegend: false,
                  xaxis: {
                    title: { text: 'mean |SHAP| (µg/m³)', font: { color: INK.muted } },
                    gridcolor: INK.grid,
                    tickfont: { color: INK.muted },
                    rangemode: 'tozero',
                  },
                  yaxis: { tickfont: { color: INK.secondary }, gridcolor: 'rgba(0,0,0,0)' },
                })}
              />
              <p className="mt-2 text-xs text-slate-500">
                {data.name} · {data.target} · {data.method} · base value{' '}
                {data.base_value.toFixed(1)} µg/m³ over {data.rows} rows
              </p>
            </>
          )
        }}
      </Async>
    </Card>
  )
}

/* --- 10.14: the prediction form -------------------------------------------- */

function Forecast() {
  const stations = useAsync(getLatestObservations, [])
  const [station, setStation] = useState<StationReading | null>(null)
  const [horizon, setHorizon] = useState(1)
  const [request, setRequest] = useState(0)

  const target = station
  const forecast = useAsync(async () => {
    if (!target || request === 0) return null
    return predict({ lat: target.lat, lon: target.lon, horizon, persist: true })
  }, [request])

  return (
    <Card
      title="Make a forecast"
      subtitle="Pick a station and a horizon; the answer arrives with its reasoning"
      note="Every forecast is written to the predictions table, so its error can be measured once the hour it describes actually arrives."
    >
      <Async state={stations.state} retry={stations.retry} height="h-24">
        {(data) => (
          <form
            className="flex flex-wrap items-end gap-3"
            onSubmit={(event) => {
              event.preventDefault()
              setStation((current) => current ?? data.readings[0] ?? null)
              setRequest((value) => value + 1)
            }}
          >
            <label className="text-xs text-slate-400">
              <span className="mb-1 block">Station</span>
              <select
                className="w-56 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-200"
                value={station?.station ?? data.readings[0]?.station}
                onChange={(event) =>
                  setStation(
                    data.readings.find((r) => r.station === event.target.value) ?? null,
                  )
                }
              >
                {data.readings.map((reading) => (
                  <option key={reading.station} value={reading.station}>
                    {reading.district_name ?? reading.station}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-xs text-slate-400">
              <span className="mb-1 block">Horizon</span>
              <select
                className="w-32 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-200"
                value={horizon}
                onChange={(event) => setHorizon(Number(event.target.value))}
              >
                <option value={1}>+1 hour</option>
                <option value={6}>+6 hours</option>
                <option value={24}>+24 hours</option>
              </select>
            </label>

            <button
              type="submit"
              className="rounded bg-sky-600 px-4 py-1.5 text-sm font-medium text-white transition hover:bg-sky-500"
            >
              Forecast
            </button>
          </form>
        )}
      </Async>

      {request > 0 && (
        <div className="mt-4">
          <Async state={forecast.state} retry={forecast.retry} height="h-40">
            {(result) =>
              result ? <ForecastResult result={result} /> : <p className="text-sm text-slate-500">Pick a station.</p>
            }
          </Async>
        </div>
      )}
    </Card>
  )
}

function ForecastResult({
  result,
}: {
  result: NonNullable<Awaited<ReturnType<typeof predict>>>
}) {
  const features = Object.entries(result.top_features)

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,18rem)_1fr]">
      <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        <p className="text-xs uppercase tracking-wide text-slate-500">
          PM2.5 at {new Date(result.target_time).toLocaleString()}
        </p>
        <p className="mt-2 flex items-baseline gap-1.5">
          <span className="text-4xl font-semibold tabular-nums text-slate-100">
            {result.prediction.toFixed(1)}
          </span>
          <span className="text-sm text-slate-500">{result.unit}</span>
        </p>
        <p className="mt-2 flex items-center gap-2 text-xs">
          <span
            aria-hidden="true"
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: band(result.prediction).color }}
          />
          <span className="text-slate-300">{band(result.prediction).label}</span>
        </p>
        <p className="mt-3 text-xs text-slate-400">
          {(result.coverage * 100).toFixed(0)}% interval{' '}
          <span className="tabular-nums text-slate-300">
            {result.confidence_interval[0].toFixed(1)}–
            {result.confidence_interval[1].toFixed(1)}
          </span>
        </p>
        <p className="mt-1 text-xs text-slate-600">{result.interval_method}</p>
        <p className="mt-3 text-xs text-slate-500">
          {result.model.name} · trained{' '}
          {result.model.trained_at
            ? new Date(result.model.trained_at).toLocaleDateString()
            : '—'}
        </p>
      </div>

      <div>
        <p className="mb-2 text-xs text-slate-400">
          Why — signed SHAP contributions from a base value of{' '}
          {result.base_value.toFixed(1)} µg/m³
        </p>
        <Plot
          ariaLabel="Feature contributions to this forecast"
          className="h-56 w-full"
          data={[
            {
              type: 'bar',
              orientation: 'h',
              x: features.map((entry) => entry[1]).reverse(),
              y: features.map((entry) => entry[0]).reverse(),
              marker: {
                color: features
                  .map((entry) =>
                    entry[1] >= 0 ? (CATEGORICAL[1] as string) : (CATEGORICAL[0] as string),
                  )
                  .reverse(),
              },
              hovertemplate: '%{y}<br>%{x:+.2f} µg/m³<extra></extra>',
            },
          ]}
          layout={baseLayout({
            margin: { l: 180, r: 24, t: 8, b: 36 },
            showlegend: false,
            xaxis: {
              title: { text: 'contribution (µg/m³)', font: { color: INK.muted } },
              gridcolor: INK.grid,
              zerolinecolor: INK.secondary,
              tickfont: { color: INK.muted },
            },
            yaxis: { tickfont: { color: INK.secondary }, gridcolor: 'rgba(0,0,0,0)' },
          })}
        />
        <p className="mt-1 text-xs text-slate-500">
          Orange pushed the forecast up, blue pulled it down.
        </p>
      </div>
    </div>
  )
}
