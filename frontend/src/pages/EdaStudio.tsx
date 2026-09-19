/**
 * EDA Studio — tasks 10.6 to 10.11 (BACSE301 Modules 3-5).
 *
 * Every panel renders something the backend computed; nothing is derived in the
 * browser. That is the point of the split: the statistics are tested in Python,
 * and this file is responsible only for reading them honestly.
 *
 * Two encodings recur and both are deliberate:
 *
 * - **Correlation uses a diverging ramp with a neutral grey midpoint.**
 *   Correlation has a sign, so a one-hue ramp would make −0.8 and +0.8 look
 *   alike, and a rainbow would invent an ordering.
 * - **Missingness and STL use small multiples, never a shared axis.** Trend,
 *   seasonal and residual live on wildly different scales; one pair of axes for
 *   all three is a dual-axis chart in disguise.
 */

import { useState } from 'react'
import { Async, useAsync } from '../components/Async'
import Card from '../components/Card'
import Plot from '../charts/Plot'
import {
  CATEGORICAL,
  DIVERGING,
  INK,
  SEQUENTIAL,
  baseLayout,
  label as columnLabel,
} from '../charts/theme'
import {
  getDecomposition,
  getProfile,
  getProjection,
  getStationSeries,
  getLatestObservations,
  type ProfileResponse,
} from '../services/api'

export default function EdaStudio() {
  const profile = useAsync(getProfile, [])

  return (
    <div className="space-y-6">
      <Async state={profile.state} retry={profile.retry} height="h-96">
        {(data) => (
          <>
            <Missingness profile={data} />
            <div className="grid gap-6 lg:grid-cols-2">
              <Distributions profile={data} />
              <Correlation profile={data} />
            </div>
          </>
        )}
      </Async>

      <Decomposition />

      <div className="grid gap-6 lg:grid-cols-2">
        <ParallelCoordinates />
        <Clusters />
      </div>
    </div>
  )
}

/* --- 10.6: missingness matrix --------------------------------------------- */

function Missingness({ profile }: { profile: ProfileResponse }) {
  const columns = profile.univariate.map((stat) => stat.column)

  return (
    <Card
      title="Missingness"
      subtitle={`${profile.rows.toLocaleString()} rows profiled`}
      note="Missingness is a signal, not a defect: the Phase 3 engine characterises each column as MCAR, MAR or MNAR before repairing it, and never deletes a row to make a chart tidier."
    >
      <Plot
        ariaLabel="Share of missing values per column"
        className="h-56 w-full"
        data={[
          {
            type: 'bar',
            orientation: 'h',
            x: profile.univariate.map((stat) => stat.missing_pct),
            y: columns.map(columnLabel),
            marker: { color: SEQUENTIAL[3] },
            text: profile.univariate.map(
              (stat) => `${stat.missing.toLocaleString()} (${stat.missing_pct.toFixed(2)}%)`,
            ),
            textposition: 'outside',
            textfont: { color: INK.secondary },
            hovertemplate: '%{y}<br>%{x:.2f}% missing<extra></extra>',
          },
        ]}
        layout={baseLayout({
          // Generous right margin *and* headroom on the axis: the counts are
          // drawn outside the bar, and the longest bar otherwise pushes its own
          // label off the edge.
          margin: { l: 110, r: 130, t: 8, b: 40 },
          xaxis: {
            title: { text: '% missing', font: { color: INK.muted } },
            gridcolor: INK.grid,
            tickfont: { color: INK.muted },
            rangemode: 'tozero',
            range: [
              0,
              Math.max(...profile.univariate.map((stat) => stat.missing_pct)) * 1.35,
            ],
          },
          yaxis: { tickfont: { color: INK.secondary }, gridcolor: 'rgba(0,0,0,0)' },
          showlegend: false,
        })}
      />
    </Card>
  )
}

/* --- 10.7: histograms and boxplots ---------------------------------------- */

function Distributions({ profile }: { profile: ProfileResponse }) {
  const [column, setColumn] = useState('pm25')
  const stat = profile.univariate.find((entry) => entry.column === column)
  const shape = profile.distributions.find((entry) => entry.column === column)

  const sample = useAsync(async () => {
    const latest = await getLatestObservations()
    const busiest = latest.readings[0]
    if (!busiest) return []
    const series = await getStationSeries(busiest.lat, busiest.lon, 720)
    return series.points
  }, [])

  return (
    <Card
      title="Distribution"
      subtitle={shape?.shape ?? ''}
      note={shape?.rationale}
      actions={
        <select
          value={column}
          onChange={(event) => setColumn(event.target.value)}
          className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200"
          aria-label="Column to profile"
        >
          {profile.univariate.map((entry) => (
            <option key={entry.column} value={entry.column}>
              {columnLabel(entry.column)}
            </option>
          ))}
        </select>
      }
    >
      <Async state={sample.state} retry={sample.retry} empty={(rows) => rows.length === 0}>
        {(rows) => {
          const values = rows
            .map((row) => row[column as 'pm25'])
            .filter((value): value is number => value !== null)

          return (
            <div className="space-y-2">
              <Plot
                ariaLabel={`Histogram of ${columnLabel(column)}`}
                className="h-52 w-full"
                data={[
                  {
                    type: 'histogram',
                    x: values,
                    marker: { color: CATEGORICAL[0], line: { color: '#0f172a', width: 1 } },
                    // @types/plotly.js omits nbinsx from PlotData; the option is
                    // real and documented, so the cast is on the property.
                    ...({ nbinsx: 28 } as object),
                    hovertemplate: '%{x}<br>%{y} readings<extra></extra>',
                  },
                ]}
                layout={baseLayout({
                  margin: { l: 48, r: 12, t: 8, b: 36 },
                  showlegend: false,
                  bargap: 0.04,
                })}
              />
              <Plot
                ariaLabel={`Boxplot of ${columnLabel(column)}`}
                className="h-24 w-full"
                data={[
                  {
                    type: 'box',
                    x: values,
                    marker: { color: CATEGORICAL[0] },
                    line: { color: CATEGORICAL[0] },
                    fillcolor: 'rgba(57,135,229,0.25)',
                    boxpoints: 'suspectedoutliers',
                    hovertemplate: '%{x:.1f}<extra></extra>',
                  },
                ]}
                layout={baseLayout({
                  margin: { l: 48, r: 12, t: 4, b: 28 },
                  showlegend: false,
                  yaxis: { showticklabels: false, gridcolor: 'rgba(0,0,0,0)' },
                })}
              />
              {stat && (
                <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
                  {[
                    ['Mean', stat.mean],
                    ['Median', stat.median],
                    ['IQR', stat.iqr],
                    ['Skew', stat.skewness],
                  ].map(([name, value]) => (
                    <div key={String(name)} className="flex justify-between gap-2">
                      <dt className="text-slate-500">{name}</dt>
                      <dd className="tabular-nums text-slate-300">
                        {value === null || value === undefined
                          ? '—'
                          : Number(value).toFixed(2)}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          )
        }}
      </Async>
    </Card>
  )
}

/* --- 10.8: correlation heatmap -------------------------------------------- */

function Correlation({ profile }: { profile: ProfileResponse }) {
  const [method, setMethod] = useState<'pearson' | 'spearman'>('pearson')
  const columns = profile.bivariate.columns
  const matrix = profile.bivariate[method]

  const z = columns.map((row) => columns.map((col) => matrix[row]?.[col] ?? null))

  return (
    <Card
      title="Correlation"
      subtitle="Every cell labelled — a heatmap read by colour alone is unreadable in greyscale"
      note={profile.bivariate.caveat}
      actions={
        <select
          value={method}
          onChange={(event) => setMethod(event.target.value as 'pearson' | 'spearman')}
          className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200"
          aria-label="Correlation method"
        >
          <option value="pearson">Pearson</option>
          <option value="spearman">Spearman</option>
        </select>
      }
    >
      <Plot
        ariaLabel={`${method} correlation matrix`}
        className="h-80 w-full"
        data={[
          {
            type: 'heatmap',
            z,
            x: columns.map(columnLabel),
            y: columns.map(columnLabel),
            zmin: -1,
            zmax: 1,
            colorscale: DIVERGING,
            xgap: 2,
            ygap: 2,
            colorbar: {
              tickfont: { color: INK.muted },
              outlinewidth: 0,
              thickness: 10,
            },
            hovertemplate: '%{y} ↔ %{x}<br>r = %{z:.2f}<extra></extra>',
          },
        ]}
        layout={baseLayout({
          margin: { l: 96, r: 12, t: 8, b: 76 },
          annotations: columns.flatMap((row, i) =>
            columns.map((col, j) => {
              const value = z[i]?.[j] ?? null
              return {
                x: columnLabel(col),
                y: columnLabel(row),
                text: value === null ? '—' : value.toFixed(2),
                showarrow: false,
                font: {
                  size: 11,
                  // White only on the dark ends of the ramp, where the ink
                  // colour would fall under 4.5:1.
                  color: Math.abs(value ?? 0) > 0.7 ? '#ffffff' : INK.primary,
                },
              }
            }),
          ),
          xaxis: { tickangle: -30, tickfont: { color: INK.muted }, gridcolor: 'rgba(0,0,0,0)' },
          yaxis: { tickfont: { color: INK.muted }, gridcolor: 'rgba(0,0,0,0)' },
        })}
      />
    </Card>
  )
}

/* --- 10.10: STL overlays --------------------------------------------------- */

function Decomposition() {
  const stl = useAsync(() => getDecomposition({ column: 'pm25', max_points: 336 }), [])

  return (
    <Card
      title="Seasonal-trend decomposition (STL)"
      subtitle="PM2.5 split into what it usually is, what the day does to it, and what is left"
      note="Each panel keeps its own scale. Three components on one pair of axes would be a dual-axis chart in disguise — the residual is an order of magnitude smaller than the trend."
    >
      <Async state={stl.state} retry={stl.retry} height="h-96">
        {(data) => {
          const panels: [string, number[], string][] = [
            ['Observed', data.observed, CATEGORICAL[0]],
            ['Trend', data.trend, CATEGORICAL[1]],
            ['Seasonal', data.seasonal, CATEGORICAL[2]],
            ['Residual', data.residual, INK.muted],
          ]

          return (
            <div className="space-y-1">
              {panels.map(([name, values, color], index) => (
                <Plot
                  key={name}
                  ariaLabel={`STL ${name} component`}
                  className="h-28 w-full"
                  data={[
                    {
                      type: 'scatter',
                      mode: 'lines',
                      name,
                      x: data.timestamps,
                      y: values,
                      line: { color, width: 2 },
                      hovertemplate: `%{x|%d %b %H:%M}<br>%{y:.1f}<extra>${name}</extra>`,
                    },
                  ]}
                  layout={baseLayout({
                    margin: { l: 56, r: 12, t: 16, b: index === panels.length - 1 ? 32 : 8 },
                    showlegend: false,
                    height: 112,
                    title: {
                      text: name,
                      font: { size: 11, color: INK.secondary },
                      x: 0,
                      xanchor: 'left',
                      y: 0.98,
                    },
                    xaxis: {
                      showticklabels: index === panels.length - 1,
                      gridcolor: INK.grid,
                      tickfont: { color: INK.muted },
                    },
                  })}
                />
              ))}
              <p className="pt-1 text-xs text-slate-500">
                Trend strength {data.strength.trend.toFixed(2)} · seasonal strength{' '}
                {data.strength.seasonal.toFixed(2)} · {data.interpolated_points}{' '}
                interpolated of {data.points_analysed} points
              </p>
            </div>
          )
        }}
      </Async>
    </Card>
  )
}

/* --- 10.9: parallel coordinates -------------------------------------------- */

function ParallelCoordinates() {
  const rows = useAsync(async () => {
    const latest = await getLatestObservations()
    const station = latest.readings[0]
    if (!station) return []
    const series = await getStationSeries(station.lat, station.lon, 500)
    return series.points.filter(
      (point) =>
        point.pm25 !== null &&
        point.pm10 !== null &&
        point.temp !== null &&
        point.humidity !== null &&
        point.traffic_score !== null,
    )
  }, [])

  return (
    <Card
      title="Parallel coordinates"
      subtitle="Each line is one hour, crossing all five measurements"
      note="Lines are coloured by PM2.5 so the high-pollution hours can be followed across the other axes. Crossing lines between two axes mean the pair is inversely related; parallel lines mean they move together."
    >
      <Async state={rows.state} retry={rows.retry} empty={(data) => data.length === 0}>
        {(data) => (
          <Plot
            ariaLabel="Parallel coordinates across the five measurements"
            className="h-80 w-full"
            data={[
              {
                type: 'parcoords',
                line: {
                  color: data.map((row) => row.pm25 as number),
                  colorscale: SEQUENTIAL.map((hex, index) => [
                    index / (SEQUENTIAL.length - 1),
                    hex,
                  ]) as [number, string][],
                  showscale: true,
                  colorbar: {
                    tickfont: { color: INK.muted },
                    outlinewidth: 0,
                    thickness: 10,
                    title: { text: 'PM2.5', font: { color: INK.muted } },
                  },
                },
                dimensions: (
                  ['pm25', 'pm10', 'temp', 'humidity', 'traffic_score'] as const
                ).map((key) => ({
                  label: columnLabel(key),
                  values: data.map((row) => row[key] as number),
                })),
              } as never,
            ]}
            // The axis names are drawn above each dimension, so the top margin
            // has to leave room for them or they are clipped by the container.
            layout={baseLayout({ margin: { l: 60, r: 70, t: 56, b: 24 } })}
          />
        )}
      </Async>
    </Card>
  )
}

/* --- 10.11: t-SNE cluster scatter ------------------------------------------ */

function Clusters() {
  const projection = useAsync(() => getProjection(1000), [])

  return (
    <Card
      title="t-SNE clusters"
      subtitle="Hours that resemble each other, coloured by hour of day"
      note="The axes have no units and no meaning. Which points sit together is informative; the distance between clusters, their sizes and the plot's orientation are artefacts of the optimisation. This projection never feeds a model."
    >
      <Async state={projection.state} retry={projection.retry}>
        {(data) => (
          <Plot
            ariaLabel="Two-dimensional t-SNE projection coloured by hour of day"
            className="h-80 w-full"
            data={[
              {
                type: 'scattergl',
                mode: 'markers',
                x: data.x,
                y: data.y,
                marker: {
                  size: 6,
                  color: data.hour_of_day,
                  // Hour of day is cyclic, so the ramp runs light→dark→light
                  // rather than implying midnight and 23:00 are far apart.
                  colorscale: [
                    [0, '#cde2fb'],
                    [0.5, '#0d366b'],
                    [1, '#cde2fb'],
                  ] as [number, string][],
                  colorbar: {
                    tickfont: { color: INK.muted },
                    outlinewidth: 0,
                    thickness: 10,
                    title: { text: 'Hour', font: { color: INK.muted } },
                  },
                  line: { color: '#0f172a', width: 0.5 },
                },
                customdata: data.timestamps.map((time, index) => [
                  time,
                  data.stations[index] ?? '',
                ]),
                hovertemplate:
                  '%{customdata[0]|%d %b %H:%M}<br>%{customdata[1]}<extra></extra>',
              },
            ]}
            layout={baseLayout({
              margin: { l: 24, r: 12, t: 8, b: 24 },
              xaxis: { showticklabels: false, gridcolor: INK.grid, zeroline: false },
              yaxis: { showticklabels: false, gridcolor: INK.grid, zeroline: false },
              showlegend: false,
            })}
          />
        )}
      </Async>
    </Card>
  )
}
