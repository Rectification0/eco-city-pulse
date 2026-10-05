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
 *
 * The Phase 12 cards (scatter, grouped bars and boxes, pair plot, Andrews
 * curves — specs §6.4) follow the same rule: the backend sends a sample of
 * points plus statistics computed on every row, and the card says which is
 * which. Flagged anomalies get a different marker *shape* as well as colour, so
 * they stay distinguishable without colour (AC-5, VIZ-6).
 */

import { useState } from 'react'
import { Async, useAsync } from '../components/Async'
import Card from '../components/Card'
import Plot from '../charts/Plot'
import {
  CATEGORICAL,
  COLUMN_UNITS,
  DIVERGING,
  INK,
  SEQUENTIAL,
  baseLayout,
  groupColors,
  label as columnLabel,
} from '../charts/theme'
import {
  getAndrews,
  getDecomposition,
  getGrouped,
  getPairPlot,
  getProfile,
  getProjection,
  getScatter,
  getStationSeries,
  getLatestObservations,
  type AndrewsClassBy,
  type Grouping,
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

      <div className="grid gap-6 lg:grid-cols-2">
        <Scatter />
        <Grouped />
      </div>

      <Decomposition />

      <div className="grid gap-6 lg:grid-cols-2">
        <ParallelCoordinates />
        <Clusters />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <PairPlot />
        <AndrewsCurves />
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

/* --- Phase 12: shared pieces ---------------------------------------------- */

const MEASURES = ['pm25', 'pm10', 'temp', 'humidity', 'traffic_score'] as const

const GROUPING_LABELS: Record<Grouping, string> = {
  hour_of_day: 'Hour of day (IST)',
  time_of_day: 'Time of day',
  day_of_week: 'Day of week',
  is_weekend: 'Weekday / weekend',
  month: 'Month',
  station: 'Station',
  pm25_band: 'PM2.5 band',
}

const GROUPINGS = Object.keys(GROUPING_LABELS) as Grouping[]

const ANDREWS_CLASSES: AndrewsClassBy[] = ['time_of_day', 'pm25_band', 'is_weekend', 'month']

/** Flagged readings: a different *shape* and the second categorical slot. */
const FLAGGED = { symbol: 'x', color: CATEGORICAL[1] } as const

interface SelectProps<T extends string> {
  value: T
  options: readonly T[]
  onChange: (value: T) => void
  label: (value: T) => string
  ariaLabel: string
}

function Select<T extends string>({ value, options, onChange, label, ariaLabel }: SelectProps<T>) {
  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value as T)}
      className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200"
      aria-label={ariaLabel}
    >
      {options.map((option) => (
        <option key={option} value={option}>
          {label(option)}
        </option>
      ))}
    </select>
  )
}

function axisTitle(column: string): string {
  const unit = COLUMN_UNITS[column]
  return unit ? `${columnLabel(column)} (${unit})` : columnLabel(column)
}

function fixed(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits)
}

/** Colour swatches *with* their labels, plus the flagged-anomaly marker. */
function Legend({ colors }: { colors: [string, string][] }) {
  return (
    <ul className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-400">
      {colors.map(([name, color]) => (
        <li key={name} className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: color }} />
          {name}
        </li>
      ))}
      <li className="flex items-center gap-1.5">
        <span className="font-bold" style={{ color: FLAGGED.color }}>
          ×
        </span>
        flagged anomaly
      </li>
    </ul>
  )
}

/* --- 12.9 / VIZ-1: scatter plot ------------------------------------------- */

function Scatter() {
  const [x, setX] = useState<string>('traffic_score')
  const [y, setY] = useState<string>('pm25')
  const [colorBy, setColorBy] = useState<Grouping | 'none'>('none')

  const scatter = useAsync(
    () => getScatter({ x, y, color_by: colorBy === 'none' ? null : colorBy }),
    [x, y, colorBy],
  )

  return (
    <Card
      title="Scatter plot"
      subtitle="Two measurements, every hour a point, with a least-squares line"
      note={scatter.state.status === 'ready' ? scatter.state.data.caveats.join(' ') : undefined}
      actions={
        <div className="flex flex-wrap gap-1.5">
          <Select
            value={x}
            options={MEASURES.filter((column) => column !== y)}
            onChange={setX}
            label={columnLabel}
            ariaLabel="Scatter x axis"
          />
          <Select
            value={y}
            options={MEASURES.filter((column) => column !== x)}
            onChange={setY}
            label={columnLabel}
            ariaLabel="Scatter y axis"
          />
          <Select
            value={colorBy}
            options={['none', ...GROUPINGS] as (Grouping | 'none')[]}
            onChange={setColorBy}
            label={(value) => (value === 'none' ? 'No colour' : GROUPING_LABELS[value])}
            ariaLabel="Colour points by"
          />
        </div>
      }
    >
      <Async state={scatter.state} retry={scatter.retry} height="h-80">
        {(data) => {
          const indices = data.x.map((_, index) => index)
          const normal = indices.filter((i) => !data.is_anomaly[i])
          const flagged = indices.filter((i) => data.is_anomaly[i])
          const pick = (chosen: number[], values: (number | null)[]) =>
            chosen.map((i) => values[i] ?? null)

          const colors =
            data.color_by && data.categories ? groupColors(data.color_by, data.categories) : {}
          const series =
            data.color_by && data.categories
              ? data.categories.map((name) => ({
                  name,
                  color: colors[name] ?? INK.muted,
                  indices: normal.filter((i) => data.groups?.[i] === name),
                }))
              : [{ name: 'Reading', color: CATEGORICAL[0], indices: normal }]

          return (
            <div className="space-y-2">
              <Plot
                ariaLabel={`Scatter of ${columnLabel(data.y_column)} against ${columnLabel(
                  data.x_column,
                )} with a least-squares line`}
                className="h-80 w-full"
                data={[
                  ...series.map((entry) => ({
                    type: 'scattergl' as const,
                    mode: 'markers' as const,
                    name: entry.name,
                    x: pick(entry.indices, data.x),
                    y: pick(entry.indices, data.y),
                    marker: { size: 4, color: entry.color, opacity: 0.6 },
                    hovertemplate: `%{x:.1f}, %{y:.1f}<extra>${entry.name}</extra>`,
                  })),
                  {
                    type: 'scattergl',
                    mode: 'markers',
                    name: 'Flagged anomaly',
                    x: pick(flagged, data.x),
                    y: pick(flagged, data.y),
                    marker: { size: 7, symbol: FLAGGED.symbol, color: FLAGGED.color },
                    hovertemplate: '%{x:.1f}, %{y:.1f}<extra>flagged</extra>',
                  },
                  {
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Least-squares line',
                    x: data.line_x,
                    y: data.line_y,
                    line: { color: INK.primary, width: 2 },
                    hoverinfo: 'skip',
                  },
                ]}
                layout={baseLayout({
                  margin: { l: 56, r: 12, t: 8, b: 72 },
                  xaxis: {
                    title: { text: axisTitle(data.x_column), font: { color: INK.muted } },
                    gridcolor: INK.grid,
                    tickfont: { color: INK.muted },
                  },
                  yaxis: {
                    title: { text: axisTitle(data.y_column), font: { color: INK.muted } },
                    gridcolor: INK.grid,
                    tickfont: { color: INK.muted },
                  },
                  legend: { orientation: 'h', y: -0.28, x: 0, font: { color: INK.secondary } },
                })}
              />
              <p className="text-xs tabular-nums text-slate-400">
                r = {fixed(data.pearson)} · ρ = {fixed(data.spearman)} · r² ={' '}
                {fixed(data.r_squared, 3)} · slope {fixed(data.slope, 3)} · n ={' '}
                {data.rows_used.toLocaleString()}
                {data.sampled &&
                  ` · ${data.points_returned.toLocaleString()} points drawn, sampled evenly`}
                {` · ${data.anomalies_in_rows.toLocaleString()} flagged`}
              </p>
            </div>
          )
        }}
      </Async>
    </Card>
  )
}

/* --- 12.9 / VIZ-2, VIZ-3: grouped bars and grouped boxplot ---------------- */

function Grouped() {
  const [measure, setMeasure] = useState<string>('pm25')
  const [groupBy, setGroupBy] = useState<Grouping>('hour_of_day')
  const [splitBy, setSplitBy] = useState<Grouping | 'none'>('none')

  const grouped = useAsync(
    () =>
      getGrouped({ measure, group_by: groupBy, split_by: splitBy === 'none' ? null : splitBy }),
    [measure, groupBy, splitBy],
  )

  return (
    <Card
      title="Grouped comparison"
      subtitle="Mean with 95% interval per group, then the full spread as boxes"
      note={grouped.state.status === 'ready' ? grouped.state.data.caveats.join(' ') : undefined}
      actions={
        <div className="flex flex-wrap gap-1.5">
          <Select
            value={measure}
            options={MEASURES}
            onChange={setMeasure}
            label={columnLabel}
            ariaLabel="Measurement to compare"
          />
          <Select
            value={groupBy}
            options={GROUPINGS}
            onChange={(value) => {
              setGroupBy(value)
              if (value === splitBy) setSplitBy('none')
            }}
            label={(value) => GROUPING_LABELS[value]}
            ariaLabel="Group by"
          />
          <Select
            value={splitBy}
            options={
              ['none', ...GROUPINGS.filter((value) => value !== groupBy)] as (Grouping | 'none')[]
            }
            onChange={setSplitBy}
            label={(value) => (value === 'none' ? 'No split' : `Split: ${GROUPING_LABELS[value]}`)}
            ariaLabel="Split bars by"
          />
        </div>
      }
    >
      <Async state={grouped.state} retry={grouped.retry} height="h-96">
        {(data) => {
          const splits = data.split_categories ?? [null]
          const colors = data.split_by
            ? groupColors(data.split_by, data.split_categories ?? [])
            : {}
          const thin = data.bars.filter((bar) => bar.thin).length
          const pastWhisker = data.boxes.reduce((total, box) => total + box.whisker_outliers, 0)

          return (
            <div className="space-y-2">
              <Plot
                ariaLabel={`Mean ${columnLabel(data.measure)} by ${
                  GROUPING_LABELS[data.group_by]
                } with 95% confidence intervals`}
                className="h-56 w-full"
                data={splits.map((split) => {
                  const cells = data.categories.map((group) =>
                    data.bars.find((bar) => bar.group === group && bar.split === split),
                  )
                  return {
                    type: 'bar' as const,
                    name: split ?? columnLabel(data.measure),
                    x: data.categories,
                    y: cells.map((cell) => cell?.mean ?? null),
                    marker: {
                      color: split ? (colors[split] ?? INK.muted) : CATEGORICAL[0],
                      // Thin cells fade: their interval is too wide to compare.
                      opacity: cells.map((cell) => (cell?.thin ? 0.35 : 1)),
                    },
                    error_y: {
                      type: 'data' as const,
                      symmetric: false,
                      array: cells.map((cell) =>
                        cell?.ci_high != null && cell.mean != null ? cell.ci_high - cell.mean : 0,
                      ),
                      arrayminus: cells.map((cell) =>
                        cell?.ci_low != null && cell.mean != null ? cell.mean - cell.ci_low : 0,
                      ),
                      color: INK.secondary,
                      thickness: 1,
                      width: 2,
                    },
                    customdata: cells.map((cell) => [cell?.n ?? 0, cell?.thin ? ' (thin)' : '']),
                    hovertemplate: `%{x}${
                      split ? ` · ${split}` : ''
                    }<br>mean %{y:.1f}<br>n = %{customdata[0]}%{customdata[1]}<extra></extra>`,
                  }
                }) as never}
                layout={baseLayout({
                  barmode: 'group',
                  bargap: 0.2,
                  margin: { l: 56, r: 12, t: 8, b: data.split_by ? 64 : 40 },
                  showlegend: Boolean(data.split_by),
                  yaxis: {
                    title: { text: axisTitle(data.measure), font: { color: INK.muted } },
                    gridcolor: INK.grid,
                    tickfont: { color: INK.muted },
                    rangemode: 'tozero',
                  },
                  xaxis: { type: 'category', tickfont: { color: INK.muted } },
                })}
              />
              <Plot
                ariaLabel={`Boxplot of ${columnLabel(data.measure)} by ${
                  GROUPING_LABELS[data.group_by]
                }`}
                className="h-56 w-full"
                data={[
                  // Plotly draws a box from precomputed quartiles, so the full
                  // distribution never has to reach the browser.
                  {
                    type: 'box',
                    name: columnLabel(data.measure),
                    x: data.boxes.map((box) => box.group),
                    q1: data.boxes.map((box) => box.q1),
                    median: data.boxes.map((box) => box.median),
                    q3: data.boxes.map((box) => box.q3),
                    lowerfence: data.boxes.map((box) => box.lower_fence),
                    upperfence: data.boxes.map((box) => box.upper_fence),
                    mean: data.boxes.map((box) => box.mean),
                    boxpoints: false,
                    marker: { color: CATEGORICAL[0] },
                    line: { color: CATEGORICAL[0], width: 1.2 },
                    fillcolor: 'rgba(57,135,229,0.25)',
                  } as never,
                  {
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Past the whisker',
                    x: data.boxes.flatMap((box) => box.outliers.map(() => box.group)),
                    y: data.boxes.flatMap((box) => box.outliers),
                    marker: { size: 5, symbol: 'circle-open', color: INK.secondary },
                    hovertemplate: '%{x}<br>%{y:.1f}<extra>past the whisker</extra>',
                  },
                ]}
                layout={baseLayout({
                  margin: { l: 56, r: 12, t: 8, b: 40 },
                  showlegend: false,
                  yaxis: {
                    title: { text: axisTitle(data.measure), font: { color: INK.muted } },
                    gridcolor: INK.grid,
                    tickfont: { color: INK.muted },
                  },
                  xaxis: { type: 'category', tickfont: { color: INK.muted } },
                })}
              />
              <p className="text-xs tabular-nums text-slate-400">
                n = {data.rows_used.toLocaleString()} · {pastWhisker.toLocaleString()} past a
                whisker · {data.anomalies_in_rows.toLocaleString()} flagged by the quality engine
                {thin > 0 &&
                  ` · ${thin} faded bar${thin === 1 ? '' : 's'} with n < ${data.thin_threshold}`}
              </p>
            </div>
          )
        }}
      </Async>
    </Card>
  )
}

/* --- 12.9 / VIZ-4: pair plot ---------------------------------------------- */

function PairPlot() {
  const [colorBy, setColorBy] = useState<Grouping>('pm25_band')
  const pairs = useAsync(() => getPairPlot(colorBy), [colorBy])

  return (
    <Card
      title="Pair plot"
      subtitle="Every measurement against every other — the heatmap's cells, drawn"
      note={pairs.state.status === 'ready' ? pairs.state.data.caveats.join(' ') : undefined}
      actions={
        <Select
          value={colorBy}
          options={GROUPINGS}
          onChange={setColorBy}
          label={(value) => `Colour: ${GROUPING_LABELS[value]}`}
          ariaLabel="Colour pair plot by"
        />
      }
    >
      <Async state={pairs.state} retry={pairs.retry} height="h-[28rem]">
        {(data) => {
          const colors = groupColors(data.color_by, data.categories)
          // splom names its axes x, x2, x3 … and y, y2, y3 …
          const axis = (index: number) => (index === 0 ? '' : String(index + 1))
          const axisStyle = {
            gridcolor: INK.grid,
            tickfont: { color: INK.muted, size: 9 },
            zeroline: false,
          }

          return (
            <div className="space-y-2">
              <Plot
                ariaLabel={`Pair plot of ${data.columns.length} measurements coloured by ${
                  GROUPING_LABELS[data.color_by]
                }`}
                className="h-[28rem] w-full"
                data={[
                  {
                    type: 'splom',
                    dimensions: data.columns.map((column) => ({
                      label: columnLabel(column),
                      values: data.values[column],
                    })),
                    // The upper triangle repeats the lower, and each column's
                    // histogram is already on the Distribution card.
                    showupperhalf: false,
                    diagonal: { visible: false },
                    marker: {
                      size: data.is_anomaly.map((flag) => (flag ? 6 : 3.5)),
                      symbol: data.is_anomaly.map((flag) => (flag ? FLAGGED.symbol : 'circle')),
                      color: data.is_anomaly.map((flag, index) =>
                        flag ? FLAGGED.color : (colors[data.groups[index] ?? ''] ?? INK.muted),
                      ),
                      opacity: 0.7,
                      line: { width: 0 },
                    },
                    text: data.groups.map((group) => group ?? ''),
                    hovertemplate: '%{x:.1f}, %{y:.1f}<br>%{text}<extra></extra>',
                  } as never,
                ]}
                layout={baseLayout({
                  margin: { l: 64, r: 8, t: 8, b: 56 },
                  showlegend: false,
                  ...Object.fromEntries(
                    data.columns.flatMap((_, index) => [
                      [`xaxis${axis(index)}`, axisStyle],
                      [`yaxis${axis(index)}`, axisStyle],
                    ]),
                  ),
                  // Each panel carries its r, computed on every row — the same
                  // number as the heatmap cell, so the two views cannot disagree.
                  annotations: data.columns.flatMap((row, i) =>
                    data.columns.slice(0, i).map((column, j) => ({
                      xref: `x${axis(j)} domain` as never,
                      yref: `y${axis(i)} domain` as never,
                      x: 0.03,
                      y: 0.97,
                      xanchor: 'left' as const,
                      yanchor: 'top' as const,
                      showarrow: false,
                      text: `r ${fixed(data.pearson[row]?.[column])}`,
                      font: { size: 10, color: INK.primary },
                      bgcolor: 'rgba(15,23,42,0.7)',
                    })),
                  ),
                })}
              />
              <Legend colors={data.categories.map((name) => [name, colors[name] ?? INK.muted])} />
              <p className="text-xs tabular-nums text-slate-400">
                {data.points_returned.toLocaleString()} of {data.rows_used.toLocaleString()}{' '}
                complete rows drawn{data.sampled ? ', sampled evenly' : ''} ·{' '}
                {data.anomalies_in_points.toLocaleString()} flagged among them
              </p>
            </div>
          )
        }}
      </Async>
    </Card>
  )
}

/* --- 12.9 / VIZ-5: Andrews curves ----------------------------------------- */

function AndrewsCurves() {
  const [classBy, setClassBy] = useState<AndrewsClassBy>('time_of_day')
  const andrews = useAsync(() => getAndrews(classBy), [classBy])

  return (
    <Card
      title="Andrews curves"
      subtitle="Each hour's standardised readings as one curve; bold lines are class means"
      note={andrews.state.status === 'ready' ? andrews.state.data.caveats.join(' ') : undefined}
      actions={
        <Select
          value={classBy}
          options={ANDREWS_CLASSES}
          onChange={setClassBy}
          label={(value) => `Class: ${GROUPING_LABELS[value]}`}
          ariaLabel="Class Andrews curves by"
        />
      }
    >
      <Async state={andrews.state} retry={andrews.retry} height="h-[28rem]">
        {(data) => {
          const colors = groupColors(
            data.class_by,
            data.classes.map((entry) => entry.label),
          )
          // One trace per class for all its curves, joined by null breaks:
          // sixty traces per class would swamp the legend and the redraw.
          const joined = (curves: (number | null)[][]) => ({
            x: curves.flatMap(() => [...data.t, null]),
            y: curves.flatMap((curve) => [...curve, null]),
          })

          return (
            <div className="space-y-2">
              <Plot
                ariaLabel={`Andrews curves classed by ${GROUPING_LABELS[data.class_by]}`}
                className="h-[28rem] w-full"
                data={data.classes.flatMap((entry) => {
                  const color = colors[entry.label] ?? INK.muted
                  const plain = entry.curves.filter((_, index) => !entry.is_anomaly[index])
                  const flagged = entry.curves.filter((_, index) => entry.is_anomaly[index])
                  return [
                    {
                      type: 'scatter' as const,
                      mode: 'lines' as const,
                      name: entry.label,
                      legendgroup: entry.label,
                      showlegend: false,
                      ...joined(plain),
                      line: { color, width: 0.8 },
                      opacity: 0.25,
                      hoverinfo: 'skip' as const,
                    },
                    {
                      type: 'scatter' as const,
                      mode: 'lines' as const,
                      name: `${entry.label}, flagged`,
                      legendgroup: entry.label,
                      showlegend: false,
                      ...joined(flagged),
                      line: { color: FLAGGED.color, width: 1, dash: 'dot' as const },
                      opacity: 0.7,
                      hoverinfo: 'skip' as const,
                    },
                    {
                      type: 'scatter' as const,
                      mode: 'lines' as const,
                      name: `${entry.label} (n = ${entry.n.toLocaleString()})`,
                      legendgroup: entry.label,
                      x: data.t,
                      y: entry.mean_curve,
                      line: { color, width: 3 },
                      hovertemplate: `t = %{x:.2f}<br>f(t) = %{y:.2f}<extra>${entry.label} mean</extra>`,
                    },
                  ]
                })}
                layout={baseLayout({
                  margin: { l: 48, r: 12, t: 8, b: 72 },
                  xaxis: {
                    title: { text: 't', font: { color: INK.muted } },
                    gridcolor: INK.grid,
                    tickfont: { color: INK.muted },
                    tickvals: [-Math.PI, -Math.PI / 2, 0, Math.PI / 2, Math.PI],
                    ticktext: ['−π', '−π/2', '0', 'π/2', 'π'],
                  },
                  yaxis: {
                    title: { text: 'f(t), standardised', font: { color: INK.muted } },
                    gridcolor: INK.grid,
                    tickfont: { color: INK.muted },
                  },
                  legend: { orientation: 'h', y: -0.22, x: 0, font: { color: INK.secondary } },
                })}
              />
              <p className="text-xs tabular-nums text-slate-400">
                Column order {data.columns.map(columnLabel).join(' → ')} ·{' '}
                {data.curves_returned.toLocaleString()} curves of{' '}
                {data.rows_used.toLocaleString()} rows; means use every row · dotted{' '}
                <span style={{ color: FLAGGED.color }}>orange</span> curves are flagged anomalies
              </p>
            </div>
          )
        }}
      </Async>
    </Card>
  )
}
