/**
 * The chart design system.
 *
 * These are the same values `backend/services/eda/report.py` draws with, so the
 * exported HTML report and the screens read as one product rather than two
 * things that happen to plot the same data.
 *
 * Colour is assigned by the *job* it does, never by series index:
 *
 * - **categorical** — identity. Fixed order, never cycled. Validated for the
 *   dark surface: worst adjacent CVD ΔE 9.4, normal-vision ΔE 26.5.
 * - **sequential** — magnitude. One hue, light→dark. Never a rainbow.
 * - **diverging** — polarity. Two hues with a *neutral grey* midpoint, because
 *   correlation has a sign and a one-hue ramp would hide the difference
 *   between −0.8 and +0.8.
 * - **status** — air-quality bands. Reserved; never reused for "series 4", and
 *   always paired with a label so colour never carries meaning alone.
 */

import type { Layout } from 'plotly.js'

/** Chart surface: the card the plot sits on, not the page background. */
export const SURFACE = '#0f172a'

export const INK = {
  primary: '#e2e8f0',
  secondary: '#94a3b8',
  muted: '#64748b',
  grid: '#1e293b',
}

/** Identity. Three slots, which is all any chart here needs. */
export const CATEGORICAL = ['#3987e5', '#d95926', '#199e70'] as const

/** Magnitude, light→dark. */
export const SEQUENTIAL = [
  '#cde2fb',
  '#9ec5f4',
  '#6da7ec',
  '#3987e5',
  '#256abf',
  '#184f95',
  '#0d366b',
] as const

/**
 * Polarity. Blue ↔ red with a neutral grey midpoint — the midpoint must read as
 * "nothing", which is why blue↔aqua was rejected: both are cool.
 */
export const DIVERGING: [number, string][] = [
  [0, '#0d366b'],
  [0.15, '#256abf'],
  [0.3, '#3987e5'],
  [0.42, '#86b6ef'],
  [0.5, '#383835'],
  [0.58, '#f0a9a9'],
  [0.7, '#e34948'],
  [0.85, '#c62f2e'],
  [1, '#8f1f1e'],
]

/**
 * Air-quality bands (CPCB-style breakpoints for PM2.5, µg/m³).
 *
 * A status palette, so it ships with a label everywhere it appears. The bands
 * are a descriptive convenience, not a health judgement — the platform makes no
 * claim about what any concentration does to a person.
 */
export interface Band {
  limit: number
  label: string
  color: string
}

export const AQI_BANDS: Band[] = [
  { limit: 30, label: 'Good', color: '#0ca30c' },
  { limit: 60, label: 'Satisfactory', color: '#8bc34a' },
  { limit: 90, label: 'Moderate', color: '#fab219' },
  { limit: 120, label: 'Poor', color: '#ec835a' },
  { limit: 250, label: 'Very poor', color: '#d03b3b' },
  { limit: Infinity, label: 'Severe', color: '#9333ea' },
]

const SEVERE: Band = { limit: Infinity, label: 'Severe', color: '#9333ea' }
const NO_READING: Band = { limit: 0, label: 'No reading', color: INK.muted }

export function band(pm25: number | null | undefined): Band {
  if (pm25 === null || pm25 === undefined || Number.isNaN(pm25)) return NO_READING
  return AQI_BANDS.find((entry) => pm25 <= entry.limit) ?? SEVERE
}

/** Human labels and units, matching the backend report's. */
export const COLUMN_LABELS: Record<string, string> = {
  pm25: 'PM2.5',
  pm10: 'PM10',
  temp: 'Temperature',
  humidity: 'Humidity',
  traffic_score: 'Traffic',
}

export const COLUMN_UNITS: Record<string, string> = {
  pm25: 'µg/m³',
  pm10: 'µg/m³',
  temp: '°C',
  humidity: '%',
  traffic_score: 'index',
}

export function label(column: string): string {
  return COLUMN_LABELS[column] ?? column
}

/**
 * Plotly layout defaults.
 *
 * Recessive grid, no chart junk, and a unified hover so a crosshair reads every
 * series at once rather than whichever line the cursor happens to be nearest.
 */
export function baseLayout(overrides: Partial<Layout> = {}): Partial<Layout> {
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: INK.secondary, family: 'Inter, ui-sans-serif, system-ui', size: 12 },
    margin: { l: 56, r: 16, t: 28, b: 44 },
    xaxis: {
      gridcolor: INK.grid,
      zerolinecolor: INK.grid,
      linecolor: INK.grid,
      tickfont: { color: INK.muted },
autorange: true,
    },
    yaxis: {
      gridcolor: INK.grid,
      zerolinecolor: INK.grid,
      linecolor: INK.grid,
      tickfont: { color: INK.muted },
    },
    hoverlabel: {
      bgcolor: '#1e293b',
      bordercolor: INK.grid,
      font: { color: INK.primary, family: 'Inter, ui-sans-serif, system-ui' },
    },
    legend: {
      orientation: 'h' as const,
      y: -0.2,
      x: 0,
      font: { color: INK.secondary },
      bgcolor: 'rgba(0,0,0,0)',
    },
    ...overrides,
  }
}

/** Plotly config: no modebar clutter, responsive, no branding. */
export const PLOT_CONFIG = {
  displayModeBar: false,
  responsive: true,
  displaylogo: false,
} as const
