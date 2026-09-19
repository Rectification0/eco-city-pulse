/**
 * A hero number — task 10.3.
 *
 * Deliberately not a chart. A single current value has no shape to plot, and a
 * sparkline beside it would invite reading a trend off six pixels. The band
 * colour is a *status* colour, so it always ships with its label: colour never
 * carries the meaning alone.
 */

interface StatTileProps {
  label: string
  value: number | null
  unit?: string
  /** Band name (Good, Poor…) shown beside the swatch. */
  band?: string
  color?: string
  caption?: string
  digits?: number
}

export default function StatTile({
  label,
  value,
  unit,
  band,
  color,
  caption,
  digits = 1,
}: StatTileProps) {
  const missing = value === null || value === undefined || Number.isNaN(value)

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>

      <p className="mt-2 flex items-baseline gap-1.5">
        <span className="text-4xl font-semibold tabular-nums text-slate-100">
          {missing ? '—' : value.toFixed(digits)}
        </span>
        {unit && !missing && (
          <span className="text-sm text-slate-500">{unit}</span>
        )}
      </p>

      {band && (
        <p className="mt-2 flex items-center gap-2 text-xs">
          <span
            aria-hidden="true"
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span className="text-slate-300">{band}</span>
        </p>
      )}

      {caption && <p className="mt-2 text-xs leading-relaxed text-slate-500">{caption}</p>}
    </div>
  )
}
