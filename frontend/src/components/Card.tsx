/**
 * The panel every chart and table sits in.
 *
 * `note` is not decoration: most panels here carry a caveat the backend sent
 * with the data (association is not causation, t-SNE axes mean nothing, an ESI
 * is relative to its window). Giving the caveat a fixed place in the component
 * means a panel cannot quietly ship without one.
 */

import type { ReactNode } from 'react'

interface CardProps {
  title: string
  subtitle?: string
  note?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
}

export default function Card({
  title,
  subtitle,
  note,
  actions,
  children,
  className = '',
}: CardProps) {
  return (
    <section
      className={`rounded-xl border border-slate-800 bg-slate-900/50 p-4 ${className}`}
    >
      <header className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">{title}</h2>
          {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
        </div>
        {actions}
      </header>

      {children}

      {note && (
        <p className="mt-3 border-t border-slate-800 pt-2 text-xs leading-relaxed text-slate-500">
          {note}
        </p>
      )}
    </section>
  )
}
