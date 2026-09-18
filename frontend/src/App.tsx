/**
 * Phase 0 app shell.
 *
 * Its only job is to prove the exit criterion: the browser reaches
 * /api/v1/health through the Nginx proxy (AC-1). Routing and the three real
 * screens land in Phase 10.
 */

import { useEffect, useState } from 'react'
import { ApiError, getHealth, type HealthResponse } from './services/api'

type Probe =
  | { state: 'loading' }
  | { state: 'ok'; health: HealthResponse }
  | { state: 'error'; message: string }

export default function App() {
  const [probe, setProbe] = useState<Probe>({ state: 'loading' })

  useEffect(() => {
    let active = true

    getHealth()
      .then((health) => {
        if (active) setProbe({ state: 'ok', health })
      })
      .catch((error: unknown) => {
        if (!active) return
        setProbe({
          state: 'error',
          message:
            error instanceof ApiError
              ? `${error.code}: ${error.message}`
              : 'Backend unreachable.',
        })
      })

    // Guards against a state update after unmount in StrictMode's double-run.
    return () => {
      active = false
    }
  }, [])

  return (
    <main className="mx-auto flex min-h-full max-w-3xl flex-col justify-center gap-8 px-6 py-16">
      <header className="space-y-2">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-emerald-400">
          Phase 0 &middot; Foundation
        </p>
        <h1 className="text-4xl font-semibold tracking-tight">Eco-City Pulse</h1>
        <p className="text-slate-400">
          Urban environmental intelligence &mdash; exploratory data analysis and
          interpretable PM2.5 forecasting.
        </p>
      </header>

      <section
        aria-live="polite"
        className="rounded-xl border border-slate-800 bg-slate-900/60 p-6"
      >
        <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-slate-400">
          Backend connectivity
        </h2>

        {probe.state === 'loading' && (
          <p className="text-slate-400">Contacting /api/v1/health&hellip;</p>
        )}

        {probe.state === 'error' && (
          <p className="font-mono text-sm text-rose-400">{probe.message}</p>
        )}

        {probe.state === 'ok' && (
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 font-mono text-sm">
            <Row label="status" value={probe.health.status} highlight />
            <Row label="version" value={probe.health.version} />
            <Row label="environment" value={probe.health.environment} />
            <Row label="ingestion mode" value={probe.health.ingestion_mode} />
            <Row
              label="live credentials"
              value={probe.health.live_credentials_configured ? 'configured' : 'none (offline)'}
            />
          </dl>
        )}
      </section>
    </main>
  )
}

function Row({
  label,
  value,
  highlight = false,
}: {
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <>
      <dt className="text-slate-500">{label}</dt>
      <dd className={highlight ? 'text-emerald-400' : 'text-slate-200'}>{value}</dd>
    </>
  )
}
