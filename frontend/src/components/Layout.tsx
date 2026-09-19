/**
 * App shell — tasks 10.2 and 10.16.
 *
 * The disclaimer lives in the shell rather than on each page, because ETH-1 and
 * AC-11 call for it to be *persistent*: a notice that can be navigated away
 * from is a notice that will be. It sits in the footer of every route, and the
 * same two sentences appear in the exported EDA report.
 */

import { NavLink, Outlet } from 'react-router-dom'

const ROUTES = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/eda', label: 'EDA Studio' },
  { to: '/models', label: 'Model Lab' },
  { to: '/admin', label: 'Admin' },
]

export const DISCLAIMER = {
  bias:
    'Predictions rely on sensor placement, which may exhibit geographic and ' +
    'socioeconomic bias: a district with no monitor is not a clean district.',
  causation:
    'Correlation shown does not equal causation. Causal inference is out of ' +
    'scope, and no causal claim is made anywhere in this platform.',
}

export default function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-8 gap-y-3 px-4 py-3 sm:px-6">
          <div className="flex items-baseline gap-2">
            <span className="text-lg font-semibold tracking-tight text-slate-100">
              Eco-City Pulse
            </span>
            <span className="hidden text-xs text-slate-500 sm:inline">
              urban environmental intelligence
            </span>
          </div>

          <nav className="flex gap-1" aria-label="Primary">
            {ROUTES.map((route) => (
              <NavLink
                key={route.to}
                to={route.to}
                end={route.end}
                className={({ isActive }) =>
                  `rounded px-3 py-1.5 text-sm transition ${
                    isActive
                      ? 'bg-slate-800 text-slate-100'
                      : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                  }`
                }
              >
                {route.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
        <Outlet />
      </main>

      <footer className="mt-8 border-t border-slate-800 bg-slate-900/60">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6">
          <p className="text-xs font-medium uppercase tracking-wide text-amber-500/80">
            Read this before acting on anything above
          </p>
          <p className="mt-1 max-w-4xl text-xs leading-relaxed text-slate-400">
            {DISCLAIMER.bias} <span className="text-slate-300">{DISCLAIMER.causation}</span>
          </p>
          <p className="mt-2 text-xs text-slate-600">
            Demo data is synthetic and labelled as such. Only public
            environmental data is used; no personal data is collected.
          </p>
        </div>
      </footer>
    </div>
  )
}
