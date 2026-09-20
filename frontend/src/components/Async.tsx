/**
 * Loading, empty and error states — task 10.17.
 *
 * Every data-backed view goes through `useAsync` + `<Async>`, so the three
 * states are handled once rather than forgotten per panel. A blank card is the
 * most common way a dashboard lies: it looks like "no pollution" when it means
 * "the request failed".
 */

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { ApiError } from '../services/api'

export type AsyncState<T> =
  | { status: 'loading' }
  | { status: 'ready'; data: T }
  | { status: 'error'; message: string; code: string }

export interface AsyncOptions {
  /**
   * Re-fetch every N milliseconds. Omitted, the request runs once on mount —
   * which is right for anything that only changes when someone changes it.
   */
  refreshMs?: number
}

export function useAsync<T>(
  load: () => Promise<T>,
  deps: unknown[] = [],
  { refreshMs }: AsyncOptions = {},
) {
  const [state, setState] = useState<AsyncState<T>>({ status: 'loading' })
  const [nonce, setNonce] = useState(0)
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null)

  const retry = useCallback(() => setNonce((value) => value + 1), [])

  // `load` is a new closure every render, so the poll must read it through a
  // ref. In the effect's dependency list it would clear and restart the
  // interval on each render, and a timer that resets faster than it fires
  // never fires at all.
  const loadRef = useRef(load)
  loadRef.current = load

  useEffect(() => {
    let active = true
    setState({ status: 'loading' })

    load()
      .then((data) => {
        if (active) setState({ status: 'ready', data })
      })
      .catch((error: unknown) => {
        if (!active) return
        setState(
          error instanceof ApiError
            ? { status: 'error', message: error.message, code: error.code }
            : {
                status: 'error',
                message: 'The backend is unreachable.',
                code: 'network_error',
              },
        )
      })

    // Guards against a state update after unmount in StrictMode's double-run.
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  // The poll deliberately does *not* go back through `status: 'loading'`. That
  // would blank the panel to a spinner on every tick, so the more often a
  // dashboard refreshed the less of it you could read. A refresh either
  // replaces the data or leaves what is on screen alone.
  useEffect(() => {
    if (!refreshMs) return

    let active = true
    const id = window.setInterval(() => {
      loadRef
        .current()
        .then((data) => {
          if (!active) return
          setState({ status: 'ready', data })
          setRefreshedAt(new Date())
        })
        .catch(() => {
          // Swallowed on purpose: a failed *refresh* still has the last good
          // reading behind it, and replacing a real number with an error
          // because one poll timed out is the worse of the two lies. The
          // initial load reports its failure normally, and `observed_at` in
          // the payload is what tells a reader the data has stopped moving.
        })
    }, refreshMs)

    return () => {
      active = false
      window.clearInterval(id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshMs, ...deps, nonce])

  return { state, retry, refreshedAt }
}

interface AsyncProps<T> {
  state: AsyncState<T>
  retry?: () => void
  /** Called with the data; return null to show the empty state instead. */
  children: (data: T) => ReactNode
  empty?: (data: T) => boolean
  emptyMessage?: string
  height?: string
}

export function Async<T>({
  state,
  retry,
  children,
  empty,
  emptyMessage = 'No data yet.',
  height = 'h-72',
}: AsyncProps<T>) {
  if (state.status === 'loading') {
    return (
      <div
        className={`flex ${height} animate-pulse items-center justify-center rounded-lg bg-slate-800/40 text-sm text-slate-500`}
      >
        Loading…
      </div>
    )
  }

  if (state.status === 'error') {
    return (
      <div
        className={`flex ${height} flex-col items-center justify-center gap-2 rounded-lg border border-red-900/50 bg-red-950/30 p-4 text-center`}
      >
        <p className="text-sm font-medium text-red-300">{state.message}</p>
        <p className="font-mono text-xs text-red-400/70">{state.code}</p>
        {retry && (
          <button
            type="button"
            onClick={retry}
            className="mt-1 rounded border border-red-800 px-3 py-1 text-xs text-red-200 hover:bg-red-900/40"
          >
            Retry
          </button>
        )}
      </div>
    )
  }

  if (empty?.(state.data)) {
    return (
      <div
        className={`flex ${height} items-center justify-center rounded-lg border border-dashed border-slate-700 text-sm text-slate-500`}
      >
        {emptyMessage}
      </div>
    )
  }

  return <>{children(state.data)}</>
}
