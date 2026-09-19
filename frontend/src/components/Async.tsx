/**
 * Loading, empty and error states — task 10.17.
 *
 * Every data-backed view goes through `useAsync` + `<Async>`, so the three
 * states are handled once rather than forgotten per panel. A blank card is the
 * most common way a dashboard lies: it looks like "no pollution" when it means
 * "the request failed".
 */

import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { ApiError } from '../services/api'

export type AsyncState<T> =
  | { status: 'loading' }
  | { status: 'ready'; data: T }
  | { status: 'error'; message: string; code: string }

export function useAsync<T>(load: () => Promise<T>, deps: unknown[] = []) {
  const [state, setState] = useState<AsyncState<T>>({ status: 'loading' })
  const [nonce, setNonce] = useState(0)

  const retry = useCallback(() => setNonce((value) => value + 1), [])

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

  return { state, retry }
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
