/**
 * Admin — task 10.15.
 *
 * Source health and the ingestion log, for the persona who keeps the data
 * flowing (specs §4, Sam). The important thing this screen gets right is that
 * **`offline` is not an error**: a source with no API key is configured and
 * deliberately not contacted (DR-1), and showing it in red would train the
 * operator to ignore red.
 */

import { useState } from 'react'
import { Async, useAsync } from '../components/Async'
import Card from '../components/Card'
import {
  ApiError,
  getIngestionRuns,
  getSources,
  runQuality,
  type QualityRunResponse,
} from '../services/api'

const STATUS_STYLE: Record<string, string> = {
  healthy: 'bg-emerald-950/60 text-emerald-300 border-emerald-900',
  degraded: 'bg-amber-950/60 text-amber-300 border-amber-900',
  offline: 'bg-slate-800 text-slate-400 border-slate-700',
}

export default function Admin() {
  const sources = useAsync(getSources, [])
  const runs = useAsync(() => getIngestionRuns(15), [])

  return (
    <div className="space-y-6">
      <Async state={sources.state} retry={sources.retry} height="h-64">
        {(data) => (
          <Card
            title="Source health"
            subtitle={`Ingestion mode: ${data.ingestion_mode} · analytics reading: ${data.analytics_source_scope}`}
            note="A keyless live source reports as offline. That is the expected demo state, not a failure: the platform runs end to end with every live API disabled (DR-1, AC-2). The analytics scope decides which provenance every EDA, training and forecast request reads when it names no sources — so a source can be healthy and ingesting while the charts deliberately read the other one."
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    {['Source', 'Status', 'Key', 'Observations', 'Last reading', 'Last run'].map(
                      (head) => (
                        <th key={head} className="whitespace-nowrap px-3 py-2 font-medium">
                          {head}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {data.sources.map((source) => (
                    <tr key={source.id} className="border-t border-slate-800">
                      <td className="px-3 py-2">
                        <span className="text-slate-200">{source.name}</span>
                        {source.domain && (
                          <span className="ml-2 text-xs text-slate-500">{source.domain}</span>
                        )}
                        {/* ETH-1: a reader has to be able to tell a generated
                            series from a measured one without knowing which
                            source name means which. */}
                        {source.is_synthetic && (
                          <span
                            className="ml-2 rounded border border-amber-500/40 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-amber-300/90"
                            title="Values are generated, not measured."
                          >
                            synthetic
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2">
                        <span
                          className={`inline-block rounded border px-2 py-0.5 text-xs ${
                            STATUS_STYLE[source.status] ?? STATUS_STYLE.offline
                          }`}
                        >
                          {source.status}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-400">
                        {!source.requires_credentials
                          ? 'not needed'
                          : source.credentials_configured
                            ? 'configured'
                            : 'absent'}
                      </td>
                      <td className="px-3 py-2 tabular-nums text-slate-300">
                        {source.observation_count.toLocaleString()}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 text-xs text-slate-500">
                        {source.last_observation_at
                          ? new Date(source.last_observation_at).toLocaleString()
                          : '—'}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 text-xs text-slate-500">
                        {source.last_run ? new Date(source.last_run).toLocaleString() : 'never'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </Async>

      <Async state={runs.state} retry={runs.retry} height="h-64">
        {(data) => (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card
              title="Ingestion log"
              subtitle="One row per attempt"
              note="A run that wrote nothing is not necessarily broken — in demo mode a repeat run finds every hour already present and upserts it in place."
            >
              {data.runs.length === 0 ? (
                <p className="text-sm text-slate-500">No runs recorded yet.</p>
              ) : (
                <ul className="space-y-2">
                  {data.runs.map((run) => (
                    <li
                      key={run.run_id}
                      className="flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-800 pb-2 text-sm last:border-0"
                    >
                      <span className="text-slate-300">
                        #{run.run_id}{' '}
                        <span className="text-xs text-slate-500">
                          {new Date(run.started_at).toLocaleString()}
                        </span>
                      </span>
                      <span className="text-xs text-slate-400">
                        {run.records_written.toLocaleString()} written of{' '}
                        {run.records_fetched.toLocaleString()} fetched
                        {run.records_quarantined > 0 && (
                          <span className="ml-2 text-amber-400">
                            {run.records_quarantined} quarantined
                          </span>
                        )}
                      </span>
                      <span className="font-mono text-xs text-slate-500">{run.status}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>

            <Card
              title="Quarantine"
              subtitle="Records that failed their source schema, kept with the reason"
              note="Rejected records are retained rather than dropped: a provider changing its payload shape should surface as a visible pile of evidence, not as silence."
            >
              {data.quarantined_sample.length === 0 ? (
                <p className="text-sm text-slate-500">
                  Nothing quarantined — every record passed its schema.
                </p>
              ) : (
                <ul className="space-y-2 text-sm">
                  {data.quarantined_sample.slice(0, 10).map((record) => (
                    <li key={record.id} className="border-b border-slate-800 pb-2 last:border-0">
                      <p className="text-slate-300">{record.reason}</p>
                      <p className="mt-0.5 font-mono text-xs text-slate-600">
                        run {record.run_id} ·{' '}
                        {new Date(record.created_at).toLocaleString()}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>
        )}
      </Async>

      <Thresholds />
    </div>
  )
}

/* --- Threshold configuration (task 10.15) ---------------------------------- */

function Thresholds() {
  const [minVotes, setMinVotes] = useState(2)
  const [maxGap, setMaxGap] = useState(3)
  const [state, setState] = useState<
    { status: 'idle' | 'running' } | { status: 'done'; result: QualityRunResponse } | { status: 'failed'; message: string }
  >({ status: 'idle' })

  return (
    <Card
      title="Quality thresholds"
      subtitle="Re-run the cleaning pipeline with different settings"
      note="Anomalies are flagged in both directions: a row that no longer meets the threshold is unflagged, so the column reflects the current detectors rather than the union of every run ever made. The row itself is never removed (AC-5)."
    >
      <form
        className="flex flex-wrap items-end gap-4"
        onSubmit={(event) => {
          event.preventDefault()
          setState({ status: 'running' })
          runQuality({ min_votes: minVotes, max_gap_hours: maxGap, persist: true })
            .then((result) => setState({ status: 'done', result }))
            .catch((error: unknown) =>
              setState({
                status: 'failed',
                message:
                  error instanceof ApiError ? error.message : 'The run could not start.',
              }),
            )
        }}
      >
        <label className="text-xs text-slate-400">
          <span className="mb-1 block">
            Detector votes to flag
            <span className="ml-1 text-slate-600">(IQR · Z-score · Isolation Forest)</span>
          </span>
          <select
            className="w-44 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-200"
            value={minVotes}
            onChange={(event) => setMinVotes(Number(event.target.value))}
          >
            <option value={1}>1 — most sensitive</option>
            <option value={2}>2 — default</option>
            <option value={3}>3 — unanimous</option>
          </select>
        </label>

        <label className="text-xs text-slate-400">
          <span className="mb-1 block">Longest gap filled locally</span>
          <select
            className="w-36 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-200"
            value={maxGap}
            onChange={(event) => setMaxGap(Number(event.target.value))}
          >
            {[1, 2, 3, 6, 12].map((hours) => (
              <option key={hours} value={hours}>
                {hours} hour{hours > 1 ? 's' : ''}
              </option>
            ))}
          </select>
        </label>

        <button
          type="submit"
          disabled={state.status === 'running'}
          className="rounded bg-sky-600 px-4 py-1.5 text-sm font-medium text-white transition hover:bg-sky-500 disabled:opacity-50"
        >
          {state.status === 'running' ? 'Running…' : 'Re-run cleaning'}
        </button>
      </form>

      {state.status === 'done' && (
        <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
          {[
            ['Rows in', state.result.rows_in.toLocaleString()],
            ['Rows out', state.result.rows_out.toLocaleString()],
            ['Anomalies flagged', state.result.anomalies.flagged.toLocaleString()],
            ['Flags written', state.result.flags_written.toLocaleString()],
          ].map(([name, value]) => (
            <div key={name}>
              <dt className="text-xs text-slate-500">{name}</dt>
              <dd className="tabular-nums text-slate-200">{value}</dd>
            </div>
          ))}
          <div className="col-span-2 sm:col-span-4">
            <p className="text-xs text-slate-400">
              {state.result.rows_preserved
                ? 'Every row was preserved — the engine flags, it never deletes.'
                : 'Row count changed, which should not happen.'}
            </p>
          </div>
        </dl>
      )}

      {state.status === 'failed' && (
        <p className="mt-3 text-sm text-red-300">{state.message}</p>
      )}
    </Card>
  )
}
