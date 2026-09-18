/**
 * API client.
 *
 * design §12: this module is the only place `fetch` appears. Pages consume
 * typed helpers, so a backend contract change surfaces as a compile error
 * instead of a blank chart. Phase 0 covers the health probe only; the data,
 * EDA, and prediction calls are added in task 10.1.
 */

const API_BASE = '/api/v1'

export type IngestionMode = 'scheduled' | 'manual' | 'upload' | 'demo'

export interface HealthResponse {
  status: 'ok'
  app: string
  version: string
  environment: string
  ingestion_mode: IngestionMode
  live_credentials_configured: boolean
}

export interface ApiErrorBody {
  error: {
    code: string
    message: string
    details: Record<string, unknown>
  }
}

/** Mirrors the backend error envelope from `core/exceptions.py`. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details: Record<string, unknown> = {},
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: 'application/json', ...init?.headers },
    ...init,
  })

  if (!response.ok) {
    // A proxy or crash can return non-JSON, so parsing must not throw here.
    const body = (await response.json().catch(() => null)) as ApiErrorBody | null
    throw new ApiError(
      response.status,
      body?.error.code ?? 'unexpected_error',
      body?.error.message ?? `Request to ${path} failed (${response.status}).`,
      body?.error.details ?? {},
    )
  }

  return (await response.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}
