/**
 * API client — task 10.1.
 *
 * design §12: this module is the only place `fetch` appears. Pages consume
 * typed helpers, so a backend contract change surfaces as a compile error
 * instead of a blank chart.
 *
 * The types mirror the Pydantic response models exactly. Where the backend
 * returns a permissive `dict[str, Any]` (leakage audits, split summaries), the
 * type here is deliberately loose too rather than pretending to a precision the
 * contract does not have.
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

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
}

function query(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) search.set(key, String(value))
  }
  const rendered = search.toString()
  return rendered ? `?${rendered}` : ''
}

// --- Health -----------------------------------------------------------------

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

// --- Districts and observations (Phase 1, 2, 10) ----------------------------

export interface DistrictProperties {
  district_id: string
  name: string
  city: string
  state: string
  country: string
  centroid_lat: number
  centroid_lon: number
}

export interface DistrictFeature {
  type: 'Feature'
  id: string
  properties: DistrictProperties
  geometry: { type: 'Polygon'; coordinates: [number, number][][] }
}

export interface DistrictsResponse {
  type: 'FeatureCollection'
  features: DistrictFeature[]
  metadata: { accuracy: string; source: string; [key: string]: unknown }
}

export function getDistricts(): Promise<DistrictsResponse> {
  return request<DistrictsResponse>('/data/districts')
}

export interface StationReading {
  station: string
  lat: number
  lon: number
  district_id: string | null
  district_name: string | null
  timestamp: string
  pm25: number | null
  pm10: number | null
  temp: number | null
  humidity: number | null
  traffic_score: number | null
  is_anomaly: boolean
}

export interface LatestObservationsResponse {
  readings: StationReading[]
  observed_at: string | null
  stale_minutes: number | null
  generated_at: string
}

export function getLatestObservations(): Promise<LatestObservationsResponse> {
  return request<LatestObservationsResponse>('/data/observations/latest')
}

export interface SeriesPoint {
  timestamp: string
  pm25: number | null
  pm10: number | null
  temp: number | null
  humidity: number | null
  traffic_score: number | null
  is_anomaly: boolean
}

export interface StationSeriesResponse {
  station: string
  lat: number
  lon: number
  hours: number
  points: SeriesPoint[]
  generated_at: string
}

export function getStationSeries(
  lat: number,
  lon: number,
  hours = 48,
): Promise<StationSeriesResponse> {
  return request<StationSeriesResponse>(
    `/data/observations/series${query({ lat, lon, hours })}`,
  )
}

// --- Sources and ingestion log (Phase 2; Admin view 10.15) ------------------

export interface RunSummary {
  run_id: number
  mode: string
  status: string
  started_at: string
  finished_at: string | null
  records_fetched: number
  records_valid: number
  records_quarantined: number
  records_written: number
  message: string | null
}

export interface SourceHealth {
  id: number
  name: string
  api_url: string | null
  status: string
  last_run: string | null
  domain: string | null
  requires_credentials: boolean
  is_synthetic: boolean
  credentials_configured: boolean
  observation_count: number
  last_observation_at: string | null
  latest_run: RunSummary | null
}

export interface SourcesResponse {
  sources: SourceHealth[]
  ingestion_mode: IngestionMode
  analytics_source_scope: 'demo' | 'live' | 'all'
  generated_at: string
}

export function getSources(): Promise<SourcesResponse> {
  return request<SourcesResponse>('/data/sources')
}

export interface QuarantineEntry {
  id: number
  run_id: number
  reason: string
  created_at: string
}

export interface IngestionRunsResponse {
  runs: RunSummary[]
  quarantined_sample: QuarantineEntry[]
}

export function getIngestionRuns(limit = 20): Promise<IngestionRunsResponse> {
  return request<IngestionRunsResponse>(`/data/ingestion/runs${query({ limit })}`)
}

export interface QualityRunResponse {
  rows_in: number
  rows_out: number
  rows_preserved: boolean
  feature_set_is_complete: boolean
  flags_written: number
  anomalies: { flagged: number; flagged_pct: number; [key: string]: unknown }
  imputation: Record<string, unknown>
  missingness: Record<string, unknown>
  outputs: Record<string, unknown>
  window: Record<string, unknown>
  generated_at: string
}

/** The Analyst's thresholds (specs §4). Detector votes and gap length are the
 *  two knobs the quality engine actually exposes. */
export function runQuality(params: {
  max_gap_hours?: number
  min_votes?: number
  backward_fill?: boolean
  persist?: boolean
}): Promise<QualityRunResponse> {
  return post<QualityRunResponse>(`/data/quality${query(params)}`)
}

// --- EDA (Phase 4) ----------------------------------------------------------

export interface UnivariateStats {
  column: string
  count: number
  missing: number
  missing_pct: number
  mean: number | null
  median: number | null
  std: number | null
  variance: number | null
  min: number | null
  max: number | null
  q1: number | null
  q3: number | null
  iqr: number | null
  p05: number | null
  p95: number | null
  skewness: number | null
  kurtosis: number | null
}

export interface CorrelationPair {
  a: string
  b: string
  pearson: number | null
  spearman: number | null
  sample_size: number
  divergence: number | null
}

export interface BivariateProfile {
  columns: string[]
  pearson: Record<string, Record<string, number | null>>
  spearman: Record<string, Record<string, number | null>>
  pairs: CorrelationPair[]
  caveat: string
}

export interface DistributionAssessment {
  column: string
  skewness: number | null
  kurtosis: number | null
  shape: string
  is_strictly_positive: boolean
  log_skewness: number | null
  recommend_log_transform: boolean
  rationale: string
  normality_statistic: number | null
  normality_p_value: number | null
}

export interface ProfileResponse {
  generated_at: string
  cached: boolean
  dataset_version: Record<string, unknown>
  window: Record<string, unknown>
  rows: number
  univariate: UnivariateStats[]
  bivariate: BivariateProfile
  distributions: DistributionAssessment[]
  caveats: string[]
}

export function getProfile(): Promise<ProfileResponse> {
  return post<ProfileResponse>('/eda/profile')
}

export interface DecompositionResponse {
  column: string
  station: string
  period: number
  timestamps: string[]
  observed: number[]
  trend: number[]
  seasonal: number[]
  residual: number[]
  strength: { trend: number; seasonal: number }
  points_returned: number
  points_analysed: number
  interpolated_points: number
  caveat: string
  cached: boolean
  dataset_version: Record<string, unknown>
}

export function getDecomposition(
  body: { column?: string; station?: string; period?: number; max_points?: number } = {},
): Promise<DecompositionResponse> {
  return post<DecompositionResponse>('/eda/decompose', body)
}

// --- ESI and projections (Phase 6) ------------------------------------------

export interface ComponentLoadings {
  index: number
  explained_variance_ratio: number
  cumulative_variance_ratio: number
  loadings: Record<string, number>
  drivers: string[]
}

export interface ReductionResponse {
  columns: string[]
  rows_used: number
  rows_dropped: number
  components: ComponentLoadings[]
  esi: {
    mean: number | null
    median: number | null
    min: number | null
    max: number | null
    latest: number | null
  }
  pc1_oriented_by: string
  pc1_sign_flipped: boolean
  caveats: string[]
  cached: boolean
  dataset_version: Record<string, unknown>
  artifact_path: string | null
}

export function getReduction(persist = false): Promise<ReductionResponse> {
  return post<ReductionResponse>('/eda/reduce', { persist })
}

export interface ProjectionResponse {
  columns: string[]
  x: number[]
  y: number[]
  timestamps: string[]
  stations: string[]
  hour_of_day: number[]
  perplexity: number
  points: number
  rows_available: number
  subsampled: boolean
  caveat: string
  cached: boolean
  dataset_version: Record<string, unknown>
}

export function getProjection(maxPoints = 1200): Promise<ProjectionResponse> {
  return post<ProjectionResponse>('/eda/tsne', { max_points: maxPoints })
}

// --- Models and predictions (Phases 7-9) ------------------------------------

export interface RegisteredModel {
  id: number
  name: string
  target: string
  features_used: string[] | Record<string, unknown>
  mae: number | null
  rmse: number | null
  r2: number | null
  created_at: string
  artifact_path: string
}

export function getModels(limit = 100): Promise<RegisteredModel[]> {
  return request<RegisteredModel[]>(`/ml/models${query({ limit })}`)
}

export interface FeatureImportance {
  feature: string
  mean_abs_shap: number
  mean_shap: number
  share: number
  direction: string
}

export interface ImportanceResponse {
  model_id: number
  name: string
  target: string
  horizon_hours: number
  trained_at: string | null
  method: string
  base_value: number
  rows: number
  features: FeatureImportance[]
  cached: boolean
  caveat: string
}

export function getImportance(
  modelId: number,
  sampleRows = 300,
): Promise<ImportanceResponse> {
  return request<ImportanceResponse>(
    `/ml/models/${modelId}/importance${query({ sample_rows: sampleRows })}`,
  )
}

/** The specs §8 contract, plus the provenance the backend adds around it. */
export interface PredictionResponse {
  prediction: number
  unit: string
  confidence_interval: [number, number]
  top_features: Record<string, number>
  target_time: string
  origin_time: string
  horizon_hours: number
  lat: number
  lon: number
  coverage: number
  interval_method: string
  base_value: number
  model: { id: number; name: string; target: string; trained_at: string | null; mae: number | null }
  prediction_id: number | null
  caveats: string[]
}

export interface PredictionRequest {
  lat: number
  lon: number
  horizon?: number
  at?: string
  coverage?: number
  top_features?: number
  model_name?: string
  persist?: boolean
}

export function predict(body: PredictionRequest): Promise<PredictionResponse> {
  return post<PredictionResponse>('/ml/predict', body)
}
