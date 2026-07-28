import type { BenchmarkJobStatusResponse } from '@apps/ensemble/services/api'

export const BENCHMARK_RESULTS_TABLE_STORAGE_KEY = 'pe-hub.benchmark.results-table'

export const BENCHMARK_TABLE_METRICS = ['pearson', 'spearman', 'mse', 'mae'] as const

export type BenchmarkTableMetric = (typeof BENCHMARK_TABLE_METRICS)[number]

export interface BenchmarkResultsTableState {
  version: 1
  ingestedJobIds: string[]
  models: string[]
  rows: string[]
  /** datasheet row → model → ordered metric snapshots from each run */
  cells: Record<string, Record<string, Record<string, number>[]>>
}

export function emptyBenchmarkResultsTable(): BenchmarkResultsTableState {
  return {
    version: 1,
    ingestedJobIds: [],
    models: [],
    rows: [],
    cells: {},
  }
}

function isMetricSnapshot(value: unknown): value is Record<string, number> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  return Object.values(value).every((entry) => typeof entry === 'number' && Number.isFinite(entry))
}

export function loadBenchmarkResultsTable(): BenchmarkResultsTableState {
  try {
    const raw = localStorage.getItem(BENCHMARK_RESULTS_TABLE_STORAGE_KEY)
    if (!raw) return emptyBenchmarkResultsTable()
    const parsed = JSON.parse(raw) as Partial<BenchmarkResultsTableState>
    if (parsed.version !== 1) return emptyBenchmarkResultsTable()
    if (!Array.isArray(parsed.models) || !Array.isArray(parsed.rows)) {
      return emptyBenchmarkResultsTable()
    }
    if (!parsed.cells || typeof parsed.cells !== 'object') {
      return emptyBenchmarkResultsTable()
    }
    return {
      version: 1,
      ingestedJobIds: Array.isArray(parsed.ingestedJobIds)
        ? parsed.ingestedJobIds.filter((id): id is string => typeof id === 'string')
        : [],
      models: parsed.models.filter((model): model is string => typeof model === 'string'),
      rows: parsed.rows.filter((row): row is string => typeof row === 'string'),
      cells: parsed.cells as BenchmarkResultsTableState['cells'],
    }
  } catch {
    return emptyBenchmarkResultsTable()
  }
}

export function saveBenchmarkResultsTable(state: BenchmarkResultsTableState): void {
  try {
    localStorage.setItem(BENCHMARK_RESULTS_TABLE_STORAGE_KEY, JSON.stringify(state))
  } catch {
    // ignore quota / private-mode failures
  }
}

function sanitizeMetrics(metrics: Record<string, number>): Record<string, number> {
  const cleaned: Record<string, number> = {}
  for (const [key, value] of Object.entries(metrics)) {
    if (key === 'n_samples') continue
    if (typeof value === 'number' && Number.isFinite(value)) {
      cleaned[key] = Number(value.toFixed(6))
    }
  }
  return cleaned
}

export function formatMetricValue(value: number | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return value.toFixed(4)
}

export function formatCellRuns(
  runs: Record<string, number>[] | undefined,
  metric: BenchmarkTableMetric
): string {
  if (!runs || runs.length === 0) return ''
  return runs.map((run) => formatMetricValue(run[metric])).join(';')
}

export function mergeBenchmarkJobIntoTable(
  state: BenchmarkResultsTableState,
  job: BenchmarkJobStatusResponse
): BenchmarkResultsTableState {
  if (job.status !== 'succeeded' || !job.result?.metrics) {
    return state
  }
  if (state.ingestedJobIds.includes(job.job_id)) {
    return state
  }

  const metrics = sanitizeMetrics(job.result.metrics)
  if (!isMetricSnapshot(metrics) || Object.keys(metrics).length === 0) {
    return state
  }

  const model = (job.model_name || job.result.model || 'unknown').trim() || 'unknown'
  const row =
    (job.benchmark_name || job.result.benchmark_name || 'unknown').trim() || 'unknown'

  const models = state.models.includes(model) ? state.models : [...state.models, model]
  const rows = state.rows.includes(row) ? state.rows : [...state.rows, row]
  const rowCells = { ...(state.cells[row] ?? {}) }
  const existing = rowCells[model] ?? []
  rowCells[model] = [...existing, metrics]

  return {
    version: 1,
    ingestedJobIds: [...state.ingestedJobIds, job.job_id],
    models,
    rows,
    cells: {
      ...state.cells,
      [row]: rowCells,
    },
  }
}

export function mergeBenchmarkJobsIntoTable(
  state: BenchmarkResultsTableState,
  jobs: BenchmarkJobStatusResponse[]
): BenchmarkResultsTableState {
  return jobs.reduce((next, job) => mergeBenchmarkJobIntoTable(next, job), state)
}
