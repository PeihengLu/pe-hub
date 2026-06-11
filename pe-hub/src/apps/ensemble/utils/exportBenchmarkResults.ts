import type { BenchmarkJobStatusResponse } from '@apps/ensemble/services/api'

export interface DatasheetInfo {
  study: string
  dataset: string
  cell_line: string
  pe_system: string
}

const CORE_METRIC_ORDER = ['pearson', 'spearman', 'mse', 'mae'] as const

export function parseBenchmarkName(benchmarkName: string): DatasheetInfo {
  const batchMatch = benchmarkName.match(/^([^/]+)\/([^·]+?)\s*·\s*([^·]+?)\s*·\s*(.+)$/)
  if (batchMatch) {
    return {
      study: batchMatch[1].trim(),
      dataset: batchMatch[2].trim(),
      cell_line: batchMatch[3].trim(),
      pe_system: batchMatch[4].trim(),
    }
  }

  const info: DatasheetInfo = {
    study: '',
    dataset: '',
    cell_line: '',
    pe_system: '',
  }
  for (const part of benchmarkName.split(',')) {
    const separatorIndex = part.indexOf('=')
    if (separatorIndex === -1) continue
    const key = part.slice(0, separatorIndex).trim()
    const value = part.slice(separatorIndex + 1).trim().replace(/\|/g, ',')
    if (key === 'study') info.study = value
    else if (key === 'dataset') info.dataset = value
    else if (key === 'cell_line') info.cell_line = value
    else if (key === 'pe_system') info.pe_system = value
  }
  return info
}

function metricSortKey(metric: string): [number, string] {
  const baseIndex = CORE_METRIC_ORDER.indexOf(metric as (typeof CORE_METRIC_ORDER)[number])
  if (baseIndex !== -1) {
    return [baseIndex, metric]
  }
  if (metric.endsWith('_pearson')) return [10, metric]
  if (metric.endsWith('_spearman')) return [11, metric]
  if (metric.endsWith('_mse')) return [12, metric]
  if (metric.endsWith('_mae')) return [13, metric]
  return [100, metric]
}

function sortMetricEntries(metrics: Record<string, number>): [string, number][] {
  return Object.entries(metrics).sort(([left], [right]) => {
    const [leftRank, leftName] = metricSortKey(left)
    const [rightRank, rightName] = metricSortKey(right)
    if (leftRank !== rightRank) return leftRank - rightRank
    return leftName.localeCompare(rightName)
  })
}

function roundMetric(value: number): number {
  return Number(value.toFixed(6))
}

function buildPerformanceMetrics(
  metrics: Record<string, number>
): Record<string, number> {
  const performance: Record<string, number> = {}
  for (const [key, value] of sortMetricEntries(metrics)) {
    if (key === 'n_samples') continue
    if (typeof value === 'number' && Number.isFinite(value)) {
      performance[key] = roundMetric(value)
    }
  }
  return performance
}

function runSortKey(run: BenchmarkRunExport): string {
  return [run.study, run.dataset, run.cell_line, run.pe_system, run.benchmark_name].join('\u0001')
}

export interface BenchmarkRunExport {
  job_id: string
  benchmark_name: string
  study: string
  dataset: string
  cell_line: string
  pe_system: string
  weights: string | null
  n_samples: number | null
  finished_at: string | null
  performance: Record<string, number>
}

export interface BenchmarkModelExport {
  model: string
  runs: BenchmarkRunExport[]
}

export interface BenchmarkResultsExport {
  exported_at: string
  run_count: number
  model_count: number
  models: BenchmarkModelExport[]
}

export function buildBenchmarkResultsExport(
  jobs: BenchmarkJobStatusResponse[]
): BenchmarkResultsExport {
  const succeededJobs = jobs.filter(
    (job) => job.status === 'succeeded' && job.result?.metrics
  )

  const runsByModel = new Map<string, BenchmarkRunExport[]>()

  for (const job of succeededJobs) {
    const datasheet = parseBenchmarkName(job.benchmark_name)
    const metrics = job.result?.metrics ?? {}
    const modelName = job.model_name || job.result?.model || 'unknown'
    const nSamples =
      job.result?.n_samples ??
      (typeof metrics.n_samples === 'number' ? metrics.n_samples : null)

    const run: BenchmarkRunExport = {
      job_id: job.job_id,
      benchmark_name: job.benchmark_name,
      study: datasheet.study,
      dataset: datasheet.dataset,
      cell_line: datasheet.cell_line,
      pe_system: datasheet.pe_system,
      weights: job.weights_id ?? job.result?.weights ?? null,
      n_samples: nSamples,
      finished_at: job.finished_at ?? null,
      performance: buildPerformanceMetrics(metrics),
    }

    if (!runsByModel.has(modelName)) {
      runsByModel.set(modelName, [])
    }
    runsByModel.get(modelName)!.push(run)
  }

  const models = [...runsByModel.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([model, runs]) => ({
      model,
      runs: [...runs].sort((left, right) => runSortKey(left).localeCompare(runSortKey(right))),
    }))

  return {
    exported_at: new Date().toISOString(),
    run_count: succeededJobs.length,
    model_count: models.length,
    models,
  }
}

export function benchmarkResultsToJson(
  jobs: BenchmarkJobStatusResponse[],
  indent = 2
): string {
  return JSON.stringify(buildBenchmarkResultsExport(jobs), null, indent)
}
