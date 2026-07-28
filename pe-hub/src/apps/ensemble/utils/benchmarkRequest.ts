import { buildFilterParams, type AttributeFilterRow } from '@apps/database/config/exportAttributes'
import type { ExportGroup } from '@apps/database/services/peDbApi'
import type { EvaluationRequest } from '@apps/ensemble/services/api'
import type { SplitExportParams } from '@apps/ensemble/config/splitParams'
import { buildDatasetLabel } from '@apps/ensemble/utils/trainingRequest'

export { buildTrainingSplitParams as buildBenchmarkSplitParams } from '@apps/ensemble/utils/trainingRequest'

/** Evaluate a trained weight on its recorded held-out test set (model + weights only). */
export function buildAutoTrainingBenchmarkRequest(input: {
  modelName: string
  device: string
  weights: string
}): EvaluationRequest {
  return {
    model_name: input.modelName,
    weights: input.weights,
    device: input.device,
    auto_training_benchmark: true,
  }
}

const DATASHEET_FILTER_KEYS = new Set(['study', 'dataset', 'cell_line', 'pe_system'])

export function buildBenchmarkRequestForGroup(input: {
  modelName: string
  device: string
  weights: string
  split: SplitExportParams
  filterRows: AttributeFilterRow[]
  group?: Pick<ExportGroup, 'study' | 'dataset' | 'cell_line' | 'pe_system'>
}): EvaluationRequest {
  const rowFilters = buildFilterParams(input.filterRows)
  const filters = input.group
    ? {
        study: [input.group.study],
        dataset: [input.group.dataset],
        cell_line: [input.group.cell_line],
        pe_system: [input.group.pe_system],
        // Keep non-datasheet filters (e.g. edit_type=sub) on each batch job.
        ...Object.fromEntries(
          Object.entries(rowFilters).filter(([key]) => !DATASHEET_FILTER_KEYS.has(key))
        ),
      }
    : rowFilters

  return {
    model_name: input.modelName,
    benchmark_name: buildDatasetLabel(input.filterRows, input.group),
    weights: input.weights,
    split: input.split,
    device: input.device,
    auto_training_benchmark: false,
    ...filters,
  }
}
