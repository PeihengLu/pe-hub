import { buildFilterParams, type AttributeFilterRow } from '@apps/database/config/exportAttributes'
import type { ExportGroup } from '@apps/database/services/peDbApi'
import type { EvaluationRequest } from '@apps/ensemble/services/api'
import type { SplitExportParams } from '@apps/ensemble/config/splitParams'
import { buildDatasetLabel, singleValueFromFilters } from '@apps/ensemble/utils/trainingRequest'

export { buildTrainingSplitParams as buildBenchmarkSplitParams } from '@apps/ensemble/utils/trainingRequest'

export function buildBenchmarkRequestForGroup(input: {
  modelName: string
  device: string
  weights: string
  split: SplitExportParams
  filterRows: AttributeFilterRow[]
  group?: Pick<ExportGroup, 'study' | 'dataset' | 'cell_line' | 'pe_system'>
  modelKwargs?: Record<string, unknown>
}): EvaluationRequest {
  const filters = input.group
    ? {
        study: [input.group.study],
        dataset: [input.group.dataset],
        cell_line: [input.group.cell_line],
        pe_system: [input.group.pe_system],
      }
    : buildFilterParams(input.filterRows)

  return {
    model_name: input.modelName,
    benchmark_name: buildDatasetLabel(input.filterRows, input.group),
    weights: input.weights,
    split: input.split,
    device: input.device,
    model_kwargs: input.modelKwargs,
    ...filters,
  }
}

export function deepprimeModelKwargsFromFilters(
  filterRows: AttributeFilterRow[],
  group?: Pick<ExportGroup, 'cell_line' | 'pe_system'>
): Record<string, unknown> | undefined {
  const cellLine = group?.cell_line ?? singleValueFromFilters(filterRows, 'cell_line')
  const peSystem = group?.pe_system ?? singleValueFromFilters(filterRows, 'pe_system')
  if (!cellLine || !peSystem) return undefined
  return { cell_type: cellLine, pe_system: peSystem }
}
