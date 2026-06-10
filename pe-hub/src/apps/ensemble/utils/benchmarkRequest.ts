import { buildFilterParams, type AttributeFilterRow } from '@apps/database/config/exportAttributes'
import type { ExportGroup } from '@apps/database/services/peDbApi'
import type { EvaluationRequest } from '@apps/ensemble/services/api'
import type { SplitExportParams } from '@apps/ensemble/config/splitParams'
import { buildDatasetLabel } from '@apps/ensemble/utils/trainingRequest'

export { buildTrainingSplitParams as buildBenchmarkSplitParams } from '@apps/ensemble/utils/trainingRequest'

export function buildBenchmarkRequestForGroup(input: {
  modelName: string
  device: string
  weights: string
  split: SplitExportParams
  filterRows: AttributeFilterRow[]
  group?: Pick<ExportGroup, 'study' | 'dataset' | 'cell_line' | 'pe_system'>
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
    ...filters,
  }
}
