import {
  filtersForDatasheetGroup,
  type AttributeFilterRow,
  type DatasheetGroup,
} from '@apps/database/config/exportAttributes'
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

export function buildBenchmarkRequestForGroup(input: {
  modelName: string
  device: string
  weights: string
  split: SplitExportParams
  filterRows: AttributeFilterRow[]
  group?: DatasheetGroup
}): EvaluationRequest {
  const filters = filtersForDatasheetGroup(input.filterRows, input.group)

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
