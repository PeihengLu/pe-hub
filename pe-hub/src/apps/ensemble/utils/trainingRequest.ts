import {
  buildFilterParams,
  buildSplitParams,
  type AttributeFilterRow,
  type SplitStrategy,
} from '@apps/database/config/exportAttributes'
import type { ExportGroup } from '@apps/database/services/peDbApi'
import type { TrainingRequest } from '@apps/ensemble/services/api'
import type { SplitExportParams } from '@apps/ensemble/config/splitParams'

export function buildTrainingSplitParams(config: {
  strategy: SplitStrategy
  trainPct: string
  valPct: string
  testPct: string
  cvFolds: string
  useOriginalFold: boolean
  originalFoldTestValue: string
  randomState: string
  batchTraining: boolean
}): SplitExportParams {
  return buildSplitParams({
    strategy: config.strategy,
    trainPct: config.trainPct,
    valPct: config.valPct,
    testPct: config.testPct,
    cvFolds: config.cvFolds,
    useOriginalFold: config.useOriginalFold,
    originalFoldTestValue: config.originalFoldTestValue,
    randomState: config.randomState,
    merge: !config.batchTraining,
  })
}

const DATASHEET_FILTER_KEYS = ['study', 'dataset', 'cell_line', 'pe_system'] as const

const ROW_FILTER_KEYS = [
  'edit_type',
  'edit_length',
  'edit_scope',
  'experimental_method',
  'target_context',
  'scaffold_name',
] as const

function formatAttachedRowFilters(
  filters: Record<string, string[] | number[]>
): string {
  const parts: string[] = []
  for (const key of ROW_FILTER_KEYS) {
    const values = filters[key]
    if (!Array.isArray(values) || values.length === 0) continue
    parts.push(`${key}=${values.join('|')}`)
  }
  return parts.join(', ')
}

export function buildDatasetLabel(
  filterRows: AttributeFilterRow[],
  group?: Pick<ExportGroup, 'study' | 'dataset' | 'cell_line' | 'pe_system'>
): string {
  const filters = buildFilterParams(filterRows)
  const attached = formatAttachedRowFilters(filters)

  let base: string
  if (group) {
    base = `${group.study}/${group.dataset} · ${group.cell_line} · ${group.pe_system}`
  } else {
    const parts: string[] = []
    for (const key of DATASHEET_FILTER_KEYS) {
      const values = filters[key]
      if (Array.isArray(values) && values.length > 0) {
        parts.push(`${key}=${values.join('|')}`)
      }
    }
    base = parts.length > 0 ? parts.join(', ') : 'all matching datasheets'
  }

  return attached ? `${base} (${attached})` : base
}

export function buildTrainingRequestForGroup(input: {
  modelName: string
  device: string
  hyperparameters: Record<string, unknown>
  modelKwargs?: Record<string, unknown>
  split: SplitExportParams
  filterRows: AttributeFilterRow[]
  group?: Pick<ExportGroup, 'study' | 'dataset' | 'cell_line' | 'pe_system'>
  notes?: string
}): TrainingRequest {
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
    dataset_source: 'pe-db',
    dataset_name: buildDatasetLabel(input.filterRows, input.group),
    split: input.split,
    device: input.device,
    hyperparameters: input.hyperparameters,
    model_kwargs: input.modelKwargs,
    notes: input.notes,
    ...filters,
  }
}

export function singleValueFromFilters(
  filterRows: AttributeFilterRow[],
  attribute: 'cell_line' | 'pe_system'
): string | undefined {
  const row = filterRows.find((item) => item.attribute === attribute)
  if (!row || row.values.length !== 1) return undefined
  return row.values[0]
}
