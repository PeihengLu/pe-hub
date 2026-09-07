export type ExportFormat = 'std' | 'deepprime' | 'pridict' | 'pridict2' | 'oped'

export type SplitStrategy = 'none' | 'holdout_2' | 'holdout_3' | 'cv'

export const FILTER_LIST_FIELDS = [
  'study',
  'dataset',
  'cell_line',
  'pe_system',
  'edit_type',
  'edit_length',
  'edit_scope',
  'experimental_method',
  'target_context',
  'scaffold_name',
] as const

export const FILTER_RANGE_FIELDS = ['edit_efficiency_min', 'edit_efficiency_max'] as const

export const SPLIT_QUERY_FIELDS = [
  'split_strategy',
  'train_pct',
  'val_pct',
  'test_pct',
  'cv_folds',
  'use_original_fold',
  'original_fold_test_value',
  'split_random_state',
  'merge',
] as const

export type FilterAttributeKey = (typeof FILTER_LIST_FIELDS)[number]
export type FilterRangeKey = (typeof FILTER_RANGE_FIELDS)[number]
export type SplitQueryField = (typeof SPLIT_QUERY_FIELDS)[number]

export const SPLIT_STRATEGIES: {
  value: SplitStrategy
  label: string
  description: string
}[] = [
  {
    value: 'none',
    label: 'No split',
    description: 'Export rows without split assignment columns.',
  },
  {
    value: 'holdout_2',
    label: 'Train / test',
    description: 'Group-aware train and test partitions (percentages must sum to 1).',
  },
  {
    value: 'holdout_3',
    label: 'Train / val / test',
    description:
      'Group-aware three-way holdout: test rows use the same logic as Train / test, then val is carved from the remaining training pool.',
  },
  {
    value: 'cv',
    label: 'Cross-validation',
    description: 'Group-aware CV folds; optional test_pct for a held-out test set.',
  },
]

export interface SplitExportParams {
  split_strategy: SplitStrategy
  train_pct?: number
  val_pct?: number
  test_pct?: number
  cv_folds?: number
  use_original_fold?: boolean
  original_fold_test_value?: number
  split_random_state?: number
  merge?: boolean
}

export function buildSplitParams(config: {
  strategy: SplitStrategy
  trainPct: string
  valPct: string
  testPct: string
  cvFolds: string
  useOriginalFold: boolean
  originalFoldTestValue: string
  randomState: string
  merge: boolean
}): SplitExportParams {
  const params: SplitExportParams = {
    split_strategy: config.strategy,
    use_original_fold: config.useOriginalFold,
    original_fold_test_value: Number(config.originalFoldTestValue),
    split_random_state: Number(config.randomState) || 0,
    merge: config.merge,
  }

  if (config.strategy === 'holdout_2') {
    params.train_pct = Number(config.trainPct)
    params.test_pct = Number(config.testPct)
  } else if (config.strategy === 'holdout_3') {
    params.train_pct = Number(config.trainPct)
    params.val_pct = Number(config.valPct)
    params.test_pct = Number(config.testPct)
  } else if (config.strategy === 'cv') {
    params.cv_folds = Number(config.cvFolds)
    if (config.testPct.trim() !== '') {
      params.test_pct = Number(config.testPct)
    }
  }

  return params
}

export interface FilterAttributeDef {
  key: FilterAttributeKey
  label: string
}

const FILTER_ATTRIBUTE_LABELS: Record<FilterAttributeKey, string> = {
  study: 'Study',
  dataset: 'Dataset',
  cell_line: 'Cell line',
  pe_system: 'PE system',
  edit_type: 'Edit type',
  edit_length: 'Edit length',
  edit_scope: 'Edit scope',
  experimental_method: 'Experimental method',
  target_context: 'Target context',
  scaffold_name: 'Scaffold',
}

export type CatalogFilterParams = {
  [K in FilterAttributeKey]?: K extends 'edit_length' ? number[] : string[]
} & {
  [K in FilterRangeKey]?: number
}

export const EXPORT_FORMATS: { value: ExportFormat; label: string; description: string }[] = [
  {
    value: 'std',
    label: 'Standardized',
    description: 'Full PE-DB standardized schema (parquet columns)',
  },
  {
    value: 'deepprime',
    label: 'DeepPrime',
    description: 'DeepPrime model input format',
  },
  {
    value: 'pridict',
    label: 'PRIDICT',
    description: 'PRIDICT v1 model input format',
  },
  {
    value: 'pridict2',
    label: 'PRIDICT2',
    description: 'PRIDICT2 model input format',
  },
  {
    value: 'oped',
    label: 'OPED',
    description: 'OPED model input format',
  },
]

export const FILTER_ATTRIBUTES: FilterAttributeDef[] = FILTER_LIST_FIELDS.map((key) => ({
  key,
  label: FILTER_ATTRIBUTE_LABELS[key],
}))

export const STATIC_FILTER_OPTIONS: Partial<Record<FilterAttributeKey, string[]>> = {
  edit_type: ['sub', 'ins', 'del'],
  edit_scope: ['on_target', 'off_target'],
  experimental_method: ['in_vitro', 'in_vivo'],
  target_context: ['endogenous', 'non_endogenous'],
}

export interface AttributeFilterRow {
  id: string
  attribute: FilterAttributeKey | ''
  values: string[]
}

export function buildFilterParams(
  rows: AttributeFilterRow[]
): Record<string, string[] | number[]> {
  const params: Record<string, string[] | number[]> = {}
  for (const row of rows) {
    if (!row.attribute || row.values.length === 0) continue
    if (row.attribute === 'edit_length') {
      params.edit_length = row.values.map((v) => Number(v))
      continue
    }
    params[row.attribute] = row.values
  }
  return params
}
