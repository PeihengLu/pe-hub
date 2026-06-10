export type ExportFormat = 'std' | 'deepprime' | 'pridict' | 'pridict2' | 'oped'

export type SplitStrategy = 'none' | 'holdout_2' | 'holdout_3' | 'cv'

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
    description: 'Group-aware three-way holdout (incompatible with original_fold).',
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

export type FilterAttributeKey =
  | 'study'
  | 'dataset'
  | 'cell_line'
  | 'pe_system'
  | 'edit_type'
  | 'edit_length'
  | 'edit_scope'
  | 'experimental_method'
  | 'target_context'
  | 'scaffold_name'

export interface FilterAttributeDef {
  key: FilterAttributeKey
  label: string
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

export const FILTER_ATTRIBUTES: FilterAttributeDef[] = [
  { key: 'study', label: 'Study' },
  { key: 'dataset', label: 'Dataset' },
  { key: 'cell_line', label: 'Cell line' },
  { key: 'pe_system', label: 'PE system' },
  { key: 'edit_type', label: 'Edit type' },
  { key: 'edit_length', label: 'Edit length' },
  { key: 'edit_scope', label: 'Edit scope' },
  { key: 'experimental_method', label: 'Experimental method' },
  { key: 'target_context', label: 'Target context' },
  { key: 'scaffold_name', label: 'Scaffold' },
]

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
