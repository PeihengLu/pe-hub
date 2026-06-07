export type ExportFormat = 'std' | 'deepprime' | 'pridict' | 'pridict2' | 'oped'

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
