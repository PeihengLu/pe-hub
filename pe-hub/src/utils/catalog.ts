import type { AttributeFilterRow, FilterAttributeKey } from '@apps/database/config/exportAttributes'
import { STATIC_FILTER_OPTIONS } from '@apps/database/config/exportAttributes'
import type { Dataset, Datasheet, Scaffold, Statistics, Study } from '@apps/database/services/peDbApi'

export function uniqueSorted(values: string[]): string[] {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b))
}

export type CatalogField = 'study' | 'dataset' | 'cell_line' | 'pe_system'

export interface CatalogSelection {
  study: string
  dataset: string
  cell_line: string
  pe_system: string
}

/** Attributes whose allowed values come from catalog tables (not per-row parquet). */
export const CATALOG_INSTANCE_ATTRIBUTES = new Set<FilterAttributeKey>([
  'study',
  'dataset',
  'cell_line',
  'pe_system',
  'scaffold_name',
  'edit_scope',
  'experimental_method',
  'target_context',
])

export interface CatalogSnapshot {
  studies: Study[]
  datasets: Dataset[]
  datasheets: Datasheet[]
  scaffolds: Scaffold[]
  statistics?: Statistics
}

export function collectFilterValues(
  rows: AttributeFilterRow[],
  attribute: FilterAttributeKey,
  excludeRowId?: string
): string[] {
  const values: string[] = []
  for (const row of rows) {
    if (row.id === excludeRowId || row.attribute !== attribute || row.values.length === 0) {
      continue
    }
    values.push(...row.values)
  }
  return values
}

function matchesAny(value: string, allowed: string[]): boolean {
  if (allowed.length === 0) return true
  const normalized = value.toLowerCase()
  return allowed.some((item) => item.toLowerCase() === normalized)
}

function datasetStudyName(dataset: Dataset, studiesById: Map<number, string>): string {
  return dataset.study_name ?? studiesById.get(dataset.study_id) ?? ''
}

function filterDatasets(
  snapshot: CatalogSnapshot,
  rows: AttributeFilterRow[],
  excludeRowId: string
): Dataset[] {
  const studiesById = new Map(snapshot.studies.map((study) => [study.id, study.name]))
  const studyFilter = collectFilterValues(rows, 'study', excludeRowId)
  const datasetFilter = collectFilterValues(rows, 'dataset', excludeRowId)
  const editScopeFilter = collectFilterValues(rows, 'edit_scope', excludeRowId)
  const experimentalMethodFilter = collectFilterValues(rows, 'experimental_method', excludeRowId)
  const targetContextFilter = collectFilterValues(rows, 'target_context', excludeRowId)

  return snapshot.datasets.filter((dataset) => {
    const studyName = datasetStudyName(dataset, studiesById)
    if (!matchesAny(studyName, studyFilter)) return false
    if (!matchesAny(dataset.name, datasetFilter)) return false
    if (dataset.edit_scope && !matchesAny(dataset.edit_scope, editScopeFilter)) return false
    if (dataset.experimental_method && !matchesAny(dataset.experimental_method, experimentalMethodFilter)) {
      return false
    }
    if (dataset.target_context && !matchesAny(dataset.target_context, targetContextFilter)) {
      return false
    }
    return true
  })
}

function filterDatasheets(
  snapshot: CatalogSnapshot,
  rows: AttributeFilterRow[],
  excludeRowId: string
): Datasheet[] {
  const allowedDatasets = new Set(filterDatasets(snapshot, rows, excludeRowId).map((dataset) => dataset.id))
  const scaffoldNames = collectFilterValues(rows, 'scaffold_name', excludeRowId)
  const scaffoldIds = new Set(
    snapshot.scaffolds
      .filter((scaffold) => matchesAny(scaffold.name, scaffoldNames))
      .map((scaffold) => scaffold.id)
  )

  const studyFilter = collectFilterValues(rows, 'study', excludeRowId)
  const datasetFilter = collectFilterValues(rows, 'dataset', excludeRowId)
  const cellLineFilter = collectFilterValues(rows, 'cell_line', excludeRowId)
  const peSystemFilter = collectFilterValues(rows, 'pe_system', excludeRowId)

  return snapshot.datasheets.filter((sheet) => {
    if (!allowedDatasets.has(sheet.dataset_id)) return false
    const studyName = sheet.study_name ?? ''
    const datasetName = sheet.dataset_name ?? ''
    if (!matchesAny(studyName, studyFilter)) return false
    if (!matchesAny(datasetName, datasetFilter)) return false
    if (!matchesAny(sheet.cell_line, cellLineFilter)) return false
    if (!matchesAny(sheet.pe_system, peSystemFilter)) return false
    if (scaffoldNames.length > 0 && !scaffoldIds.has(sheet.scaffold_id)) return false
    return true
  })
}

function rowLevelOptions(
  attribute: FilterAttributeKey,
  rows: AttributeFilterRow[],
  snapshot: CatalogSnapshot,
  excludeRowId: string
): string[] {
  const studyFilter = collectFilterValues(rows, 'study', excludeRowId)
  const staticFallback = STATIC_FILTER_OPTIONS[attribute] ?? []

  if (attribute === 'edit_length') {
    const rows_ = snapshot.statistics?.edit_length ?? []
    const filtered =
      studyFilter.length > 0
        ? rows_.filter((row) => matchesAny(row.study, studyFilter))
        : rows_
    const values = filtered.map((row) => String(row.edit_length))
    return values.length > 0 ? uniqueSorted(values) : staticFallback
  }

  if (attribute === 'edit_type') {
    const rows_ = snapshot.statistics?.edit_type ?? []
    const filtered =
      studyFilter.length > 0
        ? rows_.filter((row) => matchesAny(row.study, studyFilter))
        : rows_
    const values = filtered.map((row) => row.edit_type)
    return values.length > 0 ? uniqueSorted(values) : staticFallback
  }

  return staticFallback
}

export function optionsForAttribute(
  attribute: FilterAttributeKey,
  rows: AttributeFilterRow[],
  snapshot: CatalogSnapshot,
  excludeRowId?: string
): string[] {
  if (!CATALOG_INSTANCE_ATTRIBUTES.has(attribute)) {
    return rowLevelOptions(attribute, rows, snapshot, excludeRowId ?? '')
  }

  const rowId = excludeRowId ?? ''
  const matchingDatasheets = filterDatasheets(snapshot, rows, rowId)
  const matchingDatasets = filterDatasets(snapshot, rows, rowId)
  const scaffoldNameById = new Map(snapshot.scaffolds.map((scaffold) => [scaffold.id, scaffold.name]))

  switch (attribute) {
    case 'study':
      return uniqueSorted(
        matchingDatasheets.map((sheet) => sheet.study_name ?? '').filter(Boolean)
      )
    case 'dataset':
      return uniqueSorted(
        matchingDatasheets.map((sheet) => sheet.dataset_name ?? '').filter(Boolean)
      )
    case 'cell_line':
      return uniqueSorted(matchingDatasheets.map((sheet) => sheet.cell_line))
    case 'pe_system':
      return uniqueSorted(matchingDatasheets.map((sheet) => sheet.pe_system))
    case 'scaffold_name':
      return uniqueSorted(
        matchingDatasheets
          .map((sheet) => scaffoldNameById.get(sheet.scaffold_id) ?? '')
          .filter(Boolean)
      )
    case 'edit_scope':
      return uniqueSorted(
        matchingDatasets.map((dataset) => dataset.edit_scope ?? '').filter(Boolean)
      )
    case 'experimental_method':
      return uniqueSorted(
        matchingDatasets.map((dataset) => dataset.experimental_method ?? '').filter(Boolean)
      )
    case 'target_context':
      return uniqueSorted(
        matchingDatasets.map((dataset) => dataset.target_context ?? '').filter(Boolean)
      )
    default:
      return []
  }
}

export function buildBaseOptionsByAttribute(
  snapshot: CatalogSnapshot
): Partial<Record<FilterAttributeKey, string[]>> {
  return {
    study: uniqueSorted(snapshot.studies.map((study) => study.name)),
    dataset: uniqueSorted(snapshot.datasets.map((dataset) => dataset.name)),
    cell_line: uniqueSorted(snapshot.datasheets.map((sheet) => sheet.cell_line)),
    pe_system: uniqueSorted(snapshot.datasheets.map((sheet) => sheet.pe_system)),
    scaffold_name: uniqueSorted(snapshot.scaffolds.map((scaffold) => scaffold.name)),
    edit_length: uniqueSorted(
      (snapshot.statistics?.edit_length ?? []).map((row) => String(row.edit_length))
    ),
    edit_type: uniqueSorted(
      (snapshot.statistics?.edit_type ?? []).map((row) => row.edit_type)
    ),
    edit_scope: uniqueSorted(
      snapshot.datasets.map((dataset) => dataset.edit_scope ?? '').filter(Boolean)
    ),
    experimental_method: uniqueSorted(
      snapshot.datasets.map((dataset) => dataset.experimental_method ?? '').filter(Boolean)
    ),
    target_context: uniqueSorted(
      snapshot.datasets.map((dataset) => dataset.target_context ?? '').filter(Boolean)
    ),
  }
}
