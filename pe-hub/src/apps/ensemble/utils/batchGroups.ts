import { buildFilterParams, type AttributeFilterRow } from '@apps/database/config/exportAttributes'
import peDbApi, { type ExportGroup } from '@apps/database/services/peDbApi'

export type BatchGroup = Pick<ExportGroup, 'study' | 'dataset' | 'cell_line' | 'pe_system'>

/** Resolve batch job targets from the catalog without model-format conversion. */
export async function resolveBatchGroups(filterRows: AttributeFilterRow[]): Promise<BatchGroup[]> {
  const filters = buildFilterParams(filterRows)
  const response = await peDbApi.filterDatasheets(filters)
  const groups: BatchGroup[] = []
  for (const sheet of response.data.datasheets ?? []) {
    const study = sheet.study_name?.trim()
    const dataset = sheet.dataset_name?.trim()
    if (!study || !dataset) continue
    groups.push({
      study,
      dataset,
      cell_line: sheet.cell_line,
      pe_system: sheet.pe_system,
    })
  }
  return groups
}
