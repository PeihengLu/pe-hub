import axios from 'axios'
import { PE_DB_URL } from '@config/services'
import type {
  ExportFormat,
  FilterAttributeKey,
  SplitExportParams,
} from '@apps/database/config/exportAttributes'

const apiClient = axios.create({
  baseURL: PE_DB_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  paramsSerializer: {
    indexes: null,
  },
})

export interface Study {
  id: number
  name: string
  publication_date?: string
  authors?: string
}

export interface Dataset {
  id: number
  name: string
  description?: string
  study_id: number
  study_name?: string
  standardizable: boolean
  edit_scope?: string
  experimental_method?: string
  target_context?: string
}

export interface Datasheet {
  id: number
  file_path: string
  dataset_id: number
  cell_line: string
  pe_system: string
  scaffold_id: number
  num_samples: number
  study_name?: string
  dataset_name?: string
}

export interface Scaffold {
  id: number
  name: string
  sequence: string
  description?: string
}

export interface Statistics {
  total_entries: number
  total_studies: number
  edit_type: { study: string; edit_type: string; count: number }[]
  edit_length: { study: string; edit_length: number; count: number }[]
  edit_scope: { study: string; edit_scope: string; count: number }[]
  experimental_method: { study: string; experimental_method: string; count: number }[]
  target_context: { study: string; target_context: string; count: number }[]
}

export interface ExportGroup {
  study: string
  dataset: string
  cell_line: string
  pe_system: string
  num_records: number
  columns: string[]
  records: Record<string, unknown>[]
}

export interface ExportSkipped {
  study: string
  dataset: string
  cell_line: string
  pe_system: string
  reason: string
}

export interface SplitSummary {
  strategy: string
  use_original_fold?: boolean
  random_state?: number
  summaries?: Array<Record<string, unknown>>
}

export interface ExportResponse {
  status: string
  target_format: ExportFormat
  groups: ExportGroup[]
  skipped: ExportSkipped[]
  total_records: number
  merged?: boolean
  summary_only?: boolean
  split?: SplitSummary
}

export interface FilterDatasheetsResponse {
  status: string
  format: null
  count: number
  datasheets: Datasheet[]
}

export type ExportFilterParams = Partial<
  Record<FilterAttributeKey, string[] | number[]>
>

export const peDbApi = {
  healthCheck: () => apiClient.get('/health'),
  listStudies: () => apiClient.get<Study[]>('/api/studies'),
  listDatasets: (study?: string) =>
    apiClient.get<Dataset[]>('/api/datasets', { params: study ? { study } : {} }),
  listDatasheets: (study?: string, dataset?: string) =>
    apiClient.get<Datasheet[]>('/api/datasheets', {
      params: { ...(study && { study }), ...(dataset && { dataset }) },
    }),
  listScaffolds: () => apiClient.get<Scaffold[]>('/api/scaffolds'),
  getStatistics: () => apiClient.get<Statistics>('/api/statistics'),
  exportFiltered: (
    format: ExportFormat,
    filters: ExportFilterParams = {},
    split: SplitExportParams
  ) =>
    apiClient.get<ExportResponse>('/api/filter', {
      params: { format, ...filters, ...split },
    }),
  /** Catalog-only filter (no model conversion). Use to discover batch job targets quickly. */
  filterDatasheets: (filters: ExportFilterParams = {}) =>
    apiClient.get<FilterDatasheetsResponse>('/api/filter', {
      params: filters,
    }),
  /** Standardized-data summary for train/benchmark preview (no model conversion). */
  previewFiltered: (filters: ExportFilterParams = {}, split: SplitExportParams) =>
    apiClient.get<ExportResponse>('/api/filter', {
      params: { format: 'std', summary_only: true, ...filters, ...split },
    }),
}

export default peDbApi
