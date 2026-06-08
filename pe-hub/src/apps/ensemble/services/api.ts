import axios from 'axios'
import { ENSEMBLE_API_URL } from '@config/services'
import type { SplitExportParams } from '@apps/ensemble/config/splitParams'

const apiClient = axios.create({
  baseURL: ENSEMBLE_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  paramsSerializer: {
    indexes: null,
  },
})

export interface Model {
  name: string
  description: string
  type: string
  status: string
}

export interface ModelsListResponse {
  models: Model[]
  count: number
}

export interface PredictionRequest {
  model_name: string
  sequences: string[]
  cell_type?: string
}

export interface PredictionResponse {
  predictions: number[]
  model: string
  timestamp: string
}

export interface TrainingRequest {
  model_name: string
  dataset_source: string
  dataset_name: string
  hyperparameters?: Record<string, unknown>
  split?: SplitExportParams
  study?: string[]
  dataset?: string[]
  cell_line?: string[]
  pe_system?: string[]
  edit_type?: string[]
  edit_length?: number[]
  edit_scope?: string[]
  experimental_method?: string[]
  target_context?: string[]
  scaffold_name?: string[]
  edit_efficiency_min?: number
  edit_efficiency_max?: number
  records?: Record<string, unknown>[]
  model_kwargs?: Record<string, unknown>
}

export interface EvaluationRequest {
  model_name: string
  weights?: string
  split?: SplitExportParams
  study?: string[]
  dataset?: string[]
  cell_line?: string[]
  pe_system?: string[]
  edit_type?: string[]
  edit_length?: number[]
  edit_scope?: string[]
  experimental_method?: string[]
  target_context?: string[]
  scaffold_name?: string[]
  edit_efficiency_min?: number
  edit_efficiency_max?: number
  records?: Record<string, unknown>[]
  model_kwargs?: Record<string, unknown>
}

export type ExportFormat = 'std' | 'deepprime' | 'pridict' | 'pridict2' | 'oped'

export type ExportFilterParams = Partial<{
  study: string[]
  dataset: string[]
  cell_line: string[]
  pe_system: string[]
  edit_type: string[]
  edit_length: number[]
  edit_scope: string[]
  experimental_method: string[]
  target_context: string[]
  scaffold_name: string[]
  edit_efficiency_min: number
  edit_efficiency_max: number
}>

export interface ExportGroup {
  study: string
  dataset: string
  cell_line: string
  pe_system: string
  num_records: number
  columns: string[]
  records: Record<string, unknown>[]
}

export interface ExportResponse {
  status: string
  target_format: ExportFormat
  groups: ExportGroup[]
  skipped: { study: string; dataset: string; cell_line: string; pe_system: string; reason: string }[]
  total_records: number
  merged?: boolean
  split?: {
    strategy: string
    use_original_fold?: boolean
    random_state?: number
    summaries?: Array<Record<string, unknown>>
  }
}

export const api = {
  healthCheck: () => apiClient.get('/health'),
  listModels: () => apiClient.get<ModelsListResponse>('/models'),
  getModel: (name: string) => apiClient.get(`/models/${name}`),
  predict: (request: PredictionRequest): Promise<{ data: PredictionResponse }> =>
    apiClient.post('/predict', request),
  train: (request: TrainingRequest) => apiClient.post('/train', request),
  getTrainingStatus: (jobId: string) => apiClient.get(`/train/status/${jobId}`),
  evaluate: (request: EvaluationRequest) => apiClient.post('/evaluate', request),
  exportFiltered: (
    format: ExportFormat,
    filters: ExportFilterParams = {},
    split: SplitExportParams = { split_strategy: 'none' }
  ) =>
    apiClient.get<ExportResponse>('/data/filter', {
      params: { format, ...filters, ...split },
    }),
  ensemblePredict: (data: unknown) => apiClient.post('/ensemble', data),
}

export default api
