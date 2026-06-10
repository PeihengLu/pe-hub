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

export interface WeightSet {
  id: string
  model: string
  label: string
  source: 'vendor' | 'trained' | string
  format?: string
  created_at?: string
  metrics?: Record<string, unknown>
  notes?: string
}

export interface ModelWeightsResponse {
  model: string
  weights: WeightSet[]
  count: number
}

export interface ComputeDevice {
  device_id: string
  kind: string
  index?: number | null
  name: string
  is_accelerator: boolean
}

export interface DevicesListResponse {
  default: string
  devices: ComputeDevice[]
  count: number
}

export interface ComputeDeviceStatus {
  device_id: string
  running_job_id?: string | null
  running_job_kind?: 'train' | 'evaluate' | null
  queued_jobs: number
}

export interface ComputeDevicesResponse {
  default: string
  devices: ComputeDeviceStatus[]
}

export interface PredictionRequest {
  model_name: string
  sequences: string[]
  cell_type?: string
  weights?: string
  device?: string
}

export interface PredictionResponse {
  predictions: number[]
  model: string
  weights?: string | null
  timestamp: string
  message?: string
}

export type TrainingJobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

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
  notes?: string
  device?: string
}

export interface TrainingJobCreatedResponse {
  job_id: string
  status: TrainingJobStatus
  message: string
}

export interface TrainingJobStatusResponse {
  job_id: string
  status: TrainingJobStatus
  model_name: string
  dataset_name: string
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  device_requested?: string | null
  device_assigned?: string | null
  queue_position?: number | null
  weights_id?: string | null
  weights_label?: string | null
  error?: string | null
  result?: Record<string, unknown>
}

export interface TrainingLogResponse {
  job_id: string
  status: TrainingJobStatus
  offset: number
  next_offset: number
  log: string
}

export interface TrainingJobsListResponse {
  jobs: TrainingJobStatusResponse[]
  count: number
}

export type BenchmarkJobStatus = TrainingJobStatus

export interface EvaluationRequest {
  model_name: string
  benchmark_name: string
  weights: string
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
  device?: string
}

export interface BenchmarkJobCreatedResponse {
  job_id: string
  status: BenchmarkJobStatus
  message: string
}

export interface BenchmarkJobStatusResponse {
  job_id: string
  status: BenchmarkJobStatus
  model_name: string
  benchmark_name: string
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  device_requested?: string | null
  device_assigned?: string | null
  queue_position?: number | null
  weights_id?: string | null
  error?: string | null
  result?: {
    model?: string
    benchmark_name?: string
    weights?: string | null
    device?: string
    n_samples?: number
    metrics?: Record<string, number>
  }
}

export interface BenchmarkLogResponse {
  job_id: string
  status: BenchmarkJobStatus
  offset: number
  next_offset: number
  log: string
}

export interface BenchmarkJobsListResponse {
  jobs: BenchmarkJobStatusResponse[]
  count: number
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
  listDevices: () => apiClient.get<DevicesListResponse>('/devices'),
  listTrainingDevices: () => apiClient.get<ComputeDevicesResponse>('/train/devices'),
  listBenchmarkDevices: () => apiClient.get<ComputeDevicesResponse>('/evaluate/devices'),
  listModelWeights: (modelName: string) =>
    apiClient.get<ModelWeightsResponse>(`/models/${modelName}/weights`),
  getModel: (name: string) => apiClient.get(`/models/${name}`),
  predict: (request: PredictionRequest): Promise<{ data: PredictionResponse }> =>
    apiClient.post('/predict', request),
  train: (request: TrainingRequest) =>
    apiClient.post<TrainingJobCreatedResponse>('/train', request),
  getTrainingStatus: (jobId: string) =>
    apiClient.get<TrainingJobStatusResponse>(`/train/status/${jobId}`),
  getTrainingLogs: (jobId: string, offset = 0) =>
    apiClient.get<TrainingLogResponse>(`/train/logs/${jobId}`, { params: { offset } }),
  listTrainingJobs: (limit = 20) =>
    apiClient.get<TrainingJobsListResponse>('/train/jobs', { params: { limit } }),
  deleteTrainingJob: (jobId: string) =>
    apiClient.delete<{ job_id: string; deleted: boolean }>(`/train/jobs/${jobId}`),
  benchmark: (request: EvaluationRequest) =>
    apiClient.post<BenchmarkJobCreatedResponse>('/evaluate', request),
  getBenchmarkStatus: (jobId: string) =>
    apiClient.get<BenchmarkJobStatusResponse>(`/evaluate/status/${jobId}`),
  getBenchmarkLogs: (jobId: string, offset = 0) =>
    apiClient.get<BenchmarkLogResponse>(`/evaluate/logs/${jobId}`, { params: { offset } }),
  listBenchmarkJobs: (limit = 20) =>
    apiClient.get<BenchmarkJobsListResponse>('/evaluate/jobs', { params: { limit } }),
  deleteBenchmarkJob: (jobId: string) =>
    apiClient.delete<{ job_id: string; deleted: boolean }>(`/evaluate/jobs/${jobId}`),
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
