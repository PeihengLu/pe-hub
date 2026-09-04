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

export type TrainingJobStatus =
  | 'queued'
  | 'running'
  | 'stopping'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'skipped'

export interface JobDeleteAcceptedResponse {
  job_id: string
  accepted: boolean
  status: TrainingJobStatus | 'deleted'
}

export interface TrainingRequest {
  model_name: string
  dataset_source: string
  dataset_name: string
  hyperparameters?: Record<string, unknown>
  hyperparameter_mode?: 'merge' | 'replace'
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
  benchmark_name?: string
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
  auto_training_benchmark?: boolean
  allow_data_leak?: boolean
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
    skipped?: boolean
    skip_reason?: string
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

export type CombineMethod =
  | 'mean'
  | 'weighted_mean'
  | 'median'
  | 'trimmed_mean'
  | 'rank_mean'
  | 'percentile_mean'
  | 'geometric_mean'
  | 'harmonic_mean'
  | 'min'
  | 'max'

export interface EnsembleMemberInput {
  model_name: string
  weights: string
  member_weight?: number
}

export interface EnsembleRequest {
  ensemble_name: string
  combine: CombineMethod
  combine_options?: Record<string, unknown>
  members: EnsembleMemberInput[]
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
  allow_data_leak?: boolean
}

export type EnsembleJobStatus = TrainingJobStatus

export interface EnsembleJobCreatedResponse {
  job_id: string
  status: EnsembleJobStatus
  message: string
}

export interface EnsembleMemberMetrics {
  model_name: string
  weights: string
  metrics?: Record<string, number>
}

export interface EnsembleJobStatusResponse {
  job_id: string
  status: EnsembleJobStatus
  ensemble_name: string
  combine: CombineMethod
  member_count: number
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  device_requested?: string | null
  device_assigned?: string | null
  queue_position?: number | null
  error?: string | null
  result?: {
    ensemble_name?: string
    combine?: CombineMethod
    device?: string
    n_samples?: number
    skipped?: boolean
    skip_reason?: string
    metrics?: Record<string, number>
    member_metrics?: EnsembleMemberMetrics[]
    alignment?: Record<string, unknown>
  }
}

export interface EnsembleLogResponse {
  job_id: string
  status: EnsembleJobStatus
  offset: number
  next_offset: number
  log: string
}

export interface EnsembleJobsListResponse {
  jobs: EnsembleJobStatusResponse[]
  count: number
}

export interface CombineMethodHelp {
  id: CombineMethod
  label: string
  description: string
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

export type PluginStatus = 'pending' | 'active' | 'rejected' | string

export interface PluginValidationCheck {
  id: string
  passed: boolean
  detail: string
  duration_ms: number
}

export interface PluginValidationReport {
  plugin_name: string
  passed: boolean
  validated_at?: string | null
  checks: PluginValidationCheck[]
}

export interface PluginSummary {
  name: string
  status: PluginStatus
  version: string
  display_name: string
  updated_at?: string | null
  validation_passed?: boolean | null
  validated_at?: string | null
  check_count?: number
  failed_checks?: string[]
}

export interface PluginsListResponse {
  plugins: PluginSummary[]
  count: number
}

export interface PluginDetail {
  name: string
  status: PluginStatus
  updated_at?: string | null
  file_hashes?: Record<string, string>
  manifest: Record<string, unknown>
  validation_report?: PluginValidationReport | null
  validation_log_exists?: boolean
}

export interface PluginUploadResponse {
  name: string
  status: PluginStatus
  message: string
}

export interface PluginValidationJobCreatedResponse {
  job_id: string
  plugin_name: string
  status: TrainingJobStatus
  message: string
}

export interface PluginValidationJobStatusResponse {
  job_id: string
  plugin_name: string
  status: TrainingJobStatus
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  error?: string | null
  result?: {
    plugin_name?: string
    validation_report?: PluginValidationReport
    status?: PluginStatus
  }
}

export interface PluginValidationLogResponse {
  job_id: string
  plugin_name: string
  status: TrainingJobStatus
  offset: number
  next_offset: number
  log: string
}

export interface PluginActivateResponse {
  name: string
  status: PluginStatus
  ensemble_loaded: string[]
  pe_db_reload: Record<string, unknown>
}

export interface PluginDeleteResponse {
  name: string
  deleted: boolean
}

export type PluginUploadMode = 'bundle' | 'manifest' | 'form'

export interface PluginUploadPayload {
  upload_mode: PluginUploadMode
  name?: string
  version?: string
  display_name?: string
  description?: string
  wrapper_class?: string
  weight_format?: string
  authors?: string
  convert_entrypoint?: string
  pe_db_format?: string
  output_columns?: string
  required_std_columns?: string
  label_column?: string
  hyperparameters_json?: string
  weights_json?: string
  replace_existing?: boolean
  convert_file?: File | null
  wrapper_file?: File | null
  bundle_zip?: File | null
  weight_id?: string
  weight_file?: File | null
  manifest_file?: File | null
}

function buildPluginUploadFormData(payload: PluginUploadPayload): FormData {
  const formData = new FormData()
  if (payload.name) formData.append('name', payload.name)
  if (payload.version) formData.append('version', payload.version)
  if (payload.display_name) formData.append('display_name', payload.display_name)
  if (payload.description) formData.append('description', payload.description)
  if (payload.wrapper_class) formData.append('wrapper_class', payload.wrapper_class)
  if (payload.weight_format) formData.append('weight_format', payload.weight_format)
  if (payload.authors) formData.append('authors', payload.authors)
  if (payload.convert_entrypoint) formData.append('convert_entrypoint', payload.convert_entrypoint)
  if (payload.pe_db_format) formData.append('pe_db_format', payload.pe_db_format)
  if (payload.output_columns) formData.append('output_columns', payload.output_columns)
  if (payload.required_std_columns) {
    formData.append('required_std_columns', payload.required_std_columns)
  }
  if (payload.label_column) formData.append('label_column', payload.label_column)
  if (payload.hyperparameters_json) {
    formData.append('hyperparameters_json', payload.hyperparameters_json)
  }
  if (payload.weights_json) formData.append('weights_json', payload.weights_json)
  formData.append('replace_existing', String(Boolean(payload.replace_existing)))
  if (payload.convert_file) formData.append('convert_file', payload.convert_file)
  if (payload.wrapper_file) formData.append('wrapper_file', payload.wrapper_file)
  if (payload.bundle_zip) formData.append('bundle_zip', payload.bundle_zip)
  if (payload.weight_id) formData.append('weight_id', payload.weight_id)
  if (payload.weight_file) formData.append('weight_file', payload.weight_file)
  if (payload.manifest_file) formData.append('manifest_file', payload.manifest_file)
  return formData
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
    apiClient.delete<JobDeleteAcceptedResponse>(`/train/jobs/${jobId}`, {
      validateStatus: (status) => status === 202,
    }),
  benchmark: (request: EvaluationRequest) =>
    apiClient.post<BenchmarkJobCreatedResponse>('/evaluate', request),
  getBenchmarkStatus: (jobId: string) =>
    apiClient.get<BenchmarkJobStatusResponse>(`/evaluate/status/${jobId}`),
  getBenchmarkLogs: (jobId: string, offset = 0) =>
    apiClient.get<BenchmarkLogResponse>(`/evaluate/logs/${jobId}`, { params: { offset } }),
  listBenchmarkJobs: (limit = 20) =>
    apiClient.get<BenchmarkJobsListResponse>('/evaluate/jobs', { params: { limit } }),
  deleteBenchmarkJob: (jobId: string) =>
    apiClient.delete<JobDeleteAcceptedResponse>(`/evaluate/jobs/${jobId}`, {
      validateStatus: (status) => status === 202,
    }),
  exportFiltered: (
    format: ExportFormat,
    filters: ExportFilterParams = {},
    split: SplitExportParams = { split_strategy: 'none' }
  ) =>
    apiClient.get<ExportResponse>('/data/filter', {
      params: { format, ...filters, ...split },
    }),
  listEnsembleMethods: () =>
    apiClient.get<{ methods: CombineMethodHelp[]; count: number }>('/ensemble/methods'),
  runEnsemble: (request: EnsembleRequest) =>
    apiClient.post<EnsembleJobCreatedResponse>('/ensemble', request),
  getEnsembleStatus: (jobId: string) =>
    apiClient.get<EnsembleJobStatusResponse>(`/ensemble/status/${jobId}`),
  getEnsembleLogs: (jobId: string, offset = 0) =>
    apiClient.get<EnsembleLogResponse>(`/ensemble/logs/${jobId}`, { params: { offset } }),
  listEnsembleJobs: (limit = 20) =>
    apiClient.get<EnsembleJobsListResponse>('/ensemble/jobs', { params: { limit } }),
  deleteEnsembleJob: (jobId: string) =>
    apiClient.delete<JobDeleteAcceptedResponse>(`/ensemble/jobs/${jobId}`, {
      validateStatus: (status) => status === 202,
    }),
  listEnsembleDevices: () => apiClient.get<ComputeDevicesResponse>('/ensemble/devices'),
  listPlugins: () => apiClient.get<PluginsListResponse>('/models/plugins'),
  getPlugin: (name: string) => apiClient.get<PluginDetail>(`/models/plugins/${encodeURIComponent(name)}`),
  uploadPlugin: (payload: PluginUploadPayload) =>
    apiClient.post<PluginUploadResponse>(
      '/models/plugins',
      buildPluginUploadFormData(payload),
      {
        transformRequest: (data, headers) => {
          if (headers && data instanceof FormData) {
            delete headers['Content-Type']
          }
          return data
        },
      }
    ),
  validatePlugin: (name: string) =>
    apiClient.post<PluginValidationJobCreatedResponse>(
      `/models/plugins/${encodeURIComponent(name)}/validate`,
      undefined,
      { validateStatus: (status) => status === 202 || status === 200 }
    ),
  getPluginValidationStatus: (name: string, jobId: string) =>
    apiClient.get<PluginValidationJobStatusResponse>(
      `/models/plugins/${encodeURIComponent(name)}/validate/status/${jobId}`
    ),
  getPluginValidationLogs: (name: string, jobId: string, offset = 0) =>
    apiClient.get<PluginValidationLogResponse>(
      `/models/plugins/${encodeURIComponent(name)}/validate/logs/${jobId}`,
      { params: { offset } }
    ),
  cancelPluginValidation: (name: string, jobId: string) =>
    apiClient.delete(`/models/plugins/${encodeURIComponent(name)}/validate/jobs/${jobId}`, {
      validateStatus: (status) => status === 202,
    }),
  getPluginValidationLog: (name: string, offset = 0) =>
    apiClient.get<{ name: string; offset: number; next_offset: number; log: string }>(
      `/models/plugins/${encodeURIComponent(name)}/validation.log`,
      { params: { offset } }
    ),
  activatePlugin: (name: string) =>
    apiClient.post<PluginActivateResponse>(`/models/plugins/${encodeURIComponent(name)}/activate`),
  deletePlugin: (name: string) =>
    apiClient.delete<PluginDeleteResponse>(`/models/plugins/${encodeURIComponent(name)}`),
}

export default api
