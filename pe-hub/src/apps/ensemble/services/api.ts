import axios from 'axios'
import { ENSEMBLE_API_URL } from '@config/services'

const apiClient = axios.create({
  baseURL: ENSEMBLE_API_URL,
  headers: {
    'Content-Type': 'application/json',
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
}

export const api = {
  healthCheck: () => apiClient.get('/health'),
  listModels: () => apiClient.get<ModelsListResponse>('/models'),
  getModel: (name: string) => apiClient.get(`/models/${name}`),
  predict: (request: PredictionRequest): Promise<{ data: PredictionResponse }> =>
    apiClient.post('/predict', request),
  train: (request: TrainingRequest) => apiClient.post('/train', request),
  getTrainingStatus: (jobId: string) => apiClient.get(`/train/status/${jobId}`),
  evaluate: (data: unknown) => apiClient.post('/evaluate', data),
  ensemblePredict: (data: unknown) => apiClient.post('/ensemble', data),
}

export default api
