import axios from 'axios'

const API_BASE_URL = 'http://localhost:8001'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
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
  // Health check
  healthCheck: () => apiClient.get('/health'),

  // Models
  listModels: (): Promise<{ data: Model[] }> => apiClient.get('/models'),
  getModel: (name: string) => apiClient.get(`/models/${name}`),

  // Predictions
  predict: (request: PredictionRequest): Promise<{ data: PredictionResponse }> =>
    apiClient.post('/predict', request),

  // Training
  train: (request: TrainingRequest) => apiClient.post('/train', request),
  getTrainingStatus: (jobId: string) => apiClient.get(`/train/status/${jobId}`),

  // Evaluation
  evaluate: (data: unknown) => apiClient.post('/evaluate', data),

  // Ensemble
  ensemblePredict: (data: unknown) => apiClient.post('/ensemble', data),
}

export default api
