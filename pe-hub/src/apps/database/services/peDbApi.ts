import axios from 'axios'
import { PE_DB_URL } from '@config/services'

const apiClient = axios.create({
  baseURL: PE_DB_URL,
  headers: {
    'Content-Type': 'application/json',
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

export interface Statistics {
  total_entries: number
  total_studies: number
  edit_type: { study: string; edit_type: string; count: number }[]
}

export const peDbApi = {
  healthCheck: () => apiClient.get('/health'),
  listStudies: () => apiClient.get<Study[]>('/api/studies'),
  listDatasets: (study?: string) =>
    apiClient.get<Dataset[]>('/api/datasets', { params: study ? { study } : {} }),
  listDatasheets: (study?: string, dataset?: string) =>
    apiClient.get<Datasheet[]>('/api/datasheets', {
      params: { ...(study && { study }), ...(dataset && { dataset }) },
    }),
  getStatistics: () => apiClient.get<Statistics>('/api/statistics'),
}

export default peDbApi
