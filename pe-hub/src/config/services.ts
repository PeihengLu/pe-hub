export type ServiceId = 'pe-db' | 'pe-ensemble'

export interface ServiceDefinition {
  id: ServiceId
  name: string
  shortName: string
  description: string
  defaultPort: number
  docsPath: string
  healthPath: string
  startupCommands: string[]
}

export const PE_DB_URL =
  import.meta.env.VITE_PE_DB_URL?.trim() || 'http://localhost:8000'

export const ENSEMBLE_API_URL =
  import.meta.env.VITE_ENSEMBLE_API_URL?.trim() || 'http://localhost:8001'

export const SERVICES: Record<ServiceId, ServiceDefinition> = {
  'pe-db': {
    id: 'pe-db',
    name: 'PE Database',
    shortName: 'Database',
    description: 'Prime editing efficiency catalog and data APIs',
    defaultPort: 8000,
    docsPath: '/docs',
    healthPath: '/health',
    startupCommands: [
      './start-all.sh',
      'cd services/pe-db && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000',
    ],
  },
  'pe-ensemble': {
    id: 'pe-ensemble',
    name: 'PE Ensemble',
    shortName: 'Ensemble',
    description: 'Model training, evaluation, and ensemble prediction APIs',
    defaultPort: 8001,
    docsPath: '/docs',
    healthPath: '/health',
    startupCommands: [
      './start-all.sh',
      'cd services/pe-ensemble',
      'PE_DB_URL=http://localhost:8000 uvicorn app.main:app --reload --host 0.0.0.0 --port 8001',
    ],
  },
}

export function serviceBaseUrl(id: ServiceId): string {
  return id === 'pe-db' ? PE_DB_URL : ENSEMBLE_API_URL
}

export function serviceHealthUrl(id: ServiceId): string {
  const base = serviceBaseUrl(id)
  return `${base.replace(/\/$/, '')}${SERVICES[id].healthPath}`
}

export function serviceDocsUrl(id: ServiceId): string {
  const base = serviceBaseUrl(id)
  return `${base.replace(/\/$/, '')}${SERVICES[id].docsPath}`
}
