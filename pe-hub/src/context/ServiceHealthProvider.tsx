import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  type ServiceId,
  SERVICES,
  serviceBaseUrl,
  serviceHealthUrl,
} from '@config/services'

export type ServiceStatus = 'checking' | 'up' | 'down'

export interface ServiceHealthState {
  status: ServiceStatus
  lastChecked: Date | null
  error: string | null
}

type HealthMap = Record<ServiceId, ServiceHealthState>

interface ServiceHealthContextValue {
  health: HealthMap
  /** Re-check all services. Use background=true for polling (keeps UI mounted). */
  refresh: (options?: { background?: boolean }) => void
  /** Re-check one service (e.g. Retry on offline screen). */
  refreshService: (id: ServiceId) => void
  isUp: (id: ServiceId) => boolean
}

const defaultState = (): ServiceHealthState => ({
  status: 'checking',
  lastChecked: null,
  error: null,
})

const ServiceHealthContext = createContext<ServiceHealthContextValue | null>(
  null
)

async function probeService(id: ServiceId): Promise<ServiceHealthState> {
  const url = serviceHealthUrl(id)
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    })
    if (!response.ok) {
      return {
        status: 'down',
        lastChecked: new Date(),
        error: `HTTP ${response.status}`,
      }
    }

    const base = serviceBaseUrl(id).replace(/\/$/, '')
    const rootResponse = await fetch(`${base}/`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    })
    if (rootResponse.ok) {
      const info = (await rootResponse.json()) as { service?: string; name?: string }
      if (id === 'pe-ensemble' && info.service !== 'PE Ensemble') {
        return {
          status: 'down',
          lastChecked: new Date(),
          error:
            'Wrong API at the Ensemble URL (got PE Database). Start pe-ensemble on port 8001 or fix VITE_ENSEMBLE_API_URL.',
        }
      }
      if (id === 'pe-db' && info.name !== 'PE Database API') {
        return {
          status: 'down',
          lastChecked: new Date(),
          error: 'Wrong API at the PE Database URL.',
        }
      }
    }

    return { status: 'up', lastChecked: new Date(), error: null }
  } catch (err) {
    const message =
      err instanceof TypeError
        ? 'Cannot reach server (is it running?)'
        : err instanceof Error
          ? err.message
          : 'Health check failed'
    return { status: 'down', lastChecked: new Date(), error: message }
  }
}

export function ServiceHealthProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthMap>({
    'pe-db': defaultState(),
    'pe-ensemble': defaultState(),
  })

  const applyProbeResults = useCallback(
    (results: Record<ServiceId, ServiceHealthState>) => {
      setHealth((prev) => ({
        'pe-db': results['pe-db'] ?? prev['pe-db'],
        'pe-ensemble': results['pe-ensemble'] ?? prev['pe-ensemble'],
      }))
    },
    []
  )

  const refresh = useCallback(
    async (options?: { background?: boolean }) => {
      const background = options?.background ?? false

      if (!background) {
        setHealth((prev) => ({
          'pe-db': {
            ...prev['pe-db'],
            status:
              prev['pe-db'].lastChecked === null ? 'checking' : prev['pe-db'].status,
          },
          'pe-ensemble': {
            ...prev['pe-ensemble'],
            status:
              prev['pe-ensemble'].lastChecked === null
                ? 'checking'
                : prev['pe-ensemble'].status,
          },
        }))
      }

      const ids = Object.keys(SERVICES) as ServiceId[]
      const results = await Promise.all(ids.map((id) => probeService(id)))
      applyProbeResults(
        ids.reduce(
          (acc, id, index) => {
            acc[id] = results[index]
            return acc
          },
          {} as Record<ServiceId, ServiceHealthState>
        )
      )
    },
    [applyProbeResults]
  )

  const refreshService = useCallback(
    async (id: ServiceId) => {
      setHealth((prev) => ({
        ...prev,
        [id]: { ...prev[id], status: 'checking' },
      }))
      const result = await probeService(id)
      setHealth((prev) => ({ ...prev, [id]: result }))
    },
    []
  )

  useEffect(() => {
    void refresh()
    const interval = window.setInterval(() => void refresh({ background: true }), 15_000)
    return () => window.clearInterval(interval)
  }, [refresh])

  const value = useMemo(
    () => ({
      health,
      refresh,
      refreshService,
      isUp: (id: ServiceId) => health[id].status === 'up',
    }),
    [health, refresh, refreshService]
  )

  return (
    <ServiceHealthContext.Provider value={value}>
      {children}
    </ServiceHealthContext.Provider>
  )
}

export function useServiceHealth() {
  const ctx = useContext(ServiceHealthContext)
  if (!ctx) {
    throw new Error('useServiceHealth must be used within ServiceHealthProvider')
  }
  return ctx
}
