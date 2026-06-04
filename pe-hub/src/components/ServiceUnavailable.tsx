import { AlertCircle, ExternalLink, RefreshCw, Terminal } from 'lucide-react'
import {
  type ServiceId,
  SERVICES,
  serviceBaseUrl,
  serviceDocsUrl,
} from '@config/services'
import { useServiceHealth } from '@context/ServiceHealthProvider'

interface ServiceUnavailableProps {
  serviceId: ServiceId
}

export default function ServiceUnavailable({
  serviceId,
}: ServiceUnavailableProps) {
  const { health, refreshService } = useServiceHealth()
  const service = SERVICES[serviceId]
  const state = health[serviceId]
  const baseUrl = serviceBaseUrl(serviceId)

  return (
    <div className="w-full max-w-2xl mx-auto min-w-0">
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 sm:p-6 shadow-sm overflow-hidden">
        <div className="flex items-start gap-3 min-w-0">
          <AlertCircle className="w-6 h-6 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0 space-y-4">
            <div className="min-w-0">
              <h2 className="text-lg font-semibold text-amber-900">
                {service.name} is not reachable
              </h2>
              <p className="text-amber-800 text-sm mt-1 break-words">
                PE Hub expects the API at{' '}
                <code className="inline-block max-w-full bg-amber-100/80 px-1.5 py-0.5 rounded text-xs break-all">
                  {baseUrl}
                </code>
                . Start the backend, then retry the health check.
              </p>
              {state.error && (
                <p className="text-amber-700 text-sm mt-2 break-words">
                  Last check: {state.error}
                </p>
              )}
            </div>

            <div className="min-w-0">
              <h3 className="text-sm font-semibold text-amber-900 flex items-center gap-2">
                <Terminal className="w-4 h-4 flex-shrink-0" />
                How to start
              </h3>
              <ul className="mt-2 space-y-2 min-w-0">
                {service.startupCommands.map((cmd) => (
                  <li key={cmd} className="min-w-0">
                    <pre className="max-w-full text-xs leading-relaxed bg-slate-900 text-slate-100 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap break-words [overflow-wrap:anywhere]">
                      {cmd}
                    </pre>
                  </li>
                ))}
              </ul>
              <p className="text-xs text-amber-700 mt-2 break-words">
                From the repository root. Use{' '}
                <code className="bg-amber-100/80 px-1 rounded break-all">
                  ./start-all.sh --install
                </code>{' '}
                on first run to install dependencies.
              </p>
            </div>

            <div className="flex flex-wrap gap-3 min-w-0">
              <button
                type="button"
                onClick={() => refreshService(serviceId)}
                className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Retry connection
              </button>
              <a
                href={serviceDocsUrl(serviceId)}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 border border-amber-300 text-amber-900 rounded-lg text-sm font-medium hover:bg-amber-100/50 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open API docs
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
