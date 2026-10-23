import { Database, FlaskConical, ExternalLink } from 'lucide-react'
import Card from '@components/Card'
import { SERVICES, serviceBaseUrl, serviceDocsUrl } from '@config/services'
import { useServiceHealth } from '@context/ServiceHealthProvider'

export default function HomePage() {
  const { health } = useServiceHealth()

  const cards = [
    {
      id: 'pe-db' as const,
      icon: Database,
      title: SERVICES['pe-db'].name,
      description: SERVICES['pe-db'].description,
    },
    {
      id: 'pe-ensemble' as const,
      icon: FlaskConical,
      title: SERVICES['pe-ensemble'].name,
      description: SERVICES['pe-ensemble'].description,
    },
  ]

  return (
    <div className="space-y-6">
      <Card title="Welcome to PE Hub">
        <p className="text-ink-muted">
          A single interface for the PE Database catalog and the PE Ensemble
          prediction stack. Each section talks to its own FastAPI backend; status
          indicators in the nav show whether each service is up.
        </p>
        <p className="text-ink-muted mt-3 text-sm">
          Quick start from the repo root:{' '}
          <code className="bg-surface-raised px-1.5 py-0.5 rounded text-xs">
            ./scripts/start-all.sh --install
          </code>
        </p>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {cards.map(({ id, icon: Icon, title, description }) => {
          const state = health[id]
          const up = state.status === 'up'
          return (
            <div
              key={id}
              className="bg-surface rounded-xl shadow-sm border border-line p-6"
            >
              <div className="flex items-start justify-between gap-4">
                <Icon className="w-10 h-10 text-primary-600 flex-shrink-0" />
                <span
                  className={`text-xs font-medium px-2 py-1 rounded-full ${
                    up
                      ? 'bg-emerald-950/50 text-emerald-300'
                      : state.status === 'checking'
                        ? 'bg-amber-950/50 text-amber-200'
                        : 'bg-red-950/50 text-red-300'
                  }`}
                >
                  {up ? 'Online' : state.status === 'checking' ? 'Checking…' : 'Offline'}
                </span>
              </div>
              <h3 className="text-lg font-semibold text-ink mt-4">{title}</h3>
              <p className="text-ink-muted text-sm mt-2">{description}</p>
              <p className="text-xs text-ink-faint mt-3 font-mono">
                {serviceBaseUrl(id)}
              </p>
              <a
                href={serviceDocsUrl(id)}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700 mt-4 font-medium"
              >
                API documentation
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          )
        })}
      </div>
    </div>
  )
}
