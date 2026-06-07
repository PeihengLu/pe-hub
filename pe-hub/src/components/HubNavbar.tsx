import { Dna, Circle } from 'lucide-react'
import clsx from 'clsx'
import { SERVICES, type ServiceId } from '@config/services'
import { useServiceHealth } from '@context/ServiceHealthProvider'

export type HubSection = 'home' | 'database' | 'ensemble'
export type DatabasePage = 'catalog' | 'export'
export type EnsemblePage = 'predict' | 'train' | 'ensemble' | 'docs'

interface HubNavbarProps {
  section: HubSection
  onSectionChange: (section: HubSection) => void
  databasePage: DatabasePage
  onDatabasePageChange: (page: DatabasePage) => void
  ensemblePage: EnsemblePage
  onEnsemblePageChange: (page: EnsemblePage) => void
}

function StatusDot({ serviceId }: { serviceId: ServiceId }) {
  const { health } = useServiceHealth()
  const status = health[serviceId].status

  return (
    <Circle
      className={clsx(
        'w-2 h-2 fill-current',
        status === 'up' && 'text-emerald-500',
        status === 'down' && 'text-red-500',
        status === 'checking' && 'text-amber-400 animate-pulse'
      )}
      aria-hidden
    />
  )
}

export default function HubNavbar({
  section,
  onSectionChange,
  databasePage,
  onDatabasePageChange,
  ensemblePage,
  onEnsemblePageChange,
}: HubNavbarProps) {
  const sections: { id: HubSection; label: string; serviceId?: ServiceId }[] = [
    { id: 'home', label: 'Home' },
    { id: 'database', label: 'Database', serviceId: 'pe-db' },
    { id: 'ensemble', label: 'Ensemble', serviceId: 'pe-ensemble' },
  ]

  const databaseNav: { id: DatabasePage; label: string }[] = [
    { id: 'catalog', label: 'Catalog' },
    { id: 'export', label: 'Export' },
  ]

  const ensembleNav: { id: EnsemblePage; label: string }[] = [
    { id: 'predict', label: 'Predict' },
    { id: 'train', label: 'Train' },
    { id: 'ensemble', label: 'Ensemble' },
    { id: 'docs', label: 'Docs' },
  ]

  return (
    <nav className="sticky top-0 z-50 bg-white shadow-sm border-b border-slate-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <button
            type="button"
            onClick={() => onSectionChange('home')}
            className="flex items-center gap-3 text-left"
          >
            <Dna className="w-8 h-8 text-primary-600" />
            <div>
              <h1 className="text-2xl font-bold text-slate-900">PE Hub</h1>
              <p className="text-xs text-slate-500">
                Prime editing data &amp; models
              </p>
            </div>
          </button>

          <div className="flex flex-wrap items-center gap-2">
            {sections.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => onSectionChange(item.id)}
                className={clsx(
                  'inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all',
                  section === item.id
                    ? 'bg-primary-600 text-white'
                    : 'text-slate-700 hover:bg-slate-100'
                )}
              >
                {item.serviceId && <StatusDot serviceId={item.serviceId} />}
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {section === 'database' && (
          <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-slate-100">
            <span className="text-xs font-medium text-slate-500 self-center mr-2">
              {SERVICES['pe-db'].shortName}
            </span>
            {databaseNav.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => onDatabasePageChange(item.id)}
                className={clsx(
                  'px-3 py-1.5 rounded-md text-sm font-medium transition-all',
                  databasePage === item.id
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-600 hover:bg-slate-100'
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}

        {section === 'ensemble' && (
          <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-slate-100">
            <span className="text-xs font-medium text-slate-500 self-center mr-2">
              {SERVICES['pe-ensemble'].shortName}
            </span>
            {ensembleNav.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => onEnsemblePageChange(item.id)}
                className={clsx(
                  'px-3 py-1.5 rounded-md text-sm font-medium transition-all',
                  ensemblePage === item.id
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-600 hover:bg-slate-100'
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </nav>
  )
}
