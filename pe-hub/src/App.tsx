import { useState } from 'react'
import HubNavbar, {
  type EnsemblePage,
  type HubSection,
} from '@components/HubNavbar'
import ServiceGate from '@components/ServiceGate'
import HomePage from '@/pages/HomePage'
import CatalogPage from '@apps/database/pages/CatalogPage'
import PredictionPage from '@apps/ensemble/pages/PredictionPage'
import TrainingPage from '@apps/ensemble/pages/TrainingPage'
import EnsembleToolPage from '@apps/ensemble/pages/EnsemblePage'
import DocumentationPage from '@apps/ensemble/pages/DocumentationPage'

function App() {
  const [section, setSection] = useState<HubSection>('home')
  const [ensemblePage, setEnsemblePage] = useState<EnsemblePage>('predict')

  const renderContent = () => {
    if (section === 'home') {
      return <HomePage />
    }

    if (section === 'database') {
      return (
        <ServiceGate serviceId="pe-db">
          <CatalogPage />
        </ServiceGate>
      )
    }

    if (section === 'ensemble') {
      return (
        <ServiceGate serviceId="pe-ensemble">
          {ensemblePage === 'predict' && <PredictionPage />}
          {ensemblePage === 'train' && <TrainingPage />}
          {ensemblePage === 'ensemble' && <EnsembleToolPage />}
          {ensemblePage === 'docs' && <DocumentationPage />}
        </ServiceGate>
      )
    }

    return <HomePage />
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <HubNavbar
        section={section}
        onSectionChange={setSection}
        ensemblePage={ensemblePage}
        onEnsemblePageChange={setEnsemblePage}
      />
      <main className="container mx-auto px-4 py-8">{renderContent()}</main>
    </div>
  )
}

export default App
