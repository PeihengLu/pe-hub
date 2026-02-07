import { useState } from 'react'
import Navbar from './components/Navbar'
import PredictionPage from './pages/PredictionPage'
import TrainingPage from './pages/TrainingPage'
import EnsemblePage from './pages/EnsemblePage'
import DocumentationPage from './pages/DocumentationPage'

type Page = 'predict' | 'train' | 'ensemble' | 'docs'

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('predict')

  const renderPage = () => {
    switch (currentPage) {
      case 'predict':
        return <PredictionPage />
      case 'train':
        return <TrainingPage />
      case 'ensemble':
        return <EnsemblePage />
      case 'docs':
        return <DocumentationPage />
      default:
        return <PredictionPage />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <Navbar currentPage={currentPage} onPageChange={setCurrentPage} />
      <main className="container mx-auto px-4 py-8">
        {renderPage()}
      </main>
    </div>
  )
}

export default App
