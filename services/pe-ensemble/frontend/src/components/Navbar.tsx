import { Dna } from 'lucide-react'

interface NavbarProps {
  currentPage: string
  onPageChange: (page: any) => void
}

export default function Navbar({ currentPage, onPageChange }: NavbarProps) {
  const navItems = [
    { id: 'predict', label: 'Predict' },
    { id: 'train', label: 'Train' },
    { id: 'ensemble', label: 'Ensemble' },
    { id: 'docs', label: 'Documentation' },
  ]

  return (
    <nav className="sticky top-0 z-50 bg-white shadow-sm border-b border-slate-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Dna className="w-8 h-8 text-primary-600" />
            <h1 className="text-2xl font-bold text-slate-900">PE Ensemble</h1>
          </div>

          <div className="flex items-center gap-8">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => onPageChange(item.id)}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  currentPage === item.id
                    ? 'bg-primary-600 text-white'
                    : 'text-slate-700 hover:bg-slate-100'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  )
}
