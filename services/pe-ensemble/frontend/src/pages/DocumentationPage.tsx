import Card from '../components/Card'
import { BookOpen } from 'lucide-react'

export default function DocumentationPage() {
  return (
    <div className="space-y-6">
      <Card title="Documentation">
        <div className="prose prose-sm max-w-none">
          <h2>API Reference</h2>
          <p>PE Ensemble provides a unified interface for Prime Editing prediction models.</p>

          <h3>Models Available</h3>
          <ul>
            <li><strong>DeepPrime</strong> - Deep learning model for PE efficiency prediction</li>
            <li><strong>PRIDICT2</strong> - Improved PSSM-based model</li>
            <li><strong>OPED</strong> - Optimized Prime Editor prediction</li>
          </ul>

          <h3>Prediction Endpoint</h3>
          <pre className="bg-slate-100 p-4 rounded overflow-x-auto">
            <code>POST /predict</code>
          </pre>

          <h3>Request Body</h3>
          <pre className="bg-slate-100 p-4 rounded overflow-x-auto text-xs">
            <code>{`{
  "model_name": "deepprime",
  "sequences": ["ATGC..."],
  "cell_type": "HEK293T"
}`}</code>
          </pre>
        </div>
      </Card>

      <Card>
        <div className="flex items-start gap-4">
          <BookOpen className="w-6 h-6 text-primary-600 flex-shrink-0 mt-1" />
          <div>
            <h3 className="font-semibold text-slate-900">Need Help?</h3>
            <p className="text-slate-600 mt-1">
              Check the API documentation or contact the development team for support.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
