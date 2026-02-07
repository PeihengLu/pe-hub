import { useState } from 'react'
import { useQuery, useMutation } from 'react-query'
import { Play } from 'lucide-react'
import Card from '../components/Card'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorAlert from '../components/ErrorAlert'
import api from '../services/api'

export default function PredictionPage() {
  const [modelName, setModelName] = useState('deepprime')
  const [sequences, setSequences] = useState('')
  const [cellType, setCellType] = useState('')
  const [error, setError] = useState<string | null>(null)

  const { data: models, isLoading: modelsLoading } = useQuery(
    'models',
    () => api.listModels(),
    {
      select: (response) => response.data,
    }
  )

  const predictMutation = useMutation(
    () => api.predict({
      model_name: modelName,
      sequences: sequences.split('\n').filter(s => s.trim()),
      cell_type: cellType || undefined,
    }),
    {
      onSuccess: (response) => {
        console.log('Predictions:', response.data)
      },
      onError: (error: any) => {
        setError(error.response?.data?.detail || 'Prediction failed')
      },
    }
  )

  const handlePredict = () => {
    if (!sequences.trim()) {
      setError('Please enter at least one sequence')
      return
    }
    setError(null)
    predictMutation.mutate()
  }

  if (modelsLoading) return <LoadingSpinner message="Loading models..." />

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Introduction Panel */}
      <Card className="lg:col-span-3" title="Prime Editing Prediction">
        <p className="text-slate-600">
          Use the interface below to submit DNA sequences for Prime Editing efficiency prediction using a pretrained model. Enter one sequence per line. Select the model, cell type and PE system as needed, then click "Predict" to see the results. If a matched model weight is not found, the default weights from their original study will be used.
        </p>
      </Card>

      {/* Input Panel */}
      <Card className="lg:col-span-1" title="Prediction Input">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Model
            </label>
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {models?.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name} - {model.description}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Cell Type (Optional)
            </label>
            <input
              type="text"
              value={cellType}
              onChange={(e) => setCellType(e.target.value)}
              placeholder="e.g., HEK293T, A549"
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              DNA Sequences
            </label>
            <textarea
              value={sequences}
              onChange={(e) => setSequences(e.target.value)}
              placeholder="Enter DNA sequences (one per line)"
              rows={6}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 font-mono text-sm"
            />
          </div>

          <button
            onClick={handlePredict}
            disabled={predictMutation.isLoading}
            className="w-full bg-primary-600 hover:bg-primary-700 disabled:bg-slate-400 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition"
          >
            <Play className="w-4 h-4" />
            {predictMutation.isLoading ? 'Predicting...' : 'Predict'}
          </button>

          {error && (
            <ErrorAlert message={error} onDismiss={() => setError(null)} />
          )}
        </div>
      </Card>

      {/* Results Panel */}
      <Card className="lg:col-span-2" title="Prediction Results">
        {predictMutation.isLoading ? (
          <LoadingSpinner message="Running predictions..." />
        ) : predictMutation.data ? (
          <div className="space-y-4">
            <div className="bg-primary-50 border border-primary-200 rounded-lg p-4">
              <p className="text-sm text-slate-600">Model: {predictMutation.data.data.model}</p>
              <p className="text-sm text-slate-600 mt-1">
                Timestamp: {new Date(predictMutation.data.data.timestamp).toLocaleString()}
              </p>
            </div>

            <div>
              <h3 className="font-semibold text-slate-900 mb-3">Scores</h3>
              <div className="space-y-2">
                {predictMutation.data.data.predictions.map((pred, idx) => (
                  <div key={idx} className="flex items-center gap-4">
                    <span className="text-sm text-slate-600 w-24">Sequence {idx + 1}</span>
                    <div className="flex-1 bg-slate-200 rounded-full h-2">
                      <div
                        className="bg-primary-500 h-2 rounded-full"
                        style={{ width: `${Math.min(pred * 100, 100)}%` }}
                      />
                    </div>
                    <span className="text-sm font-semibold text-slate-900 w-16 text-right">
                      {(pred * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center text-slate-500 py-12">
            <p>Submit a prediction to see results here</p>
          </div>
        )}
      </Card>

      {/* Execution Logs Read from Backend */}
      <Card className="lg:col-span-3" title="Execution Logs">
        <div className="bg-slate-100 p-4 rounded-lg h-48 overflow-y-auto font-mono text-sm text-slate-700">
          {predictMutation.isLoading && <p>Running prediction...</p>}
          {predictMutation.data &&(
            <pre>{JSON.stringify(predictMutation.data.data.logs, null, 2)}</pre>
          ) }
          {!predictMutation.isLoading && !predictMutation.data && (
            <p>No logs available. Start execution to see logs here.</p>
          )}
        </div>
      </Card>
    </div>
  )
}
