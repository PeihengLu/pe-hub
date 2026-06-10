import Card from '@components/Card'
import { BookOpen, ExternalLink } from 'lucide-react'
import { ENSEMBLE_API_URL } from '@config/services'

const docsUrl = `${ENSEMBLE_API_URL.replace(/\/$/, '')}/docs`

export default function DocumentationPage() {
  return (
    <div className="space-y-6">
      <Card title="PE Ensemble API">
        <div className="prose prose-sm max-w-none text-slate-700">
          <p>
            FastAPI service for training and evaluating prime editing efficiency
            models. Interactive OpenAPI docs:{' '}
            <a
              href={docsUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-primary-600 hover:text-primary-700"
            >
              {docsUrl}
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </p>

          <h3>Models</h3>
          <ul>
            <li>
              <strong>DeepPrime</strong> — CNN-GRU; select a registered weight set for
              evaluation (normalization stats come from the weights)
            </li>
            <li>
              <strong>PRIDICT2</strong> — attention LSTM with transfer learning
            </li>
            <li>
              <strong>OPED</strong> — transformer on pegRNA features
            </li>
          </ul>

          <h3>Data</h3>
          <p>
            Training and evaluation fetch model-format rows from PE Database via{' '}
            <code>GET /data/filter</code> (same filters as the Database Export
            page). Rows with <code>split=test</code> are used for evaluation.
          </p>

          <h3>Evaluate</h3>
          <pre className="bg-slate-100 p-4 rounded overflow-x-auto text-xs">
            <code>{`POST /evaluate
{
  "model_name": "deepprime",
  "weights": "DeepPrime_base",
  "study": "pridict1",
  "dataset": "library2",
  "cell_line": ["hek293t"],
  "pe_system": ["pe2max"],
  "split": {
    "split_strategy": "holdout_2",
    "train_pct": 0.8,
    "test_pct": 0.2
  }
}`}</code>
          </pre>

          <h3>Train</h3>
          <p>
            <code>POST /train</code> queues a job (returns <code>job_id</code>).
            Poll <code>GET /train/status/&#123;job_id&#125;</code> and stream logs
            with <code>GET /train/logs/&#123;job_id&#125;?offset=0</code>. Pass{' '}
            <code>device</code> (<code>auto</code>, <code>cuda:0</code>,{' '}
            <code>mps</code>, <code>cpu</code>) to target a specific accelerator.
          </p>

          <h3>Weights and devices</h3>
          <ul>
            <li>
              <code>GET /models/&#123;name&#125;/weights</code> — list registered
              checkpoints under <code>services/pe-ensemble/weights/</code>
            </li>
            <li>
              <code>GET /devices</code> — available compute devices
            </li>
            <li>
              <code>GET /train/devices</code> — per-device queue depth
            </li>
          </ul>

          <p className="text-sm text-slate-500">
            See <code>services/pe-ensemble/README.md</code> and{' '}
            <code>services/pe-ensemble/jobs/README.md</code> in the repository for
            CLI and cluster usage.
          </p>
        </div>
      </Card>

      <Card>
        <div className="flex items-start gap-4">
          <BookOpen className="w-6 h-6 text-primary-600 flex-shrink-0 mt-1" />
          <div>
            <h3 className="font-semibold text-slate-900">Repository docs</h3>
            <p className="text-slate-600 mt-1 text-sm">
              Root <code>README.md</code> covers the full stack. PE Database
              catalog and export APIs are documented at{' '}
              <code>services/pe-db/README.md</code>.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
