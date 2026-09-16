import Card from '@components/Card'

export default function AddModelInstructions() {
  return (
    <Card title="How to prepare your plugin">
      <div className="text-sm text-slate-600 space-y-3">
        <p>
          Copy <code className="text-xs bg-slate-100 px-1 rounded">plugins/_template/</code>, fill in{' '}
          <code className="text-xs bg-slate-100 px-1 rounded">manifest.yaml</code>,{' '}
          <code className="text-xs bg-slate-100 px-1 rounded">convert.py</code>, and{' '}
          <code className="text-xs bg-slate-100 px-1 rounded">wrapper.py</code>, zip the folder, then
          upload below. After validation passes, activate to use the model in Train and Benchmark.
        </p>

        <ol className="list-decimal pl-5 space-y-1.5">
          <li>
            <strong className="text-slate-800">Upload</strong> — zip becomes{' '}
            <span className="text-amber-800">pending</span>
          </li>
          <li>
            <strong className="text-slate-800">Validate</strong> — harness checks manifest, convert,
            and train/eval smoke tests
          </li>
          <li>
            <strong className="text-slate-800">Activate</strong> — only when checks pass; model appears
            in Train / Benchmark
          </li>
        </ol>

        <ul className="list-disc pl-5 space-y-1 text-xs text-slate-500">
          <li>
            Name: lowercase slug <code className="bg-slate-100 px-1 rounded">[a-z0-9_]+</code>, not a
            built-in (deepprime / oped / pridict2)
          </li>
          <li>
            Wrapper must subclass <code className="bg-slate-100 px-1 rounded">BasePEModel</code> and
            implement train, evaluate, predict, and{' '}
            <code className="bg-slate-100 px-1 rounded">save_to_registry</code>
          </li>
          <li>
            <code className="bg-slate-100 px-1 rounded">convert()</code> keeps row count and index;
            produce every column listed in the manifest
          </li>
          <li>Python dependencies must already be installed in the service environment</li>
        </ul>

        <p className="text-xs text-slate-500">
          Full guide:{' '}
          <code className="bg-slate-100 px-1 rounded">plugins/README.md</code>
          {' · '}
          Reference:{' '}
          <code className="bg-slate-100 px-1 rounded">testdata/plugins/dummy_model/</code>
          {' · '}
          LLM prompt:{' '}
          <code className="bg-slate-100 px-1 rounded">docs/plugin-author-llm-prompt.md</code>
        </p>
      </div>
    </Card>
  )
}
