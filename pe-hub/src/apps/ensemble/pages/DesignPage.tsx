import Card from '@components/Card'

export default function DesignPage() {
  return (
    <div className="space-y-6">
      <Card title="pegRNA Design">
        <p className="text-slate-600">
          Design pegRNAs by specifying a target sequence with the intended edit marked using
          <code className="mx-1 rounded bg-slate-100 px-1.5 py-0.5 text-sm">(pre/after)</code>
          syntax at the edit site. This panel will support interactive sequence input and
          efficiency prediction for candidate designs.
        </p>
      </Card>

      <Card title="Coming soon">
        <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-slate-500">
          <p className="font-medium text-slate-700">Design workflow placeholder</p>
          <p className="mt-2 text-sm max-w-lg mx-auto">
            Target sequence editor, edit-site annotation, and model-backed scoring will be added
            here in a future release.
          </p>
        </div>
      </Card>
    </div>
  )
}
