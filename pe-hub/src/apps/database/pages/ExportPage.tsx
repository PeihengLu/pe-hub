import { useState } from 'react'
import { useMutation } from 'react-query'
import Card from '@components/Card'
import LoadingSpinner from '@components/LoadingSpinner'
import ErrorAlert from '@components/ErrorAlert'
import ExportFilterBuilder from '@apps/database/components/ExportFilterBuilder'
import {
  EXPORT_FORMATS,
  SPLIT_STRATEGIES,
  buildFilterParams,
  buildSplitParams,
  type AttributeFilterRow,
  type ExportFormat,
  type SplitStrategy,
} from '@apps/database/config/exportAttributes'
import peDbApi from '@apps/database/services/peDbApi'
import {
  downloadCsv,
  mergedExportGroupsToCsv,
  recordsToCsv,
} from '@apps/database/utils/downloadCsv'
import { useCatalogFilterOptions } from '@/hooks/useCatalogFilterOptions'

type DownloadMode = 'merged' | 'separate'

export default function ExportPage() {
  const [format, setFormat] = useState<ExportFormat>('std')
  const [downloadMode, setDownloadMode] = useState<DownloadMode>('merged')
  const [filterRows, setFilterRows] = useState<AttributeFilterRow[]>([])
  const [splitStrategy, setSplitStrategy] = useState<SplitStrategy>('none')
  const [trainPct, setTrainPct] = useState('0.8')
  const [valPct, setValPct] = useState('0.1')
  const [testPct, setTestPct] = useState('0.2')
  const [cvFolds, setCvFolds] = useState('5')
  const [useOriginalFold, setUseOriginalFold] = useState(false)
  const [splitRandomState, setSplitRandomState] = useState('42')

  const { optionsByAttribute, getOptionsForRow, isLoading: optionsLoading, error: optionsError } =
    useCatalogFilterOptions(filterRows)

  const exportMutation = useMutation(async () => {
    const filters = buildFilterParams(filterRows)
    const split = buildSplitParams({
      strategy: splitStrategy,
      trainPct,
      valPct,
      testPct,
      cvFolds,
      useOriginalFold,
      randomState: splitRandomState,
      merge: downloadMode === 'merged',
    })
    const response = await peDbApi.exportFiltered(format, filters, split)
    return response.data
  })

  const incompleteRows = filterRows.filter(
    (row) => row.attribute !== '' && row.values.length === 0
  )
  const canExport = incompleteRows.length === 0

  const handleExport = () => {
    if (!canExport) return
    exportMutation.mutate()
  }

  const handleDownload = () => {
    const result = exportMutation.data
    if (!result || result.groups.length === 0) return

    if (downloadMode === 'merged') {
      const csv = mergedExportGroupsToCsv(result.groups)
      downloadCsv(`pe-db-export-${format}.csv`, csv)
      return
    }

    result.groups.forEach((group) => {
      const csv = recordsToCsv(group.records, group.columns)
      const filename = `${group.study}-${group.dataset}-${group.cell_line}-${group.pe_system}-${format}.csv`
      downloadCsv(filename, csv)
    })
  }

  const groupCount = exportMutation.data?.groups.length ?? 0
  const downloadLabel =
    downloadMode === 'merged'
      ? 'Download CSV'
      : groupCount > 1
        ? `Download CSV (${groupCount} files)`
        : 'Download CSV'

  const selectedFormat = EXPORT_FORMATS.find((item) => item.value === format)

  return (
    <div className="space-y-6">
      <Card title="Export Data">
        <p className="text-sm text-slate-600">
          Export standardizable PE editing data in a model-ready format. Only
          datasets flagged as fully standardizable are included; partially
          standardizable sets are skipped automatically.
        </p>
      </Card>

      <Card title="Output format">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {EXPORT_FORMATS.map((item) => (
            <label
              key={item.value}
              className={`cursor-pointer rounded-lg border p-4 transition-all ${
                format === item.value
                  ? 'border-primary-500 bg-primary-50 ring-1 ring-primary-500'
                  : 'border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="flex items-start gap-3">
                <input
                  type="radio"
                  name="export-format"
                  value={item.value}
                  checked={format === item.value}
                  onChange={() => setFormat(item.value)}
                  className="mt-1"
                />
                <div>
                  <p className="font-medium text-slate-900">{item.label}</p>
                  <p className="text-xs text-slate-500 mt-1">{item.description}</p>
                </div>
              </div>
            </label>
          ))}
        </div>
      </Card>

      <Card title="Filter attributes">
        {optionsLoading && <LoadingSpinner message="Loading filter options…" />}
        {!!optionsError && (
          <ErrorAlert message="Failed to load filter attribute options from the catalog." />
        )}
        {!optionsLoading && !optionsError && (
          <ExportFilterBuilder
            rows={filterRows}
            optionsByAttribute={optionsByAttribute}
            getOptionsForRow={getOptionsForRow}
            onChange={setFilterRows}
          />
        )}
        {incompleteRows.length > 0 && (
          <p className="mt-3 text-sm text-amber-700">
            Each added attribute needs at least one value before export.
          </p>
        )}
      </Card>

      <Card title="Split assignment">
        <p className="mb-4 text-sm text-slate-600">
          Required for formatted exports. Splits are group-aware on{' '}
          <code className="text-xs">group_id</code>. When downloading a merged CSV,
          datasheets are merged server-side with composite group keys before splitting.
        </p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {SPLIT_STRATEGIES.map((item) => (
            <label
              key={item.value}
              className={`cursor-pointer rounded-lg border p-4 transition-all ${
                splitStrategy === item.value
                  ? 'border-primary-500 bg-primary-50 ring-1 ring-primary-500'
                  : 'border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="flex items-start gap-3">
                <input
                  type="radio"
                  name="split-strategy"
                  value={item.value}
                  checked={splitStrategy === item.value}
                  onChange={() => setSplitStrategy(item.value)}
                  className="mt-1"
                />
                <div>
                  <p className="font-medium text-slate-900">{item.label}</p>
                  <p className="text-xs text-slate-500 mt-1">{item.description}</p>
                </div>
              </div>
            </label>
          ))}
        </div>

        {(splitStrategy === 'holdout_2' || splitStrategy === 'holdout_3') && (
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <label className="text-sm text-slate-700">
              Train %
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={trainPct}
                onChange={(e) => setTrainPct(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
              />
            </label>
            {splitStrategy === 'holdout_3' && (
              <label className="text-sm text-slate-700">
                Val %
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={valPct}
                  onChange={(e) => setValPct(e.target.value)}
                  className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
                />
              </label>
            )}
            <label className="text-sm text-slate-700">
              Test %
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={testPct}
                onChange={(e) => setTestPct(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
              />
            </label>
          </div>
        )}

        {splitStrategy === 'cv' && (
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <label className="text-sm text-slate-700">
              CV folds
              <input
                type="number"
                min={2}
                step={1}
                value={cvFolds}
                onChange={(e) => setCvFolds(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
              />
            </label>
            <label className="text-sm text-slate-700">
              Test % (optional holdout)
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={testPct}
                onChange={(e) => setTestPct(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
                placeholder="Leave empty for no holdout"
              />
            </label>
          </div>
        )}

        {splitStrategy !== 'none' && splitStrategy !== 'holdout_3' && (
          <label className="mt-4 inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={useOriginalFold}
              onChange={(e) => setUseOriginalFold(e.target.checked)}
            />
            Use author original_fold when available
          </label>
        )}

        {splitStrategy !== 'none' && (
          <label className="mt-4 block text-sm text-slate-700 max-w-xs">
            Random seed
            <input
              type="number"
              min={0}
              step={1}
              value={splitRandomState}
              onChange={(e) => setSplitRandomState(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
            />
          </label>
        )}
      </Card>

      <Card title="Download options">
        <div className="flex flex-wrap gap-4">
          <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="radio"
              name="download-mode"
              value="merged"
              checked={downloadMode === 'merged'}
              onChange={() => setDownloadMode('merged')}
            />
            Single merged CSV
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="radio"
              name="download-mode"
              value="separate"
              checked={downloadMode === 'separate'}
              onChange={() => setDownloadMode('separate')}
            />
            Separate file per datasheet
          </label>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          {downloadMode === 'merged'
            ? 'All matching datasheets are merged server-side before split assignment, then downloaded as one CSV.'
            : 'Each datasheet is split independently and downloaded as its own CSV (fold_0 is local to each file).'}
        </p>
      </Card>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={handleExport}
          disabled={!canExport || exportMutation.isLoading}
          className="rounded-lg bg-primary-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-primary-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {exportMutation.isLoading ? 'Exporting…' : 'Run export'}
        </button>
        {exportMutation.data && exportMutation.data.total_records > 0 && (
          <button
            type="button"
            onClick={handleDownload}
            className="rounded-lg border border-slate-300 bg-white px-5 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            {downloadLabel}
          </button>
        )}
      </div>

      {exportMutation.isError && (
        <ErrorAlert
          message={
            (exportMutation.error as { response?: { data?: { detail?: string } } })
              ?.response?.data?.detail || 'Export failed'
          }
        />
      )}

      {exportMutation.data && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <SummaryCard label="Format" value={selectedFormat?.label ?? format} />
            <SummaryCard
              label="Records"
              value={exportMutation.data.total_records.toLocaleString()}
            />
            <SummaryCard
              label="Datasheets"
              value={String(exportMutation.data.groups.length)}
            />
          </div>

          {exportMutation.data.skipped.length > 0 && (
            <Card title="Skipped datasheets">
              <p className="mb-3 text-sm text-slate-600">
                These datasheets matched your filters but could not be exported in
                the requested format.
              </p>
              <SkippedTable rows={exportMutation.data.skipped} />
            </Card>
          )}

          {exportMutation.data.groups.length > 0 && (
            <Card title="Preview">
              <PreviewTable groups={exportMutation.data.groups} />
            </Card>
          )}

          {exportMutation.data.total_records === 0 && (
            <p className="text-sm text-slate-500 text-center py-6 bg-white rounded-xl border border-slate-200">
              No records matched your filters in a standardizable dataset.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-sm text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-bold text-slate-900">{value}</p>
    </div>
  )
}

function SkippedTable({
  rows,
}: {
  rows: { study: string; dataset: string; cell_line: string; pe_system: string; reason: string }[]
}) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50">
            {['Study', 'Dataset', 'Cell line', 'PE system', 'Reason'].map((header) => (
              <th key={header} className="px-4 py-2 text-left font-semibold text-slate-700">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-b border-slate-100 last:border-0">
              <td className="px-4 py-2">{row.study}</td>
              <td className="px-4 py-2">{row.dataset}</td>
              <td className="px-4 py-2">{row.cell_line}</td>
              <td className="px-4 py-2">{row.pe_system}</td>
              <td className="px-4 py-2 text-slate-600">{row.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PreviewTable({
  groups,
}: {
  groups: {
    study: string
    dataset: string
    cell_line: string
    pe_system: string
    num_records: number
    columns: string[]
    records: Record<string, unknown>[]
  }[]
}) {
  const firstGroup = groups[0]
  const previewRows = firstGroup.records.slice(0, 5)
  const headers = firstGroup.columns.slice(0, 8)

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">
        Showing up to 5 rows from{' '}
        <span className="font-medium">
          {firstGroup.study}/{firstGroup.dataset}/{firstGroup.cell_line}/
          {firstGroup.pe_system}
        </span>
        {groups.length > 1 && ` (+${groups.length - 1} more datasheet groups)`}
      </p>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50">
              {headers.map((header) => (
                <th key={header} className="px-4 py-2 text-left font-semibold text-slate-700">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {previewRows.map((record, rowIndex) => (
              <tr key={rowIndex} className="border-b border-slate-100 last:border-0">
                {headers.map((header) => (
                  <td key={header} className="px-4 py-2 text-slate-800">
                    {String(record[header] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
