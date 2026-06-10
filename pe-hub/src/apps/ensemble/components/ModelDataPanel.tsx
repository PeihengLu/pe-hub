import { useMutation } from 'react-query'
import Card from '@components/Card'
import LoadingSpinner from '@components/LoadingSpinner'
import ErrorAlert from '@components/ErrorAlert'
import SplitAssignmentPanel from '@components/SplitAssignmentPanel'
import ExportFilterBuilder from '@apps/database/components/ExportFilterBuilder'
import {
  EXPORT_FORMATS,
  buildFilterParams,
  type AttributeFilterRow,
  type SplitStrategy,
} from '@apps/database/config/exportAttributes'
import peDbApi, { type ExportResponse } from '@apps/database/services/peDbApi'
import { exportFormatForModel } from '@apps/ensemble/config/modelFormats'
import { buildTrainingSplitParams } from '@apps/ensemble/utils/trainingRequest'
import { useCatalogFilterOptions } from '@/hooks/useCatalogFilterOptions'

export type ModelDataPanelMode = 'train' | 'benchmark'

const MODE_COPY: Record<
  ModelDataPanelMode,
  {
    title: string
    intro: string
    splitDescription: string
    singleLabel: string
    batchLabel: string
    singleHelp: string
    batchHelp: string
    previewError: string
    incompleteHint: string
  }
> = {
  train: {
    title: 'Training data',
    intro:
      'Select PE Database records using the same attribute filters as Export. Output format is fixed by the model you train.',
    splitDescription:
      'Splits are group-aware on group_id. For a single merged training run, datasheets are merged server-side before splitting.',
    singleLabel: 'Single merged training run',
    batchLabel: 'Batch training (one job per datasheet)',
    singleHelp:
      'All matching datasheets are merged server-side, split once, and trained together in one job.',
    batchHelp:
      'Each matching datasheet is split independently and queued as its own training job.',
    previewError: 'Failed to preview training data',
    incompleteHint: 'Each added attribute needs at least one value before preview or training.',
  },
  benchmark: {
    title: 'Benchmark data',
    intro:
      'Select held-out test data from the PE Database catalog. Only rows assigned to the test split are evaluated.',
    splitDescription:
      'Split assignment determines which rows count as test data. Datasheets can be merged or benchmarked separately. DeepPrime uses original_fold -1 for held-out test; PRIDICT2 uses 0–4 (set Designate test fold to match run_N weights). Uncheck "Use author original_fold" for synthetic holdout splits.',
    singleLabel: 'Single merged benchmark',
    batchLabel: 'Batch benchmark (one job per datasheet)',
    singleHelp:
      'All matching datasheets are merged server-side, split once, and evaluated together in one job.',
    batchHelp:
      'Each matching datasheet is split independently and queued as its own benchmark job.',
    previewError: 'Failed to preview benchmark data',
    incompleteHint: 'Each added attribute needs at least one value before preview or benchmark.',
  },
}

interface ModelDataPanelProps {
  mode: ModelDataPanelMode
  modelName: string
  filterRows: AttributeFilterRow[]
  onFilterRowsChange: (rows: AttributeFilterRow[]) => void
  splitStrategy: SplitStrategy
  onSplitStrategyChange: (value: SplitStrategy) => void
  trainPct: string
  onTrainPctChange: (value: string) => void
  valPct: string
  onValPctChange: (value: string) => void
  testPct: string
  onTestPctChange: (value: string) => void
  cvFolds: string
  onCvFoldsChange: (value: string) => void
  useOriginalFold: boolean
  onUseOriginalFoldChange: (value: boolean) => void
  originalFoldTestValue: string
  onOriginalFoldTestValueChange: (value: string) => void
  splitRandomState: string
  onSplitRandomStateChange: (value: string) => void
  batchMode: boolean
  onBatchModeChange: (value: boolean) => void
  previewData: ExportResponse | undefined
  onPreviewDataChange: (data: ExportResponse | undefined) => void
}

export default function ModelDataPanel({
  mode,
  modelName,
  filterRows,
  onFilterRowsChange,
  splitStrategy,
  onSplitStrategyChange,
  trainPct,
  onTrainPctChange,
  valPct,
  onValPctChange,
  testPct,
  onTestPctChange,
  cvFolds,
  onCvFoldsChange,
  useOriginalFold,
  onUseOriginalFoldChange,
  originalFoldTestValue,
  onOriginalFoldTestValueChange,
  splitRandomState,
  onSplitRandomStateChange,
  batchMode,
  onBatchModeChange,
  previewData,
  onPreviewDataChange,
}: ModelDataPanelProps) {
  const copy = MODE_COPY[mode]
  const { optionsByAttribute, getOptionsForRow, isLoading, error } =
    useCatalogFilterOptions(filterRows)
  const exportFormat = exportFormatForModel(modelName)
  const formatMeta = EXPORT_FORMATS.find((item) => item.value === exportFormat)

  const incompleteRows = filterRows.filter(
    (row) => row.attribute !== '' && row.values.length === 0
  )
  const canPreview = incompleteRows.length === 0

  const previewMutation = useMutation(async () => {
    const filters = buildFilterParams(filterRows)
    const split = buildTrainingSplitParams({
      strategy: splitStrategy,
      trainPct,
      valPct,
      testPct,
      cvFolds,
      useOriginalFold,
      originalFoldTestValue,
      randomState: splitRandomState,
      batchTraining: batchMode,
    })
    const response = await peDbApi.previewFiltered(filters, split)
    return response.data
  }, {
    onSuccess: (data) => onPreviewDataChange(data),
  })

  return (
    <Card title={copy.title}>
      <p className="text-sm text-slate-600 mb-4">{copy.intro}</p>

      <div className="mb-6 rounded-lg border border-slate-200 bg-slate-50 p-4">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Model format</p>
        <p className="mt-1 font-medium text-slate-900">{formatMeta?.label ?? exportFormat}</p>
        <p className="mt-1 text-sm text-slate-600">{formatMeta?.description}</p>
      </div>

      <h3 className="text-sm font-semibold text-slate-900 mb-3">Filter attributes</h3>
      {isLoading && <LoadingSpinner message="Loading filter options…" />}
      {error ? (
        <ErrorAlert message="Failed to load filter attribute options from the catalog." />
      ) : null}
      {!isLoading && !error && (
        <ExportFilterBuilder
          rows={filterRows}
          optionsByAttribute={optionsByAttribute}
          getOptionsForRow={getOptionsForRow}
          onChange={onFilterRowsChange}
        />
      )}
      {incompleteRows.length > 0 && (
        <p className="mt-3 text-sm text-amber-700">{copy.incompleteHint}</p>
      )}

      <div className="mt-8">
        <h3 className="text-sm font-semibold text-slate-900 mb-3">Split assignment</h3>
        <SplitAssignmentPanel
          splitStrategy={splitStrategy}
          onSplitStrategyChange={onSplitStrategyChange}
          trainPct={trainPct}
          onTrainPctChange={onTrainPctChange}
          valPct={valPct}
          onValPctChange={onValPctChange}
          testPct={testPct}
          onTestPctChange={onTestPctChange}
          cvFolds={cvFolds}
          onCvFoldsChange={onCvFoldsChange}
          useOriginalFold={useOriginalFold}
          onUseOriginalFoldChange={onUseOriginalFoldChange}
          originalFoldTestValue={originalFoldTestValue}
          onOriginalFoldTestValueChange={onOriginalFoldTestValueChange}
          splitRandomState={splitRandomState}
          onSplitRandomStateChange={onSplitRandomStateChange}
          excludeNone
          description={copy.splitDescription}
        />
      </div>

      <div className="mt-8">
        <h3 className="text-sm font-semibold text-slate-900 mb-3">Datasheet handling</h3>
        <div className="flex flex-wrap gap-4">
          <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="radio"
              name={`${mode}-batch-mode`}
              checked={!batchMode}
              onChange={() => onBatchModeChange(false)}
            />
            {copy.singleLabel}
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="radio"
              name={`${mode}-batch-mode`}
              checked={batchMode}
              onChange={() => onBatchModeChange(true)}
            />
            {copy.batchLabel}
          </label>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          {batchMode ? copy.batchHelp : copy.singleHelp}
        </p>
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => previewMutation.mutate()}
          disabled={!canPreview || previewMutation.isLoading}
          className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:bg-slate-100"
        >
          {previewMutation.isLoading ? 'Loading preview…' : 'Preview data'}
        </button>
      </div>

      {previewMutation.isError && (
        <div className="mt-4">
          <ErrorAlert
            message={
              (previewMutation.error as { response?: { data?: { detail?: string } } })
                ?.response?.data?.detail || copy.previewError
            }
          />
        </div>
      )}

      {(previewData || previewMutation.data) && (
        <>
          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
            <SummaryCard
              label="Records"
              value={String((previewData ?? previewMutation.data)?.total_records ?? 0)}
            />
            <SummaryCard
              label="Datasheets"
              value={String((previewData ?? previewMutation.data)?.groups.length ?? 0)}
            />
            <SummaryCard
              label="Mode"
              value={batchMode ? 'Batch jobs' : 'Merged run'}
            />
          </div>

          {(previewData ?? previewMutation.data)!.skipped.length > 0 && (
            <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-4">
              <p className="text-sm font-medium text-amber-900">
                {((previewData ?? previewMutation.data)?.total_records ?? 0) === 0
                  ? 'No records loaded — all matching datasheets were skipped'
                  : `${(previewData ?? previewMutation.data)!.skipped.length} datasheet(s) skipped`}
              </p>
              <p className="mt-1 text-xs text-amber-800">
                {(previewData ?? previewMutation.data)!.skipped.some((row) =>
                  row.reason.includes('original_fold')
                )
                  ? 'DeepPrime pegRNAs sharing a protospacer can carry different author fold ids. Uncheck "Use author original_fold" to use synthetic holdout splits instead.'
                  : 'See reasons below. Export-only datasets (e.g. deepprime-off) are never converted.'}
              </p>
              <SkippedDatasheetsTable rows={(previewData ?? previewMutation.data)!.skipped} />
            </div>
          )}
        </>
      )}
    </Card>
  )
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-xl font-semibold text-slate-900">{value}</p>
    </div>
  )
}

function SkippedDatasheetsTable({
  rows,
}: {
  rows: { study: string; dataset: string; cell_line: string; pe_system: string; reason: string }[]
}) {
  return (
    <div className="mt-3 overflow-x-auto">
      <table className="min-w-full text-xs">
        <thead>
          <tr className="border-b border-amber-200">
            {['Study', 'Dataset', 'Cell line', 'PE system', 'Reason'].map((header) => (
              <th key={header} className="px-2 py-1 text-left font-semibold text-amber-900">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-b border-amber-100">
              <td className="px-2 py-1 text-amber-950">{row.study}</td>
              <td className="px-2 py-1 text-amber-950">{row.dataset}</td>
              <td className="px-2 py-1 text-amber-950">{row.cell_line}</td>
              <td className="px-2 py-1 text-amber-950">{row.pe_system}</td>
              <td className="px-2 py-1 text-amber-950">{row.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
