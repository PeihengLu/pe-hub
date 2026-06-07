import { useMemo, useState } from 'react'
import { useMutation, useQuery } from 'react-query'
import Card from '@components/Card'
import LoadingSpinner from '@components/LoadingSpinner'
import ErrorAlert from '@components/ErrorAlert'
import ExportFilterBuilder from '@apps/database/components/ExportFilterBuilder'
import {
  EXPORT_FORMATS,
  STATIC_FILTER_OPTIONS,
  buildFilterParams,
  type AttributeFilterRow,
  type ExportFormat,
  type FilterAttributeKey,
} from '@apps/database/config/exportAttributes'
import peDbApi from '@apps/database/services/peDbApi'
import { downloadCsv, mergedExportGroupsToCsv } from '@apps/database/utils/downloadCsv'

function uniqueSorted(values: string[]) {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b))
}

export default function ExportPage() {
  const [format, setFormat] = useState<ExportFormat>('std')
  const [filterRows, setFilterRows] = useState<AttributeFilterRow[]>([])

  const studiesQuery = useQuery('pe-db-export-studies', () => peDbApi.listStudies(), {
    select: (response) => response.data,
  })
  const datasetsQuery = useQuery('pe-db-export-datasets', () => peDbApi.listDatasets(), {
    select: (response) => response.data,
  })
  const datasheetsQuery = useQuery('pe-db-export-datasheets', () => peDbApi.listDatasheets(), {
    select: (response) => response.data,
  })
  const scaffoldsQuery = useQuery('pe-db-export-scaffolds', () => peDbApi.listScaffolds(), {
    select: (response) => response.data,
  })
  const statsQuery = useQuery('pe-db-export-statistics', () => peDbApi.getStatistics(), {
    select: (response) => response.data,
  })

  const optionsByAttribute = useMemo<Partial<Record<FilterAttributeKey, string[]>>>(() => {
    const studies = studiesQuery.data?.map((study) => study.name) ?? []
    const datasets = datasetsQuery.data?.map((dataset) => dataset.name) ?? []
    const cellLines = datasheetsQuery.data?.map((sheet) => sheet.cell_line) ?? []
    const peSystems = datasheetsQuery.data?.map((sheet) => sheet.pe_system) ?? []
    const scaffolds = scaffoldsQuery.data?.map((scaffold) => scaffold.name) ?? []
    const editLengths =
      statsQuery.data?.edit_length.map((row) => String(row.edit_length)) ?? []

    return {
      study: uniqueSorted(studies),
      dataset: uniqueSorted(datasets),
      cell_line: uniqueSorted(cellLines),
      pe_system: uniqueSorted(peSystems),
      scaffold_name: uniqueSorted(scaffolds),
      edit_length: uniqueSorted(editLengths),
      ...STATIC_FILTER_OPTIONS,
    }
  }, [studiesQuery.data, datasetsQuery.data, datasheetsQuery.data, scaffoldsQuery.data, statsQuery.data])

  const optionsLoading =
    studiesQuery.isLoading ||
    datasetsQuery.isLoading ||
    datasheetsQuery.isLoading ||
    scaffoldsQuery.isLoading ||
    statsQuery.isLoading

  const optionsError =
    studiesQuery.error ||
    datasetsQuery.error ||
    datasheetsQuery.error ||
    scaffoldsQuery.error ||
    statsQuery.error

  const exportMutation = useMutation(async () => {
    const filters = buildFilterParams(filterRows)
    const response = await peDbApi.exportFiltered(format, filters)
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

    const csv = mergedExportGroupsToCsv(result.groups)
    const filename = `pe-db-export-${format}.csv`
    downloadCsv(filename, csv)
  }

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
            onChange={setFilterRows}
          />
        )}
        {incompleteRows.length > 0 && (
          <p className="mt-3 text-sm text-amber-700">
            Each added attribute needs at least one value before export.
          </p>
        )}
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
            Download CSV
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
