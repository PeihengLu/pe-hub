import { useState } from 'react'
import { useQuery } from 'react-query'
import Card from '@components/Card'
import LoadingSpinner from '@components/LoadingSpinner'
import ErrorAlert from '@components/ErrorAlert'
import peDbApi from '@apps/database/services/peDbApi'
import { useCatalogFilterOptions } from '@/hooks/useCatalogFilterOptions'

type Tab = 'studies' | 'datasets' | 'datasheets' | 'statistics'

export default function CatalogPage() {
  const [tab, setTab] = useState<Tab>('studies')
  const [studyFilter, setStudyFilter] = useState('')
  const { optionsByAttribute, isLoading: catalogOptionsLoading } = useCatalogFilterOptions()
  const studyOptions = optionsByAttribute.study ?? []

  const studiesQuery = useQuery('pe-db-studies', () => peDbApi.listStudies(), {
    enabled: tab === 'studies',
    select: (r) => r.data,
  })

  const datasetsQuery = useQuery(
    ['pe-db-datasets', studyFilter],
    () => peDbApi.listDatasets(studyFilter || undefined),
    {
      enabled: tab === 'datasets',
      select: (r) => r.data,
    }
  )

  const datasheetsQuery = useQuery(
    ['pe-db-datasheets', studyFilter],
    () => peDbApi.listDatasheets(studyFilter || undefined),
    {
      enabled: tab === 'datasheets',
      select: (r) => r.data,
    }
  )

  const statsQuery = useQuery('pe-db-statistics', () => peDbApi.getStatistics(), {
    enabled: tab === 'statistics',
    select: (r) => r.data,
  })

  const activeQuery =
    tab === 'studies'
      ? studiesQuery
      : tab === 'datasets'
        ? datasetsQuery
        : tab === 'datasheets'
          ? datasheetsQuery
          : statsQuery

  const tabs: { id: Tab; label: string }[] = [
    { id: 'studies', label: 'Studies' },
    { id: 'datasets', label: 'Datasets' },
    { id: 'datasheets', label: 'Datasheets' },
    { id: 'statistics', label: 'Statistics' },
  ]

  return (
    <div className="space-y-6">
      <Card title="PE Database Catalog">
        <p className="text-slate-600 text-sm">
          Browse catalog metadata served by the PE Database API. Use the filter
          for datasets and datasheets when exploring a specific study.
        </p>
        {(tab === 'datasets' || tab === 'datasheets') && (
          <div className="mt-4">
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Study filter (optional)
            </label>
            <select
              value={studyFilter}
              onChange={(e) => setStudyFilter(e.target.value)}
              disabled={catalogOptionsLoading}
              className="w-full max-w-xs px-3 py-2 border border-slate-300 rounded-lg text-sm bg-white disabled:bg-slate-100"
            >
              <option value="">All studies</option>
              {studyOptions.map((study) => (
                <option key={study} value={study}>
                  {study}
                </option>
              ))}
            </select>
          </div>
        )}
      </Card>

      <div className="flex flex-wrap gap-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t.id
                ? 'bg-primary-600 text-white'
                : 'bg-white text-slate-700 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {activeQuery.isLoading && <LoadingSpinner message="Loading catalog…" />}

      {activeQuery.isError && (
        <ErrorAlert
          message={
            (activeQuery.error as { response?: { data?: { detail?: string } } })
              ?.response?.data?.detail || 'Failed to load catalog data'
          }
        />
      )}

      {tab === 'studies' && studiesQuery.data && (
        <DataTable
          headers={['ID', 'Name', 'Published', 'Authors']}
          rows={studiesQuery.data.map((s) => [
            String(s.id),
            s.name,
            s.publication_date ?? '—',
            s.authors ?? '—',
          ])}
        />
      )}

      {tab === 'datasets' && datasetsQuery.data && (
        <DataTable
          headers={['ID', 'Study', 'Name', 'Standardizable']}
          rows={datasetsQuery.data.map((d) => [
            String(d.id),
            d.study_name ?? '—',
            d.name,
            d.standardizable ? 'yes' : 'no',
          ])}
        />
      )}

      {tab === 'datasheets' && datasheetsQuery.data && (
        <DataTable
          headers={['ID', 'Study', 'Dataset', 'Cell line', 'PE system', 'Samples']}
          rows={datasheetsQuery.data.map((d) => [
            String(d.id),
            d.study_name ?? '—',
            d.dataset_name ?? '—',
            d.cell_line,
            d.pe_system,
            String(d.num_samples),
          ])}
        />
      )}

      {tab === 'statistics' && statsQuery.data && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <StatCard label="Total entries" value={statsQuery.data.total_entries} />
          <StatCard label="Studies" value={statsQuery.data.total_studies} />
          <Card title="Edits by type" className="sm:col-span-2">
            <DataTable
              headers={['Study', 'Edit type', 'Count']}
              rows={(statsQuery.data.edit_type ?? []).map((r) => [
                r.study,
                r.edit_type,
                String(r.count),
              ])}
            />
          </Card>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
      <p className="text-sm text-slate-500">{label}</p>
      <p className="text-3xl font-bold text-slate-900 mt-1">{value.toLocaleString()}</p>
    </div>
  )
}

function DataTable({
  headers,
  rows,
}: {
  headers: string[]
  rows: string[][]
}) {
  if (rows.length === 0) {
    return (
      <p className="text-slate-500 text-sm py-8 text-center bg-white rounded-xl border border-slate-200">
        No records found.
      </p>
    )
  }

  return (
    <div className="overflow-x-auto bg-white rounded-xl border border-slate-200 shadow-sm">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50">
            {headers.map((h) => (
              <th
                key={h}
                className="text-left px-4 py-3 font-semibold text-slate-700"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-slate-100 last:border-0">
              {row.map((cell, j) => (
                <td key={j} className="px-4 py-2.5 text-slate-800">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
