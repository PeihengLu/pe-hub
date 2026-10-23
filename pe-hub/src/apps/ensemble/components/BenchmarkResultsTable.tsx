import SelectMenu from '@components/SelectMenu'
import {
  BENCHMARK_TABLE_METRICS,
  formatCellRuns,
  type BenchmarkResultsTableState,
  type BenchmarkTableMetric,
} from '@apps/ensemble/utils/benchmarkResultsTable'

interface BenchmarkResultsTableProps {
  table: BenchmarkResultsTableState
  displayMetric: BenchmarkTableMetric
  onDisplayMetricChange: (metric: BenchmarkTableMetric) => void
  onClear: () => void
}

export default function BenchmarkResultsTable({
  table,
  displayMetric,
  onDisplayMetricChange,
  onClear,
}: BenchmarkResultsTableProps) {
  const isEmpty = table.models.length === 0 || table.rows.length === 0

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-[10rem]">
          <label className="block text-sm font-medium text-ink-muted mb-1">
            Display metric
          </label>
          <SelectMenu
            value={displayMetric}
            onChange={(value) => onDisplayMetricChange(value as BenchmarkTableMetric)}
            aria-label="Display metric"
            options={BENCHMARK_TABLE_METRICS.map((metric) => ({
              value: metric,
              label: metric,
            }))}
          />
        </div>
        <button
          type="button"
          onClick={onClear}
          disabled={isEmpty && table.ingestedJobIds.length === 0}
          className="text-sm px-3 py-1.5 rounded-lg border border-line-strong text-ink-muted hover:bg-canvas disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Clear table
        </button>
      </div>

      {isEmpty ? (
        <p className="text-ink-faint py-6 text-center text-sm">
          Completed benchmark jobs will appear here as a model × datasheet matrix.
          Duplicate model/datasheet runs are appended as{' '}
          <code className="text-xs">original;new</code>.
        </p>
      ) : (
        <div className="overflow-x-auto bg-surface rounded-xl border border-line">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-line bg-canvas">
                <th className="text-left px-4 py-3 font-semibold text-ink-muted sticky left-0 bg-canvas">
                  Datasheet
                </th>
                {table.models.map((model) => (
                  <th
                    key={model}
                    className="text-left px-4 py-3 font-semibold text-ink-muted whitespace-nowrap"
                  >
                    {model}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row) => (
                <tr key={row} className="border-b border-line last:border-0">
                  <td className="px-4 py-2.5 text-ink sticky left-0 bg-surface max-w-xs break-words">
                    {row}
                  </td>
                  {table.models.map((model) => {
                    const cell = formatCellRuns(table.cells[row]?.[model], displayMetric)
                    return (
                      <td
                        key={`${row}:${model}`}
                        className="px-4 py-2.5 font-mono text-xs text-ink whitespace-nowrap"
                      >
                        {cell || '—'}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
