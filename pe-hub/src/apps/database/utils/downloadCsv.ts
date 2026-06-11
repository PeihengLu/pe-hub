function escapeCsvCell(value: unknown): string {
  if (value === null || value === undefined) return ''
  const text = String(value)
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`
  }
  return text
}

export function recordsToCsv(
  records: Record<string, unknown>[],
  columns?: string[]
): string {
  if (records.length === 0) return ''
  const header = columns ?? Object.keys(records[0])
  const lines = [
    header.map(escapeCsvCell).join(','),
    ...records.map((record) =>
      header.map((column) => escapeCsvCell(record[column])).join(',')
    ),
  ]
  return lines.join('\n')
}

const MERGE_METADATA_COLUMNS = ['study', 'dataset', 'cell_line', 'pe_system'] as const

export interface MergeableExportGroup {
  study: string
  dataset: string
  cell_line: string
  pe_system: string
  columns: string[]
  records: Record<string, unknown>[]
}

export function mergeExportGroups(groups: MergeableExportGroup[]): {
  records: Record<string, unknown>[]
  columns: string[]
} {
  if (groups.length === 0) {
    return { records: [], columns: [] }
  }

  const dataColumns = [
    ...new Set(groups.flatMap((group) => group.columns)),
  ].filter((column) => !MERGE_METADATA_COLUMNS.includes(column as (typeof MERGE_METADATA_COLUMNS)[number]))

  const columns = [...MERGE_METADATA_COLUMNS, ...dataColumns]
  const records = groups.flatMap((group) =>
    group.records.map((record) => ({
      study: group.study,
      dataset: group.dataset,
      cell_line: group.cell_line,
      pe_system: group.pe_system,
      ...record,
    }))
  )

  return { records, columns }
}

export function mergedExportGroupsToCsv(groups: MergeableExportGroup[]): string {
  const { records, columns } = mergeExportGroups(groups)
  return recordsToCsv(records, columns)
}

export function downloadTextFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export function downloadCsv(filename: string, csv: string) {
  downloadTextFile(filename, csv, 'text/csv;charset=utf-8')
}

export function downloadJson(filename: string, value: unknown, indent = 2) {
  downloadTextFile(filename, JSON.stringify(value, null, indent), 'application/json;charset=utf-8')
}
