import {
  FILTER_ATTRIBUTES,
  type AttributeFilterRow,
  type FilterAttributeKey,
} from '@apps/database/config/exportAttributes'

interface ExportFilterBuilderProps {
  rows: AttributeFilterRow[]
  optionsByAttribute: Partial<Record<FilterAttributeKey, string[]>>
  onChange: (rows: AttributeFilterRow[]) => void
}

function nextRowId() {
  return `filter-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

export default function ExportFilterBuilder({
  rows,
  optionsByAttribute,
  onChange,
}: ExportFilterBuilderProps) {
  const usedAttributes = new Set(
    rows.map((row) => row.attribute).filter((value): value is FilterAttributeKey => value !== '')
  )

  const addRow = () => {
    onChange([...rows, { id: nextRowId(), attribute: '', values: [] }])
  }

  const updateRow = (id: string, patch: Partial<AttributeFilterRow>) => {
    onChange(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)))
  }

  const removeRow = (id: string) => {
    onChange(rows.filter((row) => row.id !== id))
  }

  const addValue = (id: string, value: string) => {
    const row = rows.find((item) => item.id === id)
    if (!row || !value || row.values.includes(value)) return
    updateRow(id, { values: [...row.values, value] })
  }

  const removeValue = (id: string, value: string) => {
    const row = rows.find((item) => item.id === id)
    if (!row) return
    updateRow(id, { values: row.values.filter((item) => item !== value) })
  }

  return (
    <div className="space-y-3">
      {rows.length === 0 && (
        <p className="text-sm text-slate-500">
          No filters yet. Export includes all standardizable data unless you add
          attributes below.
        </p>
      )}

      {rows.map((row) => {
        const availableAttributes = FILTER_ATTRIBUTES.filter(
          (attribute) =>
            attribute.key === row.attribute || !usedAttributes.has(attribute.key)
        )
        const optionList =
          row.attribute && optionsByAttribute[row.attribute]
            ? optionsByAttribute[row.attribute]!
            : []
        const remainingOptions = optionList.filter((option) => !row.values.includes(option))

        return (
          <div
            key={row.id}
            className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4 sm:flex-row sm:items-start"
          >
            <div className="min-w-[180px]">
              <label className="mb-1 block text-xs font-medium text-slate-600">
                Attribute
              </label>
              <select
                value={row.attribute}
                onChange={(event) => {
                  const attribute = event.target.value as FilterAttributeKey | ''
                  updateRow(row.id, { attribute, values: [] })
                }}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
              >
                <option value="">Select attribute…</option>
                {availableAttributes.map((attribute) => (
                  <option key={attribute.key} value={attribute.key}>
                    {attribute.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex-1">
              <label className="mb-1 block text-xs font-medium text-slate-600">
                Values
              </label>
              {row.attribute ? (
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-2">
                    {row.values.map((value) => (
                      <span
                        key={value}
                        className="inline-flex items-center gap-1 rounded-full bg-primary-100 px-3 py-1 text-sm text-primary-800"
                      >
                        {value}
                        <button
                          type="button"
                          onClick={() => removeValue(row.id, value)}
                          className="rounded-full px-1 text-primary-600 hover:bg-primary-200"
                          aria-label={`Remove ${value}`}
                        >
                          ×
                        </button>
                      </span>
                    ))}
                    {row.values.length === 0 && (
                      <span className="text-sm text-slate-400">No values selected</span>
                    )}
                  </div>
                  <select
                    value=""
                    onChange={(event) => {
                      addValue(row.id, event.target.value)
                      event.target.value = ''
                    }}
                    disabled={remainingOptions.length === 0}
                    className="w-full max-w-md rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm disabled:bg-slate-100"
                  >
                    <option value="">
                      {remainingOptions.length === 0
                        ? 'All values selected'
                        : 'Add value…'}
                    </option>
                    {remainingOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <p className="text-sm text-slate-400 py-2">Choose an attribute first.</p>
              )}
            </div>

            <button
              type="button"
              onClick={() => removeRow(row.id)}
              className="self-start rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-600 hover:bg-white"
            >
              Remove
            </button>
          </div>
        )
      })}

      <button
        type="button"
        onClick={addRow}
        disabled={usedAttributes.size >= FILTER_ATTRIBUTES.length}
        className="inline-flex items-center gap-2 rounded-lg border border-dashed border-primary-300 px-4 py-2 text-sm font-medium text-primary-700 hover:bg-primary-50 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-400"
      >
        <span className="text-lg leading-none">+</span>
        Add attribute
      </button>
    </div>
  )
}
