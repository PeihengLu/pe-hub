import {
  DESIGN_RULESET_CHOICES,
  designRulesetFromFilterRows,
  setDesignRulesetOnFilterRows,
  type AttributeFilterRow,
} from '@apps/database/config/exportAttributes'

interface DesignRulesetPanelProps {
  rows: AttributeFilterRow[]
  onChange: (rows: AttributeFilterRow[]) => void
  name?: string
}

export default function DesignRulesetPanel({
  rows,
  onChange,
  name = 'design-ruleset',
}: DesignRulesetPanelProps) {
  const selected = designRulesetFromFilterRows(rows)

  return (
    <div>
      <p className="mb-3 text-sm text-ink-muted">
        Keep only pegRNAs that match practitioner design heuristics (Hsu / OptiPrime
        argued that published libraries include geometries a designer would not order).
      </p>
      <div className="grid gap-3 sm:grid-cols-3">
        {DESIGN_RULESET_CHOICES.map((item) => (
          <label
            key={item.value || 'none'}
            className={`cursor-pointer rounded-lg border p-4 transition-all ${
              selected === item.value
                ? 'border-primary-500 bg-primary-50 ring-1 ring-primary-500'
                : 'border-line hover:border-line-strong'
            }`}
          >
            <div className="flex items-start gap-3">
              <input
                type="radio"
                name={name}
                value={item.value}
                checked={selected === item.value}
                onChange={() => onChange(setDesignRulesetOnFilterRows(rows, item.value))}
                className="mt-1"
              />
              <div>
                <p className="font-medium text-ink">{item.label}</p>
                <p className="mt-1 text-xs text-ink-faint">{item.description}</p>
              </div>
            </div>
          </label>
        ))}
      </div>
    </div>
  )
}
