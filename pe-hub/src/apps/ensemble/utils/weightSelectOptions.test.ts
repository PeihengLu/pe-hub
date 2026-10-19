import { describe, expect, it } from 'vitest'
import { isSelectMenuDivider } from '@components/SelectMenu'
import { weightMenuEntries } from '@apps/ensemble/utils/weightSelectOptions'

describe('weightMenuEntries', () => {
  const weights = [
    { id: 'custom-a', source: 'trained', label: 'Custom A' },
    { id: 'vendor-b', source: 'vendor', label: 'Vendor B' },
    { id: 'custom-c', source: 'trained', label: 'Custom C' },
    { id: 'vendor-a', source: 'vendor', label: 'Vendor A' },
  ]

  it('places vendor weights above custom weights with one divider', () => {
    const entries = weightMenuEntries(weights, (weight) => ({
      value: weight.id,
      label: weight.label,
    }))
    const values = entries.map((entry) =>
      isSelectMenuDivider(entry) ? 'divider' : entry.value
    )
    expect(values).toEqual(['vendor-b', 'vendor-a', 'divider', 'custom-a', 'custom-c'])
  })

  it('omits the divider when every weight is from one source', () => {
    const vendorOnly = weightMenuEntries(
      weights.filter((weight) => weight.source === 'vendor'),
      (weight) => ({ value: weight.id, label: weight.label })
    )
    expect(vendorOnly.some(isSelectMenuDivider)).toBe(false)
    expect(vendorOnly).toHaveLength(2)
  })
})
