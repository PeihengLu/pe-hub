import type { SelectMenuEntry, SelectMenuOption } from '@components/SelectMenu'

export function partitionWeightsBySource<T extends { source?: string }>(
  weights: readonly T[] | undefined
): { vendor: T[]; custom: T[] } {
  const vendor: T[] = []
  const custom: T[] = []
  for (const weight of weights ?? []) {
    if (weight.source === 'vendor') vendor.push(weight)
    else custom.push(weight)
  }
  return { vendor, custom }
}

/** Vendor weights, then a divider, then custom weights. No divider when either side is empty. */
export function weightMenuEntries<T extends { source?: string }>(
  weights: readonly T[] | undefined,
  toOption: (weight: T) => SelectMenuOption
): SelectMenuEntry[] {
  const { vendor, custom } = partitionWeightsBySource(weights)
  const entries: SelectMenuEntry[] = vendor.map(toOption)
  if (vendor.length > 0 && custom.length > 0) {
    entries.push({ type: 'divider' })
  }
  entries.push(...custom.map(toOption))
  return entries
}
