export type CombineMethod =
  | 'mean'
  | 'weighted_mean'
  | 'median'
  | 'trimmed_mean'
  | 'rank_mean'
  | 'percentile_mean'
  | 'geometric_mean'
  | 'harmonic_mean'
  | 'min'
  | 'max'

export interface CombineMethodOption {
  id: CombineMethod
  label: string
  description: string
  needsMemberWeights?: boolean
  needsTrimCount?: boolean
}

export const COMBINE_METHOD_OPTIONS: CombineMethodOption[] = [
  {
    id: 'mean',
    label: 'Mean',
    description: 'Unweighted arithmetic average of member predictions.',
  },
  {
    id: 'weighted_mean',
    label: 'Weighted mean',
    description: 'Weighted average using per-member weights below.',
    needsMemberWeights: true,
  },
  {
    id: 'median',
    label: 'Median',
    description: 'Per-sample median across members (robust to outliers).',
  },
  {
    id: 'trimmed_mean',
    label: 'Trimmed mean',
    description: 'Drop lowest and highest member per sample (needs ≥3 members).',
    needsTrimCount: true,
  },
  {
    id: 'rank_mean',
    label: 'Rank mean',
    description: 'Average cross-sample ranks, mapped back to the prediction scale.',
  },
  {
    id: 'percentile_mean',
    label: 'Percentile mean',
    description: 'Average within-batch percentiles, then map back per member.',
  },
  {
    id: 'geometric_mean',
    label: 'Geometric mean',
    description: 'Geometric average (values clipped to a small epsilon).',
  },
  {
    id: 'harmonic_mean',
    label: 'Harmonic mean',
    description: 'Harmonic average (conservative; values clipped to epsilon).',
  },
  {
    id: 'min',
    label: 'Minimum',
    description: 'Per-sample minimum across members.',
  },
  {
    id: 'max',
    label: 'Maximum',
    description: 'Per-sample maximum across members.',
  },
]

export function combineMethodOption(id: CombineMethod): CombineMethodOption | undefined {
  return COMBINE_METHOD_OPTIONS.find((option) => option.id === id)
}
