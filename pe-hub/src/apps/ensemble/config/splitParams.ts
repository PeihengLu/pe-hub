export type SplitStrategy = 'none' | 'holdout_2' | 'holdout_3' | 'cv'

export interface SplitExportParams {
  split_strategy: SplitStrategy
  train_pct?: number
  val_pct?: number
  test_pct?: number
  cv_folds?: number
  use_original_fold?: boolean
  original_fold_test_value?: number
  split_random_state?: number
  merge?: boolean
}

export const DEFAULT_EVAL_SPLIT: SplitExportParams = {
  split_strategy: 'holdout_2',
  train_pct: 0.8,
  test_pct: 0.2,
  use_original_fold: true,
  original_fold_test_value: -1,
  split_random_state: 42,
  merge: false,
}

export const DEFAULT_TRAIN_SPLIT: SplitExportParams = {
  split_strategy: 'holdout_3',
  train_pct: 0.7,
  val_pct: 0.15,
  test_pct: 0.15,
  split_random_state: 42,
  merge: false,
}
