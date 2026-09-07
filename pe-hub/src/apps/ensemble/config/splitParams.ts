export type { SplitExportParams, SplitStrategy } from '@apps/database/config/exportAttributes'

import type { SplitExportParams } from '@apps/database/config/exportAttributes'

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
