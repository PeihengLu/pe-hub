import {
  filtersForDatasheetGroup,
  type AttributeFilterRow,
  type DatasheetGroup,
} from '@apps/database/config/exportAttributes'
import type { CombineMethod } from '@apps/ensemble/config/combineMethods'
import type { EnsembleMemberInput, EnsembleRequest } from '@apps/ensemble/services/api'
import type { SplitExportParams } from '@apps/ensemble/config/splitParams'
import { buildDatasetLabel } from '@apps/ensemble/utils/trainingRequest'

export { buildTrainingSplitParams as buildEnsembleSplitParams } from '@apps/ensemble/utils/trainingRequest'

export function buildEnsembleRequestForGroup(input: {
  ensembleName: string
  combine: CombineMethod
  combineOptions: Record<string, unknown>
  members: EnsembleMemberInput[]
  device: string
  split: SplitExportParams
  filterRows: AttributeFilterRow[]
  group?: DatasheetGroup
}): EnsembleRequest {
  const filters = filtersForDatasheetGroup(input.filterRows, input.group)

  return {
    ensemble_name: buildDatasetLabel(input.filterRows, input.group),
    combine: input.combine,
    combine_options: input.combineOptions,
    members: input.members,
    split: input.split,
    device: input.device,
    ...filters,
  }
}
