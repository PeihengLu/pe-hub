import { buildFilterParams, type AttributeFilterRow } from '@apps/database/config/exportAttributes'
import type { ExportGroup } from '@apps/database/services/peDbApi'
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
  group?: Pick<ExportGroup, 'study' | 'dataset' | 'cell_line' | 'pe_system'>
}): EnsembleRequest {
  const filters = input.group
    ? {
        study: [input.group.study],
        dataset: [input.group.dataset],
        cell_line: [input.group.cell_line],
        pe_system: [input.group.pe_system],
      }
    : buildFilterParams(input.filterRows)

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
