import { useQueries } from 'react-query'
import api, { type WeightSet } from '@apps/ensemble/services/api'

export interface MemberWeightSets {
  weightSetsByModel: Record<string, WeightSet[] | undefined>
  loadingModelNames: string[]
  failedModelNames: string[]
}

/** Load each model's weight sets on its own so one failure does not clear the others. */
export function useMemberWeightSets(modelNames: string[], enabled = true): MemberWeightSets {
  const queries = useQueries(
    modelNames.map((modelName) => ({
      queryKey: ['model-weights', modelName],
      queryFn: () => api.listModelWeights(modelName),
      staleTime: 60_000,
      enabled: enabled && Boolean(modelName),
    }))
  )

  const weightSetsByModel: Record<string, WeightSet[] | undefined> = {}
  const loadingModelNames: string[] = []
  const failedModelNames: string[] = []

  modelNames.forEach((modelName, index) => {
    const query = queries[index]
    if (!query) return
    if (query.data) {
      weightSetsByModel[modelName] = query.data.data.weights
    }
    if (query.isFetching && !query.data) {
      loadingModelNames.push(modelName)
    } else if (query.isError) {
      failedModelNames.push(modelName)
    }
  })

  return { weightSetsByModel, loadingModelNames, failedModelNames }
}
