import { useCallback, useMemo } from 'react'
import { useQuery } from 'react-query'
import peDbApi from '@apps/database/services/peDbApi'
import type { AttributeFilterRow, FilterAttributeKey } from '@apps/database/config/exportAttributes'
import {
  buildBaseOptionsByAttribute,
  optionsForAttribute,
  type CatalogSnapshot,
} from '@/utils/catalog'

export function useCatalogFilterOptions(filterRows: AttributeFilterRow[] = []) {
  const studiesQuery = useQuery('pe-db-catalog-studies', () => peDbApi.listStudies(), {
    select: (response) => response.data,
  })
  const datasetsQuery = useQuery('pe-db-catalog-datasets', () => peDbApi.listDatasets(), {
    select: (response) => response.data,
  })
  const datasheetsQuery = useQuery('pe-db-catalog-datasheets', () => peDbApi.listDatasheets(), {
    select: (response) => response.data,
  })
  const scaffoldsQuery = useQuery('pe-db-catalog-scaffolds', () => peDbApi.listScaffolds(), {
    select: (response) => response.data,
  })
  const statsQuery = useQuery('pe-db-catalog-statistics', () => peDbApi.getStatistics(), {
    select: (response) => response.data,
  })

  const snapshot = useMemo<CatalogSnapshot | undefined>(() => {
    if (
      studiesQuery.data === undefined ||
      datasetsQuery.data === undefined ||
      datasheetsQuery.data === undefined ||
      scaffoldsQuery.data === undefined
    ) {
      return undefined
    }

    return {
      studies: studiesQuery.data,
      datasets: datasetsQuery.data,
      datasheets: datasheetsQuery.data,
      scaffolds: scaffoldsQuery.data,
      statistics: statsQuery.data,
    }
  }, [studiesQuery.data, datasetsQuery.data, datasheetsQuery.data, scaffoldsQuery.data, statsQuery.data])

  const optionsByAttribute = useMemo<Partial<Record<FilterAttributeKey, string[]>>>(() => {
    if (!snapshot) return {}
    return buildBaseOptionsByAttribute(snapshot)
  }, [snapshot])

  const getOptionsForRow = useCallback(
    (rowId: string, attribute: FilterAttributeKey) => {
      if (!snapshot) return optionsByAttribute[attribute] ?? []
      return optionsForAttribute(attribute, filterRows, snapshot, rowId)
    },
    [filterRows, optionsByAttribute, snapshot]
  )

  const isLoading =
    studiesQuery.isLoading ||
    datasetsQuery.isLoading ||
    datasheetsQuery.isLoading ||
    scaffoldsQuery.isLoading ||
    statsQuery.isLoading

  const error =
    studiesQuery.error ||
    datasetsQuery.error ||
    datasheetsQuery.error ||
    scaffoldsQuery.error ||
    statsQuery.error

  return { optionsByAttribute, getOptionsForRow, isLoading, error }
}
