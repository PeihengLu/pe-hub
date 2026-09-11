import { describe, expect, it } from 'vitest'
import {
  DATASHEET_FILTER_KEYS,
  FILTER_LIST_FIELDS,
  buildFilterParams,
  designRulesetFromFilterRows,
  filtersForDatasheetGroup,
  setDesignRulesetOnFilterRows,
  type AttributeFilterRow,
  type FilterAttributeKey,
} from '@apps/database/config/exportAttributes'
import { DEFAULT_TRAIN_SPLIT } from '@apps/ensemble/config/splitParams'
import { buildBenchmarkRequestForGroup } from '@apps/ensemble/utils/benchmarkRequest'
import { buildEnsembleRequestForGroup } from '@apps/ensemble/utils/ensembleRequest'
import {
  buildDatasetLabel,
  buildTrainingRequestForGroup,
} from '@apps/ensemble/utils/trainingRequest'

function row(attribute: FilterAttributeKey, ...values: string[]): AttributeFilterRow {
  return { id: attribute, attribute, values }
}

const GROUP = {
  study: 'deepprime',
  dataset: 'deepprime-clinvar',
  cell_line: 'HEK293T',
  pe_system: 'PE2max',
}

const FILTER_ROWS: AttributeFilterRow[] = [
  row('study', 'deepprime', 'optiprime'),
  row('dataset', 'deepprime-clinvar', 'lib-mmr'),
  row('cell_line', 'HEK293T', 'K562'),
  row('pe_system', 'PE2max'),
  row('edit_type', 'sub'),
  row('edit_length', '1', '2'),
  row('design_ruleset', 'optiprime'),
  row('scaffold_name', 'tevopreQ1'),
  { id: 'empty', attribute: 'edit_scope', values: [] },
]

describe('filtersForDatasheetGroup', () => {
  it('passes the UI selection through when there is no batch group', () => {
    expect(filtersForDatasheetGroup(FILTER_ROWS)).toEqual(buildFilterParams(FILTER_ROWS))
    expect(filtersForDatasheetGroup(FILTER_ROWS).study).toEqual(['deepprime', 'optiprime'])
  })

  it('pins the datasheet and keeps row-level filters including design_ruleset', () => {
    const filters = filtersForDatasheetGroup(FILTER_ROWS, GROUP)
    expect(filters.study).toEqual(['deepprime'])
    expect(filters.dataset).toEqual(['deepprime-clinvar'])
    expect(filters.cell_line).toEqual(['HEK293T'])
    expect(filters.pe_system).toEqual(['PE2max'])
    expect(filters.edit_type).toEqual(['sub'])
    expect(filters.edit_length).toEqual([1, 2])
    expect(filters.design_ruleset).toEqual(['optiprime'])
    expect(filters.scaffold_name).toEqual(['tevopreQ1'])
    expect(filters).not.toHaveProperty('edit_scope')
  })

  it('keeps every non-datasheet FILTER_LIST_FIELDS key that the UI selected', () => {
    const filters = filtersForDatasheetGroup(FILTER_ROWS, GROUP)
    for (const key of FILTER_LIST_FIELDS) {
      if ((DATASHEET_FILTER_KEYS as readonly string[]).includes(key)) continue
      const selected = FILTER_ROWS.find((item) => item.attribute === key && item.values.length > 0)
      if (!selected) continue
      expect(filters[key], key).toBeDefined()
    }
  })
})

describe('batch request builders', () => {
  it('train, ensemble, and evaluate batch jobs all keep design_ruleset', () => {
    const train = buildTrainingRequestForGroup({
      modelName: 'deepprime',
      device: 'cpu',
      hyperparameters: {},
      split: DEFAULT_TRAIN_SPLIT,
      filterRows: FILTER_ROWS,
      group: GROUP,
    })
    const ensemble = buildEnsembleRequestForGroup({
      ensembleName: 'demo',
      combine: 'mean',
      combineOptions: {},
      members: [{ model_name: 'deepprime', weights: 'w1' }],
      device: 'cpu',
      split: DEFAULT_TRAIN_SPLIT,
      filterRows: FILTER_ROWS,
      group: GROUP,
    })
    const evaluate = buildBenchmarkRequestForGroup({
      modelName: 'deepprime',
      device: 'cpu',
      weights: 'w1',
      split: DEFAULT_TRAIN_SPLIT,
      filterRows: FILTER_ROWS,
      group: GROUP,
    })

    for (const request of [train, ensemble, evaluate]) {
      expect(request.study).toEqual(['deepprime'])
      expect(request.dataset).toEqual(['deepprime-clinvar'])
      expect(request.cell_line).toEqual(['HEK293T'])
      expect(request.pe_system).toEqual(['PE2max'])
      expect(request.edit_type).toEqual(['sub'])
      expect(request.edit_length).toEqual([1, 2])
      expect(request.design_ruleset).toEqual(['optiprime'])
      expect(request.scaffold_name).toEqual(['tevopreQ1'])
    }
  })

  it('merged jobs (no group) keep the original multi-datasheet filters', () => {
    const train = buildTrainingRequestForGroup({
      modelName: 'deepprime',
      device: 'cpu',
      hyperparameters: {},
      split: DEFAULT_TRAIN_SPLIT,
      filterRows: FILTER_ROWS,
    })
    const ensemble = buildEnsembleRequestForGroup({
      ensembleName: 'demo',
      combine: 'mean',
      combineOptions: {},
      members: [{ model_name: 'deepprime', weights: 'w1' }],
      device: 'cpu',
      split: DEFAULT_TRAIN_SPLIT,
      filterRows: FILTER_ROWS,
    })
    const evaluate = buildBenchmarkRequestForGroup({
      modelName: 'deepprime',
      device: 'cpu',
      weights: 'w1',
      split: DEFAULT_TRAIN_SPLIT,
      filterRows: FILTER_ROWS,
    })

    for (const request of [train, ensemble, evaluate]) {
      expect(request.study).toEqual(['deepprime', 'optiprime'])
      expect(request.dataset).toEqual(['deepprime-clinvar', 'lib-mmr'])
      expect(request.design_ruleset).toEqual(['optiprime'])
    }
  })

  it('includes design_ruleset in the batch job label', () => {
    expect(buildDatasetLabel(FILTER_ROWS, GROUP)).toContain('design_ruleset=optiprime')
    expect(buildDatasetLabel(FILTER_ROWS, GROUP)).toContain('edit_type=sub')
  })
})

describe('design ruleset filter rows', () => {
  it('sets, replaces, and clears the dedicated ruleset without dropping other filters', () => {
    const withOptiprime = setDesignRulesetOnFilterRows(
      [row('study', 'deepprime'), row('edit_type', 'sub')],
      'optiprime'
    )
    expect(designRulesetFromFilterRows(withOptiprime)).toBe('optiprime')
    expect(buildFilterParams(withOptiprime)).toEqual({
      study: ['deepprime'],
      edit_type: ['sub'],
      design_ruleset: ['optiprime'],
    })

    const withAnzalone = setDesignRulesetOnFilterRows(withOptiprime, 'anzalone')
    expect(designRulesetFromFilterRows(withAnzalone)).toBe('anzalone')
    expect(withAnzalone.filter((item) => item.attribute === 'design_ruleset')).toHaveLength(1)

    const cleared = setDesignRulesetOnFilterRows(withAnzalone, '')
    expect(designRulesetFromFilterRows(cleared)).toBe('')
    expect(buildFilterParams(cleared)).toEqual({
      study: ['deepprime'],
      edit_type: ['sub'],
    })
  })
})
