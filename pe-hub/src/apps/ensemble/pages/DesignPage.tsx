import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from 'react-query'
import { Play } from 'lucide-react'
import Card from '@components/Card'
import LoadingSpinner from '@components/LoadingSpinner'
import ErrorAlert from '@components/ErrorAlert'
import SelectMenu from '@components/SelectMenu'
import EnsembleMembersPanel, {
  createDefaultMember,
  type EnsembleMemberRow,
} from '@apps/ensemble/components/EnsembleMembersPanel'
import api, { type DesignResponse, type DesignResultRow } from '@apps/ensemble/services/api'
import {
  COMBINE_METHOD_OPTIONS,
  type CombineMethod,
} from '@apps/ensemble/config/combineMethods'
import { DESIGN_RULESET_CHOICES } from '@apps/database/config/exportAttributes'

type DesignMode = 'single' | 'ensemble'
type DesignPolicy = 'optiprime' | 'anzalone'

const POLICY_CHOICES = DESIGN_RULESET_CHOICES.filter(
  (item) => item.value === 'optiprime' || item.value === 'anzalone'
) as Array<{ value: DesignPolicy; label: string; description: string }>

const EXAMPLE_SEQUENCE =
  'ATGCATGCATGCATGCATGCGTCAGTCAGTCAGTCAGTCACGGAAAA(A/G)AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'

function editTypeLabel(row: DesignResultRow): string {
  if (row.type_ins) return `ins${row.edit_len}`
  if (row.type_del) return `del${row.edit_len}`
  return `sub${row.edit_len}`
}

function formatScore(score: number): string {
  if (!Number.isFinite(score)) return '—'
  if (Math.abs(score) >= 1) return score.toFixed(3)
  return score.toFixed(4)
}

export default function DesignPage() {
  const [sequence, setSequence] = useState(EXAMPLE_SEQUENCE)
  const [designPolicy, setDesignPolicy] = useState<DesignPolicy>('optiprime')
  const [mode, setMode] = useState<DesignMode>('single')
  const [modelName, setModelName] = useState('deepprime')
  const [weightId, setWeightId] = useState('')
  const [members, setMembers] = useState<EnsembleMemberRow[]>([
    createDefaultMember('deepprime'),
    createDefaultMember('pridict2'),
  ])
  const [combine, setCombine] = useState<CombineMethod>('mean')
  const [combineOptions, setCombineOptions] = useState<Record<string, unknown>>({
    trim_count: 1,
  })
  const [device, setDevice] = useState('auto')
  const [topK, setTopK] = useState('50')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DesignResponse | null>(null)

  const {
    data: models,
    isLoading: modelsLoading,
    isError: modelsError,
  } = useQuery('models', () => api.listModels(), {
    select: (response) => response.data.models,
  })

  const { data: devices } = useQuery('design-devices', () => api.listDevices(), {
    select: (response) => response.data,
  })

  const { data: weightSets, isFetching: weightsFetching } = useQuery(
    ['model-weights', modelName],
    () => api.listModelWeights(modelName),
    {
      enabled: mode === 'single' && Boolean(modelName),
      select: (response) => response.data.weights,
    }
  )

  const uniqueMemberModels = useMemo(
    () => [...new Set(members.map((member) => member.modelName))],
    [members]
  )

  const weightQueries = useQuery(
    ['design-member-weights', uniqueMemberModels.join(',')],
    async () => {
      const entries = await Promise.all(
        uniqueMemberModels.map(async (name) => {
          const response = await api.listModelWeights(name)
          return [name, response.data.weights] as const
        })
      )
      return Object.fromEntries(entries)
    },
    {
      enabled: mode === 'ensemble' && uniqueMemberModels.length > 0,
      staleTime: 60_000,
    }
  )

  useEffect(() => {
    if (!weightSets?.length) return
    if (!weightId || !weightSets.some((item) => item.id === weightId)) {
      setWeightId(weightSets[0].id)
    }
  }, [weightSets, weightId])

  const designMutation = useMutation(
    () => {
      const top = Number(topK)
      const payload =
        mode === 'single'
          ? {
              sequence,
              design_policy: designPolicy,
              mode: 'single' as const,
              model_name: modelName,
              weights: weightId,
              device,
              top_k: Number.isFinite(top) && top > 0 ? top : 50,
            }
          : {
              sequence,
              design_policy: designPolicy,
              mode: 'ensemble' as const,
              members: members.map((member) => ({
                model_name: member.modelName,
                weights: member.weightId,
                ...(combine === 'weighted_mean'
                  ? { member_weight: Number(member.memberWeight) || 0 }
                  : {}),
              })),
              combine,
              combine_options:
                combine === 'trimmed_mean'
                  ? { trim_count: Number(combineOptions.trim_count ?? 1) }
                  : {},
              device,
              top_k: Number.isFinite(top) && top > 0 ? top : 50,
            }
      return api.design(payload)
    },
    {
      onSuccess: (response) => {
        setResult(response.data)
        setError(null)
      },
      onError: (err: unknown) => {
        const detail =
          (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
          (err as Error)?.message ||
          'Design failed'
        setError(typeof detail === 'string' ? detail : JSON.stringify(detail))
      },
    }
  )

  const handleDesign = () => {
    if (!sequence.trim()) {
      setError('Enter a target sequence with one (pre/after) edit annotation')
      return
    }
    if (mode === 'single' && !weightId) {
      setError('Select a weight set')
      return
    }
    if (mode === 'ensemble') {
      if (members.some((member) => !member.weightId)) {
        setError('Select weights for every ensemble member')
        return
      }
    }
    setError(null)
    designMutation.mutate()
  }

  if (modelsLoading) return <LoadingSpinner message="Loading models..." />
  if (modelsError) {
    return <ErrorAlert message="Failed to load models from PE Ensemble" />
  }

  return (
    <div className="space-y-6">
      <Card title="pegRNA Design">
        <p className="text-slate-600">
          Annotate the intended edit with{' '}
          <code className="mx-1 rounded bg-slate-100 px-1.5 py-0.5 text-sm">(pre/after)</code>
          on the target strand, choose a design policy and model weights (or an ensemble), then
          rank SpCas9 pegRNAs that pass the policy.
        </p>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1" title="Design input">
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Target sequence
              </label>
              <textarea
                value={sequence}
                onChange={(event) => setSequence(event.target.value)}
                rows={8}
                spellCheck={false}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="…NGG…(A/G)…"
              />
              <p className="mt-2 text-xs text-slate-500">
                Substitution <code>(A/G)</code>, insertion <code>(/GGG)</code>, deletion{' '}
                <code>(AAA/)</code>. Forward-strand SpCas9 NGG only.
              </p>
            </div>

            <div>
              <p className="mb-2 text-sm font-medium text-slate-700">Design policy</p>
              <div className="space-y-2">
                {POLICY_CHOICES.map((item) => (
                  <label
                    key={item.value}
                    className={`flex cursor-pointer gap-3 rounded-lg border p-3 ${
                      designPolicy === item.value
                        ? 'border-primary-500 bg-primary-50 ring-1 ring-primary-500'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <input
                      type="radio"
                      name="design-policy"
                      checked={designPolicy === item.value}
                      onChange={() => setDesignPolicy(item.value)}
                      className="mt-1"
                    />
                    <span>
                      <span className="block font-medium text-slate-900">{item.label}</span>
                      <span className="mt-1 block text-xs text-slate-500">{item.description}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <p className="mb-2 text-sm font-medium text-slate-700">Scoring mode</p>
              <div className="grid grid-cols-2 gap-2">
                {(
                  [
                    { id: 'single', label: 'Single weight' },
                    { id: 'ensemble', label: 'Ensemble' },
                  ] as const
                ).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setMode(item.id)}
                    className={`rounded-lg border px-3 py-2 text-sm font-medium ${
                      mode === item.id
                        ? 'border-primary-500 bg-primary-50 text-primary-800'
                        : 'border-slate-200 text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            {mode === 'single' && (
              <>
                <div>
                  <label className="mb-2 block text-sm font-medium text-slate-700">Model</label>
                  <SelectMenu
                    value={modelName}
                    onChange={(value) => {
                      setModelName(value)
                      setWeightId('')
                    }}
                    aria-label="Model"
                    options={(models ?? []).map((model) => ({
                      value: model.name,
                      label: model.name,
                    }))}
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium text-slate-700">Weights</label>
                  <SelectMenu
                    value={weightId}
                    onChange={setWeightId}
                    aria-label="Weights"
                    disabled={weightsFetching || !weightSets?.length}
                    options={(weightSets ?? []).map((weight) => ({
                      value: weight.id,
                      label: weight.label || weight.id,
                    }))}
                  />
                </div>
              </>
            )}

            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">Device</label>
              <SelectMenu
                value={device}
                onChange={setDevice}
                aria-label="Device"
                options={[
                  { value: 'auto', label: 'auto' },
                  ...(devices?.devices ?? []).map((item) => ({
                    value: item.device_id,
                    label: item.name || item.device_id,
                  })),
                ]}
              />
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Top designs to return
              </label>
              <input
                type="number"
                min={1}
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>

            <button
              type="button"
              onClick={handleDesign}
              disabled={designMutation.isLoading}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary-600 px-4 py-2 font-semibold text-white transition hover:bg-primary-700 disabled:bg-slate-400"
            >
              <Play className="h-4 w-4" />
              {designMutation.isLoading ? 'Designing…' : 'Design & rank'}
            </button>

            {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}
          </div>
        </Card>

        <div className="space-y-6 lg:col-span-2">
          {mode === 'ensemble' && (
            <EnsembleMembersPanel
              models={models}
              weightSetsByModel={weightQueries.data ?? {}}
              weightsLoadingModel={weightQueries.isFetching ? uniqueMemberModels[0] ?? null : null}
              onLoadWeights={() => undefined}
              members={members}
              onMembersChange={setMembers}
              combine={combine}
              onCombineChange={setCombine}
              combineOptions={combineOptions}
              onCombineOptionsChange={setCombineOptions}
              combineOptionsList={COMBINE_METHOD_OPTIONS}
            />
          )}

          <Card title="Ranked designs">
            {designMutation.isLoading ? (
              <LoadingSpinner message="Enumerating and scoring pegRNAs…" />
            ) : result ? (
              <div className="space-y-4">
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                  <p>
                    Policy <span className="font-medium">{result.design_policy}</span> ·{' '}
                    {result.n_qualified} qualified of {result.n_candidates} candidates · showing{' '}
                    {result.designs.length}
                  </p>
                  {result.message && <p className="mt-1 text-slate-500">{result.message}</p>}
                </div>

                {result.designs.length === 0 ? (
                  <p className="py-8 text-center text-slate-500">
                    No qualified designs for this sequence and policy.
                  </p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-slate-200 text-sm">
                      <thead className="bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                        <tr>
                          <th className="px-3 py-2">Rank</th>
                          <th className="px-3 py-2">Score</th>
                          <th className="px-3 py-2">PBS</th>
                          <th className="px-3 py-2">RTT</th>
                          <th className="px-3 py-2">Homology</th>
                          <th className="px-3 py-2">PAM</th>
                          <th className="px-3 py-2">Spacer</th>
                          <th className="px-3 py-2">Edit</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white">
                        {result.designs.map((row) => (
                          <tr key={`${row.rank}-${row.spacer}-${row.pbs_len}-${row.rtt_len}`}>
                            <td className="px-3 py-2 font-medium text-slate-900">{row.rank}</td>
                            <td className="px-3 py-2 font-mono text-slate-900">
                              {formatScore(row.score)}
                            </td>
                            <td className="px-3 py-2">{row.pbs_len ?? '—'}</td>
                            <td className="px-3 py-2">{row.rtt_len ?? '—'}</td>
                            <td className="px-3 py-2">{row.homology_len ?? '—'}</td>
                            <td className="px-3 py-2 font-mono">{row.pam ?? '—'}</td>
                            <td className="px-3 py-2 font-mono text-xs">{row.spacer ?? '—'}</td>
                            <td className="px-3 py-2">{editTypeLabel(row)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-12 text-center text-slate-500">
                Submit a design to see ranked pegRNAs here.
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
