import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { Play } from 'lucide-react'
import Card from '@components/Card'
import LoadingSpinner from '@components/LoadingSpinner'
import ErrorAlert from '@components/ErrorAlert'
import SelectMenu from '@components/SelectMenu'
import ComputeJobList from '@apps/ensemble/components/ComputeJobList'
import ModelDataPanel from '@apps/ensemble/components/ModelDataPanel'
import EnsembleMembersPanel, {
  createDefaultMember,
  type EnsembleMemberRow,
} from '@apps/ensemble/components/EnsembleMembersPanel'
import api, { type EnsembleMemberInput, type EnsembleJobStatusResponse, type EnsembleMemberMetrics } from '@apps/ensemble/services/api'
import { DEFAULT_EVAL_SPLIT } from '@apps/ensemble/config/splitParams'
import {
  COMBINE_METHOD_OPTIONS,
  type CombineMethod,
} from '@apps/ensemble/config/combineMethods'
import {
  buildEnsembleRequestForGroup,
  buildEnsembleSplitParams,
} from '@apps/ensemble/utils/ensembleRequest'
import { resolveBatchGroups } from '@apps/ensemble/utils/batchGroups'
import type { AttributeFilterRow, SplitStrategy } from '@apps/database/config/exportAttributes'
import type { ExportResponse } from '@apps/database/services/peDbApi'
import {
  jobStatusRefetchInterval,
  TERMINAL_JOB_STATUSES,
} from '@apps/ensemble/utils/jobStatus'

function membersToPayload(members: EnsembleMemberRow[], combine: CombineMethod): EnsembleMemberInput[] {
  return members.map((member) => ({
    model_name: member.modelName,
    weights: member.weightId,
    ...(combine === 'weighted_mean'
      ? { member_weight: Number(member.memberWeight) || 0 }
      : {}),
  }))
}

function buildCombineOptions(
  combine: CombineMethod,
  combineOptions: Record<string, unknown>
): Record<string, unknown> {
  if (combine === 'trimmed_mean') {
    return { trim_count: Number(combineOptions.trim_count ?? 1) }
  }
  return {}
}

export default function EnsemblePage() {
  const [members, setMembers] = useState<EnsembleMemberRow[]>([
    createDefaultMember('deepprime'),
    createDefaultMember('pridict2'),
  ])
  const [combine, setCombine] = useState<CombineMethod>('mean')
  const [combineOptions, setCombineOptions] = useState<Record<string, unknown>>({ trim_count: 1 })
  const [device, setDevice] = useState('auto')
  const [filterRows, setFilterRows] = useState<AttributeFilterRow[]>([])
  const [splitStrategy, setSplitStrategy] = useState<SplitStrategy>(
    DEFAULT_EVAL_SPLIT.split_strategy
  )
  const [trainPct, setTrainPct] = useState(String(DEFAULT_EVAL_SPLIT.train_pct ?? 0.8))
  const [valPct, setValPct] = useState('0.15')
  const [testPct, setTestPct] = useState(String(DEFAULT_EVAL_SPLIT.test_pct ?? 0.2))
  const [cvFolds, setCvFolds] = useState('5')
  const [useOriginalFold, setUseOriginalFold] = useState(
    DEFAULT_EVAL_SPLIT.use_original_fold ?? true
  )
  const [originalFoldTestValue, setOriginalFoldTestValue] = useState(
    String(DEFAULT_EVAL_SPLIT.original_fold_test_value ?? -1)
  )
  const [splitRandomState, setSplitRandomState] = useState(
    String(DEFAULT_EVAL_SPLIT.split_random_state ?? 42)
  )
  const [batchEnsemble, setBatchEnsemble] = useState(false)
  const [previewData, setPreviewData] = useState<ExportResponse | undefined>()
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [logText, setLogText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [weightsLoadingModel, setWeightsLoadingModel] = useState<string | null>(null)
  const [queueProgress, setQueueProgress] = useState<{
    phase: 'discovering' | 'queueing'
    current: number
    total: number
  } | null>(null)
  const logRef = useRef<HTMLPreElement>(null)
  const logOffsetRef = useRef(0)
  const queryClient = useQueryClient()

  const uniqueMemberModels = useMemo(
    () => [...new Set(members.map((member) => member.modelName))],
    [members]
  )

  const {
    data: models,
    isLoading: modelsLoading,
    isError: modelsError,
  } = useQuery('models', () => api.listModels(), {
    select: (response) => response.data.models,
  })

  const { data: devices } = useQuery('ensemble-devices', () => api.listDevices(), {
    select: (response) => response.data,
  })

  const { data: deviceStatus } = useQuery('ensemble-run-devices', () => api.listEnsembleDevices(), {
    refetchInterval: 3000,
    select: (response) => response.data,
  })

  const { data: jobs, refetch: refetchJobs } = useQuery(
    'ensemble-jobs',
    () => api.listEnsembleJobs(30),
    {
      refetchInterval: 3000,
      select: (response) => response.data.jobs,
    }
  )

  const { data: jobStatusResponse } = useQuery(
    ['ensemble-status', selectedJobId],
    () => api.getEnsembleStatus(selectedJobId!),
    {
      enabled: Boolean(selectedJobId),
      refetchInterval: jobStatusRefetchInterval,
      refetchIntervalInBackground: true,
    }
  )
  const jobStatus = jobStatusResponse?.data as EnsembleJobStatusResponse | undefined

  const weightQueries = useQuery(
    ['ensemble-member-weights', uniqueMemberModels.join(',')],
    async () => {
      const entries = await Promise.all(
        uniqueMemberModels.map(async (modelName) => {
          const response = await api.listModelWeights(modelName)
          return [modelName, response.data.weights] as const
        })
      )
      return Object.fromEntries(entries)
    },
    {
      enabled: uniqueMemberModels.length > 0,
      staleTime: 60_000,
    }
  )

  const weightSetsByModel = weightQueries.data ?? {}

  useEffect(() => {
    if (!jobStatus?.status || !TERMINAL_JOB_STATUSES.has(jobStatus.status)) return
    void queryClient.invalidateQueries('ensemble-jobs')
  }, [jobStatus?.status, queryClient])

  useEffect(() => {
    setPreviewData(undefined)
  }, [filterRows, splitStrategy, trainPct, valPct, testPct, cvFolds, useOriginalFold, originalFoldTestValue, splitRandomState, batchEnsemble])

  useEffect(() => {
    if (!selectedJobId) return undefined
    let cancelled = false
    setLogText('')
    logOffsetRef.current = 0

    const pollLogs = async () => {
      try {
        const response = await api.getEnsembleLogs(selectedJobId, logOffsetRef.current)
        if (cancelled) return
        if (response.data.log) {
          setLogText((prev) => prev + response.data.log)
          logOffsetRef.current = response.data.next_offset
        }
        if (TERMINAL_JOB_STATUSES.has(response.data.status)) {
          void queryClient.invalidateQueries(['ensemble-status', selectedJobId])
          void queryClient.invalidateQueries('ensemble-jobs')
        }
      } catch {
        // status endpoint surfaces terminal errors
      }
    }

    const interval = window.setInterval(pollLogs, 1500)
    pollLogs()
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [queryClient, selectedJobId])

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logText])

  const loadWeightsForModel = (modelName: string) => {
    setWeightsLoadingModel(modelName)
    void queryClient
      .fetchQuery(['ensemble-member-weights', uniqueMemberModels.join(',')])
      .finally(() => setWeightsLoadingModel(null))
  }

  const incompleteRows = filterRows.filter(
    (row) => row.attribute !== '' && row.values.length === 0
  )
  const membersReady = members.every((member) => member.weightId.trim().length > 0)
  const canSubmit = incompleteRows.length === 0 && members.length >= 2 && membersReady

  const ensembleMutation = useMutation(async () => {
    const split = buildEnsembleSplitParams({
      strategy: splitStrategy,
      trainPct,
      valPct,
      testPct,
      cvFolds,
      useOriginalFold,
      originalFoldTestValue,
      randomState: splitRandomState,
      batchTraining: batchEnsemble,
    })
    const memberPayload = membersToPayload(members, combine)
    const resolvedCombineOptions = buildCombineOptions(combine, combineOptions)

    if (batchEnsemble) {
      setQueueProgress({ phase: 'discovering', current: 0, total: 0 })
      const groups = await resolveBatchGroups(filterRows)
      if (groups.length === 0) {
        throw new Error('No datasheets matched your filters')
      }

      let lastJobId: string | null = null
      for (let index = 0; index < groups.length; index += 1) {
        const group = groups[index]
        setQueueProgress({ phase: 'queueing', current: index + 1, total: groups.length })
        const response = await api.runEnsemble(
          buildEnsembleRequestForGroup({
            ensembleName: 'ensemble',
            combine,
            combineOptions: resolvedCombineOptions,
            members: memberPayload,
            device,
            split,
            filterRows,
            group,
          })
        )
        lastJobId = response.data.job_id
        void queryClient.invalidateQueries('ensemble-jobs')
      }
      setQueueProgress(null)
      return { job_id: lastJobId!, batch_count: groups.length }
    }

    const response = await api.runEnsemble(
      buildEnsembleRequestForGroup({
        ensembleName: 'ensemble',
        combine,
        combineOptions: resolvedCombineOptions,
        members: memberPayload,
        device,
        split,
        filterRows,
      })
    )
    return { job_id: response.data.job_id, batch_count: 1 }
  })

  const handleRunEnsemble = () => {
    if (!membersReady) {
      setError('Select a weight set for every ensemble member')
      return
    }
    if (!canSubmit) {
      setError('Complete all filter rows and add at least two members before running')
      return
    }
    if (combine === 'trimmed_mean' && members.length < 3) {
      setError('Trimmed mean requires at least three ensemble members')
      return
    }
    setError(null)
    ensembleMutation.mutate(undefined, {
      onSuccess: (result) => {
        setSelectedJobId(result.job_id)
        void refetchJobs()
      },
      onError: (err: unknown) => {
        setQueueProgress(null)
        const message =
          (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
          (err as Error)?.message ||
          'Failed to queue ensemble job'
        setError(typeof message === 'string' ? message : 'Failed to queue ensemble job')
      },
      onSettled: () => {
        setQueueProgress(null)
      },
    })
  }

  if (modelsLoading) {
    return <LoadingSpinner message="Loading models..." />
  }

  if (modelsError) {
    return (
      <ErrorAlert message="Could not load models from the Ensemble API. Confirm pe-ensemble is running on port 8001 (or use ./start-all.sh)." />
    )
  }

  const statusLabel = (status: string, queuePosition?: number | null) => {
    if (status === 'queued' && queuePosition) return `queued (#${queuePosition})`
    return status
  }

  const metrics = jobStatus?.result?.metrics as Record<string, number> | undefined
  const memberMetrics = jobStatus?.result?.member_metrics ?? []

  return (
    <div className="space-y-6">
      <Card title="Model Ensemble">
        <p className="text-slate-600">
          Combine predictions from multiple models on the same PE Database test split. Choose a
          fusion method (mean, rank mean, geometric mean, and more), queue jobs on the shared
          device scheduler, and compare per-member metrics against the ensemble.
        </p>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ModelDataPanel
          mode="ensemble"
          modelName={members[0]?.modelName ?? 'deepprime'}
          filterRows={filterRows}
          onFilterRowsChange={setFilterRows}
          splitStrategy={splitStrategy}
          onSplitStrategyChange={setSplitStrategy}
          trainPct={trainPct}
          onTrainPctChange={setTrainPct}
          valPct={valPct}
          onValPctChange={setValPct}
          testPct={testPct}
          onTestPctChange={setTestPct}
          cvFolds={cvFolds}
          onCvFoldsChange={setCvFolds}
          useOriginalFold={useOriginalFold}
          onUseOriginalFoldChange={setUseOriginalFold}
          originalFoldTestValue={originalFoldTestValue}
          onOriginalFoldTestValueChange={setOriginalFoldTestValue}
          splitRandomState={splitRandomState}
          onSplitRandomStateChange={setSplitRandomState}
          batchMode={batchEnsemble}
          onBatchModeChange={setBatchEnsemble}
          previewData={previewData}
          onPreviewDataChange={setPreviewData}
        />

        <div className="space-y-6">
          <EnsembleMembersPanel
            models={models}
            weightSetsByModel={weightSetsByModel}
            weightsLoadingModel={weightsLoadingModel}
            onLoadWeights={loadWeightsForModel}
            members={members}
            onMembersChange={setMembers}
            combine={combine}
            onCombineChange={setCombine}
            combineOptions={combineOptions}
            onCombineOptionsChange={setCombineOptions}
            combineOptionsList={COMBINE_METHOD_OPTIONS}
          />

          <Card title="Run configuration">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Device</label>
                <SelectMenu
                  value={device}
                  onChange={setDevice}
                  aria-label="Device"
                  options={[
                    {
                      value: 'auto',
                      label: `auto (${devices?.default ?? 'best available'})`,
                    },
                    ...(devices?.devices.map((item) => ({
                      value: item.device_id,
                      label: `${item.device_id} — ${item.name}`,
                    })) ?? []),
                  ]}
                />
              </div>

              <button
                onClick={handleRunEnsemble}
                disabled={ensembleMutation.isLoading || !canSubmit}
                className="w-full bg-primary-600 hover:bg-primary-700 disabled:bg-slate-400 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4" />
                {ensembleMutation.isLoading
                  ? queueProgress?.phase === 'discovering'
                    ? 'Finding datasheets…'
                    : queueProgress && queueProgress.total > 0
                      ? `Queueing ${queueProgress.current}/${queueProgress.total}…`
                      : 'Queueing…'
                  : batchEnsemble
                    ? 'Start batch ensemble'
                    : 'Start ensemble'}
              </button>

              {ensembleMutation.isSuccess && ensembleMutation.data.batch_count > 1 && (
                <p className="text-sm text-green-700">
                  Queued {ensembleMutation.data.batch_count} ensemble jobs.
                </p>
              )}

              {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}
            </div>
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ComputeJobList
          jobs={jobs}
          selectedJobId={selectedJobId}
          onSelectJob={setSelectedJobId}
          onJobKilled={(jobId) => {
            if (selectedJobId === jobId) {
              setSelectedJobId(null)
              setLogText('')
            }
            void refetchJobs()
          }}
          onKillError={(message) => setError(message)}
          getJobTitle={(job) =>
            `${job.ensemble_name ?? 'ensemble'} · ${job.combine ?? 'mean'} (${job.member_count ?? 0} members)`
          }
          killJob={(jobId) => api.deleteEnsembleJob(jobId)}
          emptyMessage="No ensemble jobs yet."
        />

        <Card title="Device occupancy">
          {!deviceStatus?.devices.length ? (
            <p className="text-slate-500 py-4 text-center text-sm">No devices reported.</p>
          ) : (
            <ul className="space-y-2 text-sm">
              {deviceStatus.devices.map((item) => (
                <li
                  key={item.device_id}
                  className="flex justify-between gap-2 border border-slate-200 rounded-lg px-3 py-2"
                >
                  <span className="font-mono">{item.device_id}</span>
                  <span className="text-slate-600 text-right">
                    {item.running_job_id
                      ? `${item.running_job_kind ?? 'job'} ${item.running_job_id.slice(0, 8)}…`
                      : 'idle'}
                    {item.queued_jobs > 0 ? ` · ${item.queued_jobs} queued` : ''}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <Card title="Ensemble results">
        {!selectedJobId ? (
          <p className="text-slate-500 py-8 text-center">Select a job to view metrics and logs.</p>
        ) : (
          <div className="space-y-4">
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-sm space-y-1">
              <p><span className="font-medium">Job ID:</span> {selectedJobId}</p>
              <p>
                <span className="font-medium">Status:</span>{' '}
                {statusLabel(jobStatus?.status ?? 'loading...', jobStatus?.queue_position)}
              </p>
              {jobStatus?.combine && (
                <p>
                  <span className="font-medium">Combine:</span> {jobStatus.combine}
                </p>
              )}
              {jobStatus?.device_assigned && (
                <p>
                  <span className="font-medium">Device:</span> {jobStatus.device_assigned}
                </p>
              )}
              {jobStatus?.result?.n_samples != null && !jobStatus?.result?.skipped && (
                <p>
                  <span className="font-medium">Test samples:</span> {jobStatus.result.n_samples}
                </p>
              )}
              {jobStatus?.status === 'skipped' && (
                <p className="text-amber-700">
                  <span className="font-medium">Skipped:</span>{' '}
                  {jobStatus.result?.skip_reason ?? jobStatus.error ?? 'No evaluable test data.'}
                </p>
              )}
              {metrics && (
                <div className="pt-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                    Combined metrics
                  </p>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {Object.entries(metrics).map(([key, value]) => (
                      <div key={key} className="rounded border border-slate-200 bg-white px-3 py-2">
                        <p className="text-xs text-slate-500">{key}</p>
                        <p className="font-semibold text-slate-900">
                          {typeof value === 'number' ? value.toFixed(4) : String(value)}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {memberMetrics.length > 0 && (
                <div className="pt-3 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Per-member metrics
                  </p>
                  {memberMetrics.map((member: EnsembleMemberMetrics) => (
                    <div
                      key={`${member.model_name}-${member.weights}`}
                      className="rounded border border-slate-200 bg-white px-3 py-2"
                    >
                      <p className="font-medium text-slate-900">
                        {member.model_name} · {member.weights}
                      </p>
                      {member.metrics && (
                        <p className="text-xs text-slate-600 mt-1">
                          Pearson {member.metrics.pearson?.toFixed(4) ?? '—'} · Spearman{' '}
                          {member.metrics.spearman?.toFixed(4) ?? '—'}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
              {jobStatus?.error && jobStatus.status !== 'skipped' && (
                <p className="text-red-600">
                  <span className="font-medium">Error:</span> {jobStatus.error}
                </p>
              )}
            </div>

            <div>
              <h3 className="font-semibold text-slate-900 mb-2">Execution log</h3>
              <pre
                ref={logRef}
                className="bg-slate-900 text-slate-100 p-4 rounded-lg h-64 overflow-y-auto text-xs font-mono whitespace-pre-wrap"
              >
                {logText ||
                  (jobStatus?.status === 'queued' ||
                  jobStatus?.status === 'running' ||
                  jobStatus?.status === 'stopping'
                    ? 'Waiting for log output...'
                    : 'No log output yet.')}
              </pre>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
