import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { Play } from 'lucide-react'
import Card from '@components/Card'
import LoadingSpinner from '@components/LoadingSpinner'
import ErrorAlert from '@components/ErrorAlert'
import SelectMenu from '@components/SelectMenu'
import ComputeJobList from '@apps/ensemble/components/ComputeJobList'
import ModelDataPanel from '@apps/ensemble/components/ModelDataPanel'
import api from '@apps/ensemble/services/api'
import { DEFAULT_EVAL_SPLIT } from '@apps/ensemble/config/splitParams'
import { exportFormatForModel } from '@apps/ensemble/config/modelFormats'
import {
  buildBenchmarkRequestForGroup,
  buildBenchmarkSplitParams,
} from '@apps/ensemble/utils/benchmarkRequest'
import { buildFilterParams, type AttributeFilterRow } from '@apps/database/config/exportAttributes'
import type { SplitStrategy } from '@apps/database/config/exportAttributes'
import peDbApi, { type ExportResponse } from '@apps/database/services/peDbApi'
import {
  jobStatusRefetchInterval,
  TERMINAL_JOB_STATUSES,
} from '@apps/ensemble/utils/jobStatus'

export default function BenchmarkPage() {
  const [modelName, setModelName] = useState('deepprime')
  const [weightId, setWeightId] = useState('')
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
  const [splitRandomState, setSplitRandomState] = useState(
    String(DEFAULT_EVAL_SPLIT.split_random_state ?? 42)
  )
  const [batchBenchmark, setBatchBenchmark] = useState(false)
  const [previewData, setPreviewData] = useState<ExportResponse | undefined>()
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [logText, setLogText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const logRef = useRef<HTMLPreElement>(null)
  const logOffsetRef = useRef(0)
  const queryClient = useQueryClient()

  const {
    data: models,
    isLoading: modelsLoading,
    isError: modelsError,
  } = useQuery('models', () => api.listModels(), {
    select: (response) => response.data.models,
  })

  const { data: weightSets, isFetching: weightsFetching } = useQuery(
    ['model-weights', modelName],
    () => api.listModelWeights(modelName),
    {
      enabled: Boolean(modelName),
      select: (response) => response.data.weights,
    }
  )

  const { data: devices } = useQuery('ensemble-devices', () => api.listDevices(), {
    select: (response) => response.data,
  })

  const { data: deviceStatus } = useQuery('benchmark-devices', () => api.listBenchmarkDevices(), {
    refetchInterval: 3000,
    select: (response) => response.data,
  })

  const { data: jobs, refetch: refetchJobs } = useQuery(
    'benchmark-jobs',
    () => api.listBenchmarkJobs(30),
    {
      refetchInterval: 3000,
      select: (response) => response.data.jobs,
    }
  )

  const { data: jobStatusResponse } = useQuery(
    ['benchmark-status', selectedJobId],
    () => api.getBenchmarkStatus(selectedJobId!),
    {
      enabled: Boolean(selectedJobId),
      refetchInterval: jobStatusRefetchInterval,
      refetchIntervalInBackground: true,
    }
  )
  const jobStatus = jobStatusResponse?.data

  useEffect(() => {
    if (!jobStatus?.status || !TERMINAL_JOB_STATUSES.has(jobStatus.status)) return
    void queryClient.invalidateQueries('benchmark-jobs')
  }, [jobStatus?.status, queryClient])

  useEffect(() => {
    setWeightId('')
    setPreviewData(undefined)
  }, [modelName])

  useEffect(() => {
    setPreviewData(undefined)
  }, [filterRows, splitStrategy, trainPct, valPct, testPct, cvFolds, useOriginalFold, splitRandomState, batchBenchmark])

  useEffect(() => {
    if (!selectedJobId) return undefined
    let cancelled = false
    setLogText('')
    logOffsetRef.current = 0

    const pollLogs = async () => {
      try {
        const response = await api.getBenchmarkLogs(selectedJobId, logOffsetRef.current)
        if (cancelled) return
        if (response.data.log) {
          setLogText((prev) => prev + response.data.log)
          logOffsetRef.current = response.data.next_offset
        }
        if (TERMINAL_JOB_STATUSES.has(response.data.status)) {
          void queryClient.invalidateQueries(['benchmark-status', selectedJobId])
          void queryClient.invalidateQueries('benchmark-jobs')
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

  const incompleteRows = filterRows.filter(
    (row) => row.attribute !== '' && row.values.length === 0
  )
  const hasWeightSelection = weightId.trim().length > 0
  const hasRegisteredWeights = (weightSets?.length ?? 0) > 0
  const canSubmit =
    incompleteRows.length === 0 &&
    hasWeightSelection &&
    hasRegisteredWeights &&
    !weightsFetching

  const benchmarkMutation = useMutation(async () => {
    const split = buildBenchmarkSplitParams({
      strategy: splitStrategy,
      trainPct,
      valPct,
      testPct,
      cvFolds,
      useOriginalFold,
      randomState: splitRandomState,
      batchTraining: batchBenchmark,
    })

    if (batchBenchmark) {
      const filters = buildFilterParams(filterRows)
      const exportFormat = exportFormatForModel(modelName)
      const preview = await peDbApi.exportFiltered(exportFormat, filters, split)
      const groups = preview.data.groups
      if (groups.length === 0) {
        throw new Error('No datasheets matched your filters')
      }

      let lastJobId: string | null = null
      for (const group of groups) {
        const response = await api.benchmark(
          buildBenchmarkRequestForGroup({
            modelName,
            device,
            weights: weightId,
            split,
            filterRows,
            group,
          })
        )
        lastJobId = response.data.job_id
      }
      return { job_id: lastJobId!, batch_count: groups.length }
    }

    const response = await api.benchmark(
      buildBenchmarkRequestForGroup({
        modelName,
        device,
        weights: weightId,
        split,
        filterRows,
      })
    )
    return { job_id: response.data.job_id, batch_count: 1 }
  })

  const handleBenchmark = () => {
    if (!hasRegisteredWeights) {
      setError(`No weight sets are registered for ${modelName}. Check GET /models/${modelName}/weights.`)
      return
    }
    if (!hasWeightSelection) {
      setError('Select a weight set before starting the benchmark')
      return
    }
    if (!canSubmit) {
      setError('Complete all filter rows before starting the benchmark')
      return
    }
    setError(null)
    benchmarkMutation.mutate(undefined, {
      onSuccess: (result) => {
        setSelectedJobId(result.job_id)
        refetchJobs()
      },
      onError: (err: any) => {
        setError(err.response?.data?.detail || err.message || 'Failed to queue benchmark job')
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

  return (
    <div className="space-y-6">
      <Card title="Model Benchmark">
        <p className="text-slate-600">
          Evaluate a model on held-out PE Database test splits. Select catalog filters, pick a weight
          set, and queue benchmark jobs on the shared device queue.
        </p>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ModelDataPanel
          mode="benchmark"
          modelName={modelName}
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
          splitRandomState={splitRandomState}
          onSplitRandomStateChange={setSplitRandomState}
          batchMode={batchBenchmark}
          onBatchModeChange={setBatchBenchmark}
          previewData={previewData}
          onPreviewDataChange={setPreviewData}
        />

        <Card title="Benchmark configuration">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Model</label>
              <SelectMenu
                value={modelName}
                onChange={setModelName}
                aria-label="Model"
                options={
                  models?.map((model) => ({
                    value: model.name,
                    label: `${model.name} — ${model.description}`,
                  })) ?? []
                }
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Weight set <span className="text-red-600">*</span>
              </label>
              <SelectMenu
                value={weightId}
                onChange={setWeightId}
                disabled={weightsFetching || !hasRegisteredWeights}
                aria-label="Weight set"
                placeholder={
                  weightsFetching
                    ? 'Loading weight sets…'
                    : hasRegisteredWeights
                      ? 'Select a weight set…'
                      : 'No weights registered'
                }
                options={
                  weightSets?.map((weight) => ({
                    value: weight.id,
                    label: `${weight.label}${weight.source === 'vendor' ? ' [vendor]' : ''}`,
                  })) ?? []
                }
              />
              {!weightsFetching && !hasRegisteredWeights && (
                <p className="mt-1 text-xs text-amber-700">
                  No checkpoints found for {modelName}. Weights must be registered under{' '}
                  <code className="text-xs">services/pe-ensemble/weights/{modelName}/</code>.
                </p>
              )}
            </div>

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
              onClick={handleBenchmark}
              disabled={benchmarkMutation.isLoading || !canSubmit}
              className="w-full bg-primary-600 hover:bg-primary-700 disabled:bg-slate-400 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4" />
              {benchmarkMutation.isLoading
                ? 'Queueing…'
                : batchBenchmark
                  ? 'Start batch benchmark'
                  : 'Start benchmark'}
            </button>

            {benchmarkMutation.isSuccess && benchmarkMutation.data.batch_count > 1 && (
              <p className="text-sm text-green-700">
                Queued {benchmarkMutation.data.batch_count} benchmark jobs.
              </p>
            )}

            {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}
          </div>
        </Card>
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
          getJobTitle={(job) => `${job.model_name ?? ''} · ${job.benchmark_name ?? ''}`}
          killJob={(jobId) => api.deleteBenchmarkJob(jobId)}
          emptyMessage="No benchmark jobs yet."
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

      <Card title="Benchmark results">
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
              {jobStatus?.device_assigned && (
                <p>
                  <span className="font-medium">Device:</span> {jobStatus.device_assigned}
                </p>
              )}
              {jobStatus?.result?.n_samples != null && (
                <p>
                  <span className="font-medium">Test samples:</span> {jobStatus.result.n_samples}
                </p>
              )}
              {metrics && (
                <div className="pt-2 grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {Object.entries(metrics).map(([key, value]) => (
                    <div key={key} className="rounded border border-slate-200 bg-white px-3 py-2">
                      <p className="text-xs text-slate-500">{key}</p>
                      <p className="font-semibold text-slate-900">
                        {typeof value === 'number' ? value.toFixed(4) : String(value)}
                      </p>
                    </div>
                  ))}
                </div>
              )}
              {jobStatus?.error && (
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
                  (jobStatus?.status === 'queued' || jobStatus?.status === 'running'
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
