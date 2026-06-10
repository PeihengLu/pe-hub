import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { Play } from 'lucide-react'
import Card from '@components/Card'
import LoadingSpinner from '@components/LoadingSpinner'
import ErrorAlert from '@components/ErrorAlert'
import SelectMenu from '@components/SelectMenu'
import api from '@apps/ensemble/services/api'
import ComputeJobList from '@apps/ensemble/components/ComputeJobList'
import ModelDataPanel from '@apps/ensemble/components/ModelDataPanel'
import TrainingHyperparametersPanel, {
  DEFAULT_HYPERPARAMETERS,
  buildHyperparametersPayload,
  type HyperparameterFormState,
} from '@apps/ensemble/components/TrainingHyperparametersPanel'
import { exportFormatForModel } from '@apps/ensemble/config/modelFormats'
import {
  buildFilterParams,
  type AttributeFilterRow,
  type SplitStrategy,
} from '@apps/database/config/exportAttributes'
import peDbApi, { type ExportResponse } from '@apps/database/services/peDbApi'
import {
  buildTrainingRequestForGroup,
  buildTrainingSplitParams,
  singleValueFromFilters,
} from '@apps/ensemble/utils/trainingRequest'
import {
  jobStatusRefetchInterval,
  TERMINAL_JOB_STATUSES,
} from '@apps/ensemble/utils/jobStatus'

export default function TrainingPage() {
  const [modelName, setModelName] = useState('deepprime')
  const [device, setDevice] = useState('auto')
  const [filterRows, setFilterRows] = useState<AttributeFilterRow[]>([])
  const [splitStrategy, setSplitStrategy] = useState<SplitStrategy>('holdout_3')
  const [trainPct, setTrainPct] = useState('0.7')
  const [valPct, setValPct] = useState('0.15')
  const [testPct, setTestPct] = useState('0.15')
  const [cvFolds, setCvFolds] = useState('5')
  const [useOriginalFold, setUseOriginalFold] = useState(false)
  const [originalFoldTestValue, setOriginalFoldTestValue] = useState('-1')
  const [splitRandomState, setSplitRandomState] = useState('42')
  const [batchTraining, setBatchTraining] = useState(false)
  const [previewData, setPreviewData] = useState<ExportResponse | undefined>()
  const [hyperparameters, setHyperparameters] =
    useState<HyperparameterFormState>(DEFAULT_HYPERPARAMETERS)
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

  const { data: devices } = useQuery('ensemble-devices', () => api.listDevices(), {
    select: (response) => response.data,
  })

  const { data: deviceStatus } = useQuery('training-devices', () => api.listTrainingDevices(), {
    refetchInterval: 3000,
    select: (response) => response.data,
  })

  const { data: jobs, refetch: refetchJobs } = useQuery('training-jobs', () => api.listTrainingJobs(30), {
    refetchInterval: 3000,
    select: (response) => response.data.jobs,
  })

  const { data: jobStatusResponse } = useQuery(
    ['training-status', selectedJobId],
    () => api.getTrainingStatus(selectedJobId!),
    {
      enabled: Boolean(selectedJobId),
      refetchInterval: jobStatusRefetchInterval,
      refetchIntervalInBackground: true,
    }
  )
  const jobStatus = jobStatusResponse?.data

  useEffect(() => {
    if (!jobStatus?.status || !TERMINAL_JOB_STATUSES.has(jobStatus.status)) return
    void queryClient.invalidateQueries('training-jobs')
  }, [jobStatus?.status, queryClient])

  useEffect(() => {
    setPreviewData(undefined)
  }, [modelName, filterRows, splitStrategy, trainPct, valPct, testPct, cvFolds, useOriginalFold, originalFoldTestValue, splitRandomState, batchTraining])

  useEffect(() => {
    if (!selectedJobId) return undefined
    let cancelled = false
    setLogText('')
    logOffsetRef.current = 0

    const pollLogs = async () => {
      try {
        const response = await api.getTrainingLogs(selectedJobId, logOffsetRef.current)
        if (cancelled) return
        if (response.data.log) {
          setLogText((prev) => prev + response.data.log)
          logOffsetRef.current = response.data.next_offset
        }
        if (TERMINAL_JOB_STATUSES.has(response.data.status)) {
          void queryClient.invalidateQueries(['training-status', selectedJobId])
          void queryClient.invalidateQueries('training-jobs')
        }
      } catch {
        // status endpoint will surface terminal errors
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
  const canSubmit = incompleteRows.length === 0

  const trainMutation = useMutation(async () => {
    const split = buildTrainingSplitParams({
      strategy: splitStrategy,
      trainPct,
      valPct,
      testPct,
      cvFolds,
      useOriginalFold,
      originalFoldTestValue,
      randomState: splitRandomState,
      batchTraining,
    })
    const hyperparameterPayload = buildHyperparametersPayload(modelName, hyperparameters)

    const modelKwargsForDeepprime = (cellLine?: string, peSystem?: string) =>
      modelName === 'deepprime' && cellLine && peSystem
        ? { cell_type: cellLine, pe_system: peSystem }
        : undefined

    if (batchTraining) {
      const filters = buildFilterParams(filterRows)
      const exportFormat = exportFormatForModel(modelName)
      const preview = await peDbApi.exportFiltered(exportFormat, filters, split)
      const groups = preview.data.groups
      if (groups.length === 0) {
        throw new Error('No datasheets matched your filters')
      }

      let lastJobId: string | null = null
      for (const group of groups) {
        const response = await api.train(
          buildTrainingRequestForGroup({
            modelName,
            device,
            hyperparameters: hyperparameterPayload,
            modelKwargs: modelKwargsForDeepprime(group.cell_line, group.pe_system),
            split,
            filterRows,
            group,
          })
        )
        lastJobId = response.data.job_id
      }
      return { job_id: lastJobId!, batch_count: groups.length }
    }

    const cellLine = singleValueFromFilters(filterRows, 'cell_line')
    const peSystem = singleValueFromFilters(filterRows, 'pe_system')
    const response = await api.train(
      buildTrainingRequestForGroup({
        modelName,
        device,
        hyperparameters: hyperparameterPayload,
        modelKwargs: modelKwargsForDeepprime(cellLine, peSystem),
        split,
        filterRows,
      })
    )
    return { job_id: response.data.job_id, batch_count: 1 }
  })

  const handleTrain = () => {
    if (!canSubmit) {
      setError('Complete all filter rows before starting training')
      return
    }
    setError(null)
    trainMutation.mutate(undefined, {
      onSuccess: (result) => {
        setSelectedJobId(result.job_id)
        refetchJobs()
      },
      onError: (err: any) => {
        setError(err.response?.data?.detail || err.message || 'Failed to queue training job')
      },
    })
  }

  if (modelsLoading) return <LoadingSpinner message="Loading models..." />

  if (modelsError) {
    return (
      <ErrorAlert message="Could not load models from the Ensemble API. Confirm pe-ensemble is running on port 8001 (or use ./start-all.sh)." />
    )
  }

  const statusLabel = (status: string, queuePosition?: number | null) => {
    if (status === 'queued' && queuePosition) return `queued (#${queuePosition})`
    return status
  }

  return (
    <div className="space-y-6">
      <Card title="Model Training">
        <p className="text-slate-600">
          Choose training data from the PE Database catalog, configure hyperparameters, and queue
          jobs. Batch mode trains each matching datasheet separately; otherwise datasheets are
          merged into one training run.
        </p>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ModelDataPanel
          mode="train"
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
          originalFoldTestValue={originalFoldTestValue}
          onOriginalFoldTestValueChange={setOriginalFoldTestValue}
          splitRandomState={splitRandomState}
          onSplitRandomStateChange={setSplitRandomState}
          batchMode={batchTraining}
          onBatchModeChange={setBatchTraining}
          previewData={previewData}
          onPreviewDataChange={setPreviewData}
        />

        <Card title="Training configuration">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Model</label>
              <SelectMenu
                value={modelName}
                onChange={setModelName}
                aria-label="Model"
                options={
                  models?.map((model) => ({
                    value: model.name,
                    label: model.name,
                  })) ?? []
                }
              />
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

            <div>
              <h3 className="text-sm font-semibold text-slate-900 mb-3">Hyperparameters</h3>
              <TrainingHyperparametersPanel
                modelName={modelName}
                values={hyperparameters}
                onChange={setHyperparameters}
              />
            </div>

            <button
              onClick={handleTrain}
              disabled={trainMutation.isLoading || !canSubmit}
              className="w-full bg-primary-600 hover:bg-primary-700 disabled:bg-slate-400 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4" />
              {trainMutation.isLoading
                ? 'Queueing…'
                : batchTraining
                  ? 'Start batch training'
                  : 'Start training'}
            </button>

            {trainMutation.isSuccess && trainMutation.data.batch_count > 1 && (
              <p className="text-sm text-green-700">
                Queued {trainMutation.data.batch_count} training jobs. Select any job below to
                monitor progress.
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
          getJobTitle={(job) => `${job.model_name ?? ''} · ${job.dataset_name ?? ''}`}
          killJob={(jobId) => api.deleteTrainingJob(jobId)}
          emptyMessage="No training jobs yet."
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

      <Card title="Job Status">
        {!selectedJobId ? (
          <p className="text-slate-500 py-8 text-center">Select a job to view status and logs.</p>
        ) : (
          <div className="space-y-4">
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-sm space-y-1">
              <p><span className="font-medium">Job ID:</span> {selectedJobId}</p>
              <p>
                <span className="font-medium">Status:</span>{' '}
                {statusLabel(jobStatus?.status ?? 'loading...', jobStatus?.queue_position)}
              </p>
              {jobStatus?.device_requested && (
                <p>
                  <span className="font-medium">Device requested:</span> {jobStatus.device_requested}
                </p>
              )}
              {jobStatus?.device_assigned && (
                <p>
                  <span className="font-medium">Device assigned:</span> {jobStatus.device_assigned}
                </p>
              )}
              {jobStatus?.weights_id && (
                <p><span className="font-medium">Weights:</span> {jobStatus.weights_label ?? jobStatus.weights_id}</p>
              )}
              {jobStatus?.error && (
                <p className="text-red-600"><span className="font-medium">Error:</span> {jobStatus.error}</p>
              )}
            </div>

            <div>
              <h3 className="font-semibold text-slate-900 mb-2">Training log</h3>
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
