import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { CheckCircle2, Plus, Trash2, Upload, XCircle } from 'lucide-react'
import clsx from 'clsx'
import Card from '@components/Card'
import LoadingSpinner from '@components/LoadingSpinner'
import ErrorAlert from '@components/ErrorAlert'
import FormFieldLabel from '@components/FormFieldLabel'
import { PLUGIN_FORM_HINTS } from '@apps/ensemble/config/pluginFormHints'
import api, {
  type PluginSummary,
  type PluginUploadPayload,
  type PluginValidationCheck,
} from '@apps/ensemble/services/api'
import {
  ACTIVE_JOB_STATUSES,
  TERMINAL_JOB_STATUSES,
} from '@apps/ensemble/utils/jobStatus'

const inputClass =
  'w-full rounded-md border border-slate-300 px-3 py-2 text-sm disabled:bg-slate-100 disabled:text-slate-500'

const fieldLabelClass = 'text-sm font-medium text-slate-700'

function FieldLabel({ children, hint }: { children: string; hint: string }) {
  return (
    <FormFieldLabel className={fieldLabelClass} hint={hint}>
      {children}
    </FormFieldLabel>
  )
}

interface HyperparameterRow {
  name: string
  type: string
  default: string
}

interface WeightMetaRow {
  id: string
  notes: string
}

const emptyHyperparameter = (): HyperparameterRow => ({ name: '', type: 'int', default: '' })
const emptyWeightMeta = (): WeightMetaRow => ({ id: '', notes: '' })

function statusBadgeClass(status: string): string {
  switch (status) {
    case 'active':
      return 'bg-emerald-100 text-emerald-800 border-emerald-200'
    case 'rejected':
      return 'bg-red-100 text-red-800 border-red-200'
    case 'pending':
      return 'bg-amber-100 text-amber-900 border-amber-200'
    default:
      return 'bg-slate-100 text-slate-700 border-slate-200'
  }
}

function extractErrorMessage(error: unknown): string {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: { detail?: string } } }).response
    if (typeof response?.data?.detail === 'string') return response.data.detail
  }
  if (error instanceof Error) return error.message
  return 'Request failed'
}

export default function AddModelPage() {
  const queryClient = useQueryClient()
  const [error, setError] = useState<string | null>(null)
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const [validationJobId, setValidationJobId] = useState<string | null>(null)
  const [logText, setLogText] = useState('')
  const logOffsetRef = useRef(0)
  const logRef = useRef<HTMLPreElement>(null)

  const [name, setName] = useState('')
  const [version, setVersion] = useState('0.1.0')
  const [displayName, setDisplayName] = useState('')
  const [description, setDescription] = useState('')
  const [authors, setAuthors] = useState('')
  const [wrapperClass, setWrapperClass] = useState('')
  const [weightFormat, setWeightFormat] = useState('')
  const [convertEntrypoint, setConvertEntrypoint] = useState('convert')
  const [peDbFormat, setPeDbFormat] = useState('')
  const [outputColumns, setOutputColumns] = useState('feature,Efficiency')
  const [requiredStdColumns, setRequiredStdColumns] = useState('')
  const [labelColumn, setLabelColumn] = useState('Efficiency')
  const [replaceExisting, setReplaceExisting] = useState(false)
  const [convertFile, setConvertFile] = useState<File | null>(null)
  const [wrapperFile, setWrapperFile] = useState<File | null>(null)
  const [bundleZip, setBundleZip] = useState<File | null>(null)
  const [weightId, setWeightId] = useState('')
  const [weightFile, setWeightFile] = useState<File | null>(null)
  const [hyperparameterRows, setHyperparameterRows] = useState<HyperparameterRow[]>([
    emptyHyperparameter(),
  ])
  const [weightMetaRows, setWeightMetaRows] = useState<WeightMetaRow[]>([emptyWeightMeta()])
  const [useZipBundle, setUseZipBundle] = useState(false)

  const {
    data: plugins,
    isLoading: pluginsLoading,
    refetch: refetchPlugins,
  } = useQuery('plugins', () => api.listPlugins(), {
    select: (response) => response.data.plugins,
    refetchInterval: 5000,
  })

  const { data: pluginDetail, refetch: refetchPluginDetail } = useQuery(
    ['plugin-detail', selectedName],
    () => api.getPlugin(selectedName!),
    {
      enabled: Boolean(selectedName),
      select: (response) => response.data,
      refetchInterval: validationJobId ? 3000 : false,
    }
  )

  const { data: validationStatusResponse } = useQuery(
    ['plugin-validation-status', selectedName, validationJobId],
    () => api.getPluginValidationStatus(selectedName!, validationJobId!),
    {
      enabled: Boolean(selectedName && validationJobId),
      refetchInterval: (data) => {
        const status = data?.data?.status
        if (!status) return 2000
        return ACTIVE_JOB_STATUSES.has(status) ? 2000 : false
      },
      refetchIntervalInBackground: true,
    }
  )
  const validationJob = validationStatusResponse?.data

  useEffect(() => {
    if (!validationJobId || !validationJob?.status) return
    if (!TERMINAL_JOB_STATUSES.has(validationJob.status)) return
    void refetchPluginDetail()
    void refetchPlugins()
    void queryClient.invalidateQueries('models')
  }, [
    validationJobId,
    validationJob?.status,
    refetchPluginDetail,
    refetchPlugins,
    queryClient,
  ])

  useEffect(() => {
    if (!selectedName || !validationJobId) return undefined
    let cancelled = false
    setLogText('')
    logOffsetRef.current = 0

    const pollLogs = async () => {
      while (!cancelled) {
        try {
          const response = await api.getPluginValidationLogs(
            selectedName,
            validationJobId,
            logOffsetRef.current
          )
          if (response.data.log) {
            setLogText((prev) => prev + response.data.log)
            logOffsetRef.current = response.data.next_offset
          }
          const status = response.data.status
          if (TERMINAL_JOB_STATUSES.has(status)) break
        } catch {
          break
        }
        await new Promise((resolve) => setTimeout(resolve, 1500))
      }
    }

    void pollLogs()
    return () => {
      cancelled = true
    }
  }, [selectedName, validationJobId])

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logText])

  const uploadMutation = useMutation(
    (payload: PluginUploadPayload) => api.uploadPlugin(payload),
    {
      onSuccess: (response) => {
        setError(null)
        setSelectedName(response.data.name)
        setValidationJobId(null)
        void refetchPlugins()
        void queryClient.invalidateQueries(['plugin-detail', response.data.name])
      },
      onError: (err) => setError(extractErrorMessage(err)),
    }
  )

  const validateMutation = useMutation(
    (pluginName: string) => api.validatePlugin(pluginName),
    {
      onSuccess: (response, pluginName) => {
        setError(null)
        setSelectedName(pluginName)
        setValidationJobId(response.data.job_id)
        setLogText('')
        logOffsetRef.current = 0
      },
      onError: (err) => setError(extractErrorMessage(err)),
    }
  )

  const activateMutation = useMutation(
    (pluginName: string) => api.activatePlugin(pluginName),
    {
      onSuccess: () => {
        setError(null)
        void refetchPlugins()
        void refetchPluginDetail()
        void queryClient.invalidateQueries('models')
      },
      onError: (err) => setError(extractErrorMessage(err)),
    }
  )

  const deleteMutation = useMutation((pluginName: string) => api.deletePlugin(pluginName), {
    onSuccess: (_, pluginName) => {
      setError(null)
      if (selectedName === pluginName) {
        setSelectedName(null)
        setValidationJobId(null)
      }
      void refetchPlugins()
      void queryClient.invalidateQueries('models')
    },
    onError: (err) => setError(extractErrorMessage(err)),
  })

  const buildUploadPayload = (): PluginUploadPayload | null => {
    const trimmedName = name.trim().toLowerCase()
    if (!trimmedName) {
      setError('Plugin name is required')
      return null
    }
    if (!wrapperClass.trim()) {
      setError('Wrapper class name is required')
      return null
    }
    if (!weightFormat.trim()) {
      setError('Weight format is required')
      return null
    }
    if (useZipBundle) {
      if (!bundleZip) {
        setError('Zip bundle is required when using zip upload mode')
        return null
      }
    } else if (!convertFile || !wrapperFile) {
      setError('convert.py and wrapper.py are required')
      return null
    }

    const hyperparameters = hyperparameterRows
      .filter((row) => row.name.trim())
      .map((row) => ({
        name: row.name.trim(),
        type: row.type.trim() || 'string',
        default: row.default.trim() ? Number.isNaN(Number(row.default)) ? row.default : Number(row.default) : undefined,
      }))

    const weights = weightMetaRows
      .filter((row) => row.id.trim())
      .map((row) => ({
        id: row.id.trim(),
        notes: row.notes.trim() || undefined,
      }))

    return {
      name: trimmedName,
      version: version.trim() || '0.1.0',
      display_name: displayName.trim() || trimmedName,
      description: description.trim() || 'Plugin model',
      wrapper_class: wrapperClass.trim(),
      weight_format: weightFormat.trim(),
      authors: authors.trim() || undefined,
      convert_entrypoint: convertEntrypoint.trim() || 'convert',
      pe_db_format: peDbFormat.trim() || undefined,
      output_columns: outputColumns.trim() || undefined,
      required_std_columns: requiredStdColumns.trim() || undefined,
      label_column: labelColumn.trim() || undefined,
      hyperparameters_json: hyperparameters.length ? JSON.stringify(hyperparameters) : undefined,
      weights_json: weights.length ? JSON.stringify(weights) : undefined,
      replace_existing: replaceExisting,
      convert_file: useZipBundle ? null : convertFile,
      wrapper_file: useZipBundle ? null : wrapperFile,
      bundle_zip: useZipBundle ? bundleZip : null,
      weight_id: weightId.trim() || undefined,
      weight_file: weightFile,
    }
  }

  const handleUpload = () => {
    const payload = buildUploadPayload()
    if (!payload) return
    uploadMutation.mutate(payload)
  }

  const validationReport =
    validationJob?.result?.validation_report ?? pluginDetail?.validation_report
  const validationPassed = Boolean(validationReport?.passed)
  const validationRunning =
    validationJobId &&
    validationJob?.status &&
    ACTIVE_JOB_STATUSES.has(validationJob.status)

  const canActivate =
    pluginDetail?.status !== 'active' &&
    validationPassed &&
    !validationRunning &&
    !validateMutation.isLoading

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Add New Model</h2>
        <p className="text-sm text-slate-600 mt-1">
          Upload a plugin bundle, run the validation harness, and activate it for Train and
          Benchmark.
        </p>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      <Card title="Installed plugins">
        {pluginsLoading ? (
          <LoadingSpinner />
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-slate-500">
                  <th className="py-2 pr-4">Name</th>
                  <th className="py-2 pr-4">Display</th>
                  <th className="py-2 pr-4">Status</th>
                  <th className="py-2 pr-4">Validation</th>
                  <th className="py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {(plugins ?? []).map((plugin: PluginSummary) => (
                  <tr
                    key={plugin.name}
                    className={clsx(
                      'border-b border-slate-100 hover:bg-slate-50',
                      selectedName === plugin.name && 'bg-primary-50'
                    )}
                  >
                    <td className="py-3 pr-4 font-mono text-slate-800">{plugin.name}</td>
                    <td className="py-3 pr-4">{plugin.display_name}</td>
                    <td className="py-3 pr-4">
                      <span
                        className={clsx(
                          'inline-flex rounded-full border px-2 py-0.5 text-xs font-medium',
                          statusBadgeClass(plugin.status)
                        )}
                      >
                        {plugin.status}
                      </span>
                    </td>
                    <td className="py-3 pr-4">
                      {plugin.validation_passed == null
                        ? '—'
                        : plugin.validation_passed
                          ? 'Passed'
                          : 'Failed'}
                    </td>
                    <td className="py-3">
                      <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          className="text-primary-700 hover:underline"
                          onClick={() => {
                            setSelectedName(plugin.name)
                            setValidationJobId(null)
                          }}
                        >
                          View
                        </button>
                        <button
                          type="button"
                          className="text-slate-700 hover:underline disabled:opacity-50"
                          disabled={validateMutation.isLoading}
                          onClick={() => validateMutation.mutate(plugin.name)}
                        >
                          Validate
                        </button>
                        <button
                          type="button"
                          className="text-emerald-700 hover:underline disabled:opacity-50"
                          disabled={
                            plugin.status === 'active' ||
                            plugin.validation_passed !== true ||
                            activateMutation.isLoading
                          }
                          onClick={() => activateMutation.mutate(plugin.name)}
                        >
                          Activate
                        </button>
                        <button
                          type="button"
                          className="text-red-700 hover:underline disabled:opacity-50"
                          disabled={deleteMutation.isLoading}
                          onClick={() => deleteMutation.mutate(plugin.name)}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {!plugins?.length && (
                  <tr>
                    <td colSpan={5} className="py-6 text-center text-slate-500">
                      No plugins yet. Upload one below.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {selectedName && pluginDetail && (
        <Card title={`Plugin: ${selectedName}`}>
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <span
                className={clsx(
                  'inline-flex rounded-full border px-2 py-0.5 text-xs font-medium',
                  statusBadgeClass(pluginDetail.status)
                )}
              >
                {pluginDetail.status}
              </span>
              {validationRunning && (
                <span className="text-sm text-amber-700">Validation in progress…</span>
              )}
              {validationJob?.status && (
                <span className="text-sm text-slate-600">
                  Job {validationJobId}: {validationJob.status}
                </span>
              )}
            </div>

            {validationReport && (
              <div>
                <h3 className="text-sm font-semibold text-slate-800 mb-2">Validation report</h3>
                <ul className="space-y-1">
                  {validationReport.checks.map((check: PluginValidationCheck) => (
                    <li
                      key={check.id}
                      className={clsx(
                        'flex items-start gap-2 rounded-md border px-3 py-2 text-sm',
                        check.passed
                          ? 'border-emerald-200 bg-emerald-50'
                          : 'border-red-200 bg-red-50'
                      )}
                    >
                      {check.passed ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                      ) : (
                        <XCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                      )}
                      <div>
                        <span className="font-medium">{check.id}</span>
                        <span className="text-slate-600"> — {check.detail}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {(logText || validationRunning) && (
              <div>
                <h3 className="text-sm font-semibold text-slate-800 mb-2">Validation log</h3>
                <pre
                  ref={logRef}
                  className="max-h-64 overflow-auto rounded-md bg-slate-900 text-slate-100 p-3 text-xs font-mono"
                >
                  {logText || 'Waiting for log output…'}
                </pre>
              </div>
            )}

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                className="rounded-md bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-900 disabled:opacity-50"
                disabled={validateMutation.isLoading || Boolean(validationRunning)}
                onClick={() => validateMutation.mutate(selectedName)}
              >
                Run validation
              </button>
              <button
                type="button"
                className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                disabled={!canActivate || activateMutation.isLoading}
                onClick={() => activateMutation.mutate(selectedName)}
              >
                Activate plugin
              </button>
            </div>
          </div>
        </Card>
      )}

      <Card title="Upload new plugin">
        <div className="grid gap-4 md:grid-cols-2">
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.name}>Name (slug)</FieldLabel>
            <input
              className={inputClass}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my_model"
            />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.version}>Version</FieldLabel>
            <input className={inputClass} value={version} onChange={(e) => setVersion(e.target.value)} />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.displayName}>Display name</FieldLabel>
            <input
              className={inputClass}
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
            />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.authors}>Authors</FieldLabel>
            <input
              className={inputClass}
              value={authors}
              onChange={(e) => setAuthors(e.target.value)}
              placeholder="comma-separated or JSON array"
            />
          </label>
          <label className="block md:col-span-2">
            <FieldLabel hint={PLUGIN_FORM_HINTS.description}>Description</FieldLabel>
            <textarea
              className={inputClass}
              rows={2}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.wrapperClass}>Wrapper class</FieldLabel>
            <input
              className={inputClass}
              value={wrapperClass}
              onChange={(e) => setWrapperClass(e.target.value)}
              placeholder="MyModelWrapper"
            />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.weightFormat}>Weight format</FieldLabel>
            <input
              className={inputClass}
              value={weightFormat}
              onChange={(e) => setWeightFormat(e.target.value)}
              placeholder="state_dict"
            />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.convertEntrypoint}>Convert entrypoint</FieldLabel>
            <input
              className={inputClass}
              value={convertEntrypoint}
              onChange={(e) => setConvertEntrypoint(e.target.value)}
            />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.peDbFormat}>PE-DB format (optional)</FieldLabel>
            <input
              className={inputClass}
              value={peDbFormat}
              onChange={(e) => setPeDbFormat(e.target.value)}
              placeholder="defaults to plugin name"
            />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.outputColumns}>Output columns</FieldLabel>
            <input
              className={inputClass}
              value={outputColumns}
              onChange={(e) => setOutputColumns(e.target.value)}
              placeholder="feature,Efficiency"
            />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.requiredStdColumns}>Required std columns</FieldLabel>
            <input
              className={inputClass}
              value={requiredStdColumns}
              onChange={(e) => setRequiredStdColumns(e.target.value)}
              placeholder="edit_len,editing_efficiency"
            />
          </label>
          <label className="block">
            <FieldLabel hint={PLUGIN_FORM_HINTS.labelColumn}>Label column</FieldLabel>
            <input
              className={inputClass}
              value={labelColumn}
              onChange={(e) => setLabelColumn(e.target.value)}
            />
          </label>
        </div>

        <div className="mt-6 space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <FieldLabel hint={PLUGIN_FORM_HINTS.hyperparameters}>Hyperparameters</FieldLabel>
              <button
                type="button"
                className="inline-flex items-center gap-1 text-sm text-primary-700 hover:underline"
                onClick={() => setHyperparameterRows((rows) => [...rows, emptyHyperparameter()])}
              >
                <Plus className="w-4 h-4" /> Add row
              </button>
            </div>
            <div className="mb-2 grid gap-2 md:grid-cols-4">
              <FieldLabel hint={PLUGIN_FORM_HINTS.hyperparameterName}>Name</FieldLabel>
              <FieldLabel hint={PLUGIN_FORM_HINTS.hyperparameterType}>Type</FieldLabel>
              <FieldLabel hint={PLUGIN_FORM_HINTS.hyperparameterDefault}>Default</FieldLabel>
              <span className="text-sm font-medium text-slate-500">Actions</span>
            </div>
            <div className="space-y-2">
              {hyperparameterRows.map((row, index) => (
                <div key={index} className="grid gap-2 md:grid-cols-4">
                  <input
                    className={inputClass}
                    placeholder="name"
                    value={row.name}
                    onChange={(e) => {
                      const next = [...hyperparameterRows]
                      next[index] = { ...row, name: e.target.value }
                      setHyperparameterRows(next)
                    }}
                  />
                  <input
                    className={inputClass}
                    placeholder="type"
                    value={row.type}
                    onChange={(e) => {
                      const next = [...hyperparameterRows]
                      next[index] = { ...row, type: e.target.value }
                      setHyperparameterRows(next)
                    }}
                  />
                  <input
                    className={inputClass}
                    placeholder="default"
                    value={row.default}
                    onChange={(e) => {
                      const next = [...hyperparameterRows]
                      next[index] = { ...row, default: e.target.value }
                      setHyperparameterRows(next)
                    }}
                  />
                  <button
                    type="button"
                    className="inline-flex items-center justify-center gap-1 rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50"
                    onClick={() =>
                      setHyperparameterRows((rows) => rows.filter((_, i) => i !== index))
                    }
                  >
                    <Trash2 className="w-4 h-4" />
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <FieldLabel hint={PLUGIN_FORM_HINTS.weightMetadata}>Weight metadata</FieldLabel>
              <button
                type="button"
                className="inline-flex items-center gap-1 text-sm text-primary-700 hover:underline"
                onClick={() => setWeightMetaRows((rows) => [...rows, emptyWeightMeta()])}
              >
                <Plus className="w-4 h-4" /> Add row
              </button>
            </div>
            <div className="mb-2 grid gap-2 md:grid-cols-3">
              <FieldLabel hint={PLUGIN_FORM_HINTS.weightId}>Weight id</FieldLabel>
              <FieldLabel hint={PLUGIN_FORM_HINTS.weightNotes}>Notes</FieldLabel>
              <span className="text-sm font-medium text-slate-500">Actions</span>
            </div>
            <div className="space-y-2">
              {weightMetaRows.map((row, index) => (
                <div key={index} className="grid gap-2 md:grid-cols-3">
                  <input
                    className={inputClass}
                    placeholder="weight id"
                    value={row.id}
                    onChange={(e) => {
                      const next = [...weightMetaRows]
                      next[index] = { ...row, id: e.target.value }
                      setWeightMetaRows(next)
                    }}
                  />
                  <input
                    className={inputClass}
                    placeholder="notes"
                    value={row.notes}
                    onChange={(e) => {
                      const next = [...weightMetaRows]
                      next[index] = { ...row, notes: e.target.value }
                      setWeightMetaRows(next)
                    }}
                  />
                  <button
                    type="button"
                    className="inline-flex items-center justify-center gap-1 rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50"
                    onClick={() =>
                      setWeightMetaRows((rows) => rows.filter((_, i) => i !== index))
                    }
                  >
                    <Trash2 className="w-4 h-4" />
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={useZipBundle}
                onChange={(e) => setUseZipBundle(e.target.checked)}
              />
              <FieldLabel hint={PLUGIN_FORM_HINTS.useZipBundle}>
                Upload as zip bundle instead of separate files
              </FieldLabel>
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={replaceExisting}
                onChange={(e) => setReplaceExisting(e.target.checked)}
              />
              <FieldLabel hint={PLUGIN_FORM_HINTS.replaceExisting}>
                Replace existing pending/rejected plugin with same name
              </FieldLabel>
            </label>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {useZipBundle ? (
              <label className="block">
                <FieldLabel hint={PLUGIN_FORM_HINTS.bundleZip}>Bundle zip</FieldLabel>
                <input
                  type="file"
                  accept=".zip"
                  className={inputClass}
                  onChange={(e) => setBundleZip(e.target.files?.[0] ?? null)}
                />
              </label>
            ) : (
              <>
                <label className="block">
                  <FieldLabel hint={PLUGIN_FORM_HINTS.convertFile}>convert.py</FieldLabel>
                  <input
                    type="file"
                    accept=".py"
                    className={inputClass}
                    onChange={(e) => setConvertFile(e.target.files?.[0] ?? null)}
                  />
                </label>
                <label className="block">
                  <FieldLabel hint={PLUGIN_FORM_HINTS.wrapperFile}>wrapper.py</FieldLabel>
                  <input
                    type="file"
                    accept=".py"
                    className={inputClass}
                    onChange={(e) => setWrapperFile(e.target.files?.[0] ?? null)}
                  />
                </label>
              </>
            )}
            <label className="block">
              <FieldLabel hint={PLUGIN_FORM_HINTS.weightIdUpload}>Weight id (optional)</FieldLabel>
              <input
                className={inputClass}
                value={weightId}
                onChange={(e) => setWeightId(e.target.value)}
                placeholder="base_weights"
              />
            </label>
            <label className="block">
              <FieldLabel hint={PLUGIN_FORM_HINTS.weightFileUpload}>Weight file (optional)</FieldLabel>
              <input
                type="file"
                className={inputClass}
                onChange={(e) => setWeightFile(e.target.files?.[0] ?? null)}
              />
            </label>
          </div>

          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-md bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
            disabled={uploadMutation.isLoading}
            onClick={handleUpload}
          >
            <Upload className="w-4 h-4" />
            {uploadMutation.isLoading ? 'Uploading…' : 'Upload plugin'}
          </button>
        </div>
      </Card>
    </div>
  )
}
