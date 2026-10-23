import { useEffect, useMemo, useState } from 'react'
import { Download, Trash2 } from 'lucide-react'
import Card from '@components/Card'
import KillJobConfirmDialog, {
  persistSkipKillJobConfirm,
  shouldSkipKillJobConfirm,
} from '@components/KillJobConfirmDialog'
import { formatJobListTitle, sortJobsByCreatedAt } from '@apps/ensemble/utils/jobStatus'

export interface ComputeJobListItem {
  job_id: string
  status: string
  created_at?: string
  model_name?: string
  dataset_name?: string
  benchmark_name?: string
  ensemble_name?: string
  kind?: string
  combine?: string
  member_count?: number
  queue_position?: number | null
  device_assigned?: string | null
}

interface ComputeJobListProps {
  jobs: ComputeJobListItem[] | undefined
  selectedJobId: string | null
  onSelectJob: (jobId: string) => void
  onJobKilled: (jobId: string) => void
  onKillError: (message: string) => void
  getJobTitle: (job: ComputeJobListItem) => string
  killJob: (jobId: string) => Promise<unknown>
  emptyMessage?: string
  onExportSelected?: (selectedJobIds: string[]) => void | Promise<void>
  exportSelectedLoading?: boolean
}

interface PendingKill {
  jobIds: string[]
  label: string
}

function statusLabel(status: string, queuePosition?: number | null) {
  if (status === 'queued' && queuePosition) return `queued (#${queuePosition})`
  if (status === 'stopping') return 'stopping…'
  return status
}

function isJobSelectable(job: ComputeJobListItem): boolean {
  return job.status !== 'stopping'
}

function buildKillLabel(jobs: ComputeJobListItem[], getJobTitle: (job: ComputeJobListItem) => string): string {
  if (jobs.length === 1) {
    const job = jobs[0]
    return formatJobListTitle(getJobTitle(job), job.created_at)
  }
  const preview = jobs.slice(0, 3).map((job) => formatJobListTitle(getJobTitle(job), job.created_at))
  if (jobs.length > 3) {
    preview.push(`…and ${jobs.length - 3} more`)
  }
  return preview.join('\n')
}

export default function ComputeJobList({
  jobs,
  selectedJobId,
  onSelectJob,
  onJobKilled,
  onKillError,
  getJobTitle,
  killJob,
  emptyMessage = 'No jobs yet.',
  onExportSelected,
  exportSelectedLoading = false,
}: ComputeJobListProps) {
  const [selectedJobIds, setSelectedJobIds] = useState<Set<string>>(() => new Set())
  const [pendingKill, setPendingKill] = useState<PendingKill | null>(null)
  const [killingJobIds, setKillingJobIds] = useState<Set<string>>(() => new Set())
  const sortedJobs = useMemo(() => sortJobsByCreatedAt(jobs ?? []), [jobs])
  const selectableJobs = useMemo(() => sortedJobs.filter(isJobSelectable), [sortedJobs])
  const visibleJobIds = useMemo(() => new Set(sortedJobs.map((job) => job.job_id)), [sortedJobs])

  useEffect(() => {
    setSelectedJobIds((current) => {
      const next = new Set(
        [...current].filter((jobId) => {
          const job = sortedJobs.find((entry) => entry.job_id === jobId)
          return job !== undefined && visibleJobIds.has(jobId) && isJobSelectable(job)
        })
      )
      return next.size === current.size ? current : next
    })
  }, [visibleJobIds, sortedJobs])

  const selectedCount = selectedJobIds.size
  const allSelected =
    selectableJobs.length > 0 && selectableJobs.every((job) => selectedJobIds.has(job.job_id))
  const isKilling = killingJobIds.size > 0

  const jobLabelFor = (job: ComputeJobListItem) => formatJobListTitle(getJobTitle(job), job.created_at)

  const toggleJobSelected = (jobId: string) => {
    setSelectedJobIds((current) => {
      const next = new Set(current)
      if (next.has(jobId)) {
        next.delete(jobId)
      } else {
        next.add(jobId)
      }
      return next
    })
  }

  const toggleSelectAll = () => {
    if (allSelected) {
      setSelectedJobIds(new Set())
      return
    }
    setSelectedJobIds(new Set(selectableJobs.map((job) => job.job_id)))
  }

  const executeKill = async (jobIds: string[]) => {
    if (jobIds.length === 0) {
      return
    }

    setKillingJobIds(new Set(jobIds))
    const failures: string[] = []

    const results = await Promise.allSettled(
      jobIds.map(async (jobId) => {
        await killJob(jobId)
        onJobKilled(jobId)
        setSelectedJobIds((current) => {
          if (!current.has(jobId)) {
            return current
          }
          const next = new Set(current)
          next.delete(jobId)
          return next
        })
      })
    )

    for (const result of results) {
      if (result.status === 'rejected') {
        const err = result.reason
        const message =
          (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
          (err as Error)?.message ||
          'Failed to delete job'
        failures.push(message)
      }
    }

    if (failures.length > 0) {
      onKillError(
        failures.length === 1
          ? failures[0]
          : `Failed to delete ${failures.length} jobs. ${failures[0]}`
      )
    }

    setKillingJobIds(new Set())
  }

  const requestKill = (targetJobs: ComputeJobListItem[]) => {
    if (targetJobs.length === 0) {
      return
    }

    const jobIds = targetJobs.map((job) => job.job_id)
    const label = buildKillLabel(targetJobs, getJobTitle)

    if (shouldSkipKillJobConfirm()) {
      void executeKill(jobIds)
      return
    }

    setPendingKill({ jobIds, label })
  }

  const requestKillJob = (job: ComputeJobListItem) => {
    requestKill([job])
  }

  const requestKillSelected = () => {
    const targetJobs = sortedJobs.filter((job) => selectedJobIds.has(job.job_id))
    requestKill(targetJobs)
  }

  return (
    <>
      <Card title="Jobs">
        {!sortedJobs.length ? (
          <p className="text-ink-faint py-4 text-center text-sm">{emptyMessage}</p>
        ) : (
          <>
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-line pb-3">
              <label className="inline-flex items-center gap-2 text-sm text-ink-muted cursor-pointer">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={toggleSelectAll}
                  disabled={isKilling}
                  className="rounded border-line-strong"
                />
                <span>{allSelected ? 'Deselect all' : 'Select all'}</span>
                {selectedCount > 0 && (
                  <span className="text-ink-faint">({selectedCount} selected)</span>
                )}
              </label>
              {selectedCount > 0 && (
                <div className="flex flex-wrap items-center gap-2">
                  {onExportSelected && (
                    <button
                      type="button"
                      onClick={() => {
                        void onExportSelected([...selectedJobIds])
                      }}
                      disabled={isKilling || exportSelectedLoading}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm font-medium text-ink-muted hover:bg-canvas disabled:opacity-50"
                    >
                      <Download className="w-4 h-4" />
                      {exportSelectedLoading ? 'Exporting…' : `Export JSON (${selectedCount})`}
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={requestKillSelected}
                    disabled={isKilling}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-red-800/50 bg-red-950/40 px-3 py-1.5 text-sm font-medium text-red-300 hover:bg-red-950/50 disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete selected ({selectedCount})
                  </button>
                </div>
              )}
            </div>

            <ul className="divide-y divide-line max-h-96 overflow-y-auto">
              {sortedJobs.map((job) => {
                const jobLabel = jobLabelFor(job)
                const isRowSelected = selectedJobIds.has(job.job_id)
                const isRowKilling = killingJobIds.has(job.job_id)
                const isRowSelectable = isJobSelectable(job)

                return (
                  <li key={job.job_id} className="flex items-stretch">
                    <label
                      className={`flex items-center px-3 ${
                        isRowSelectable ? 'cursor-pointer' : 'cursor-not-allowed'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isRowSelected}
                        onChange={() => toggleJobSelected(job.job_id)}
                        disabled={isKilling || !isRowSelectable}
                        aria-label={`Select ${jobLabel}`}
                        className="rounded border-line-strong disabled:opacity-40"
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() => onSelectJob(job.job_id)}
                      className={`flex-1 text-left px-3 py-2 text-sm hover:bg-canvas ${
                        selectedJobId === job.job_id
                          ? 'bg-primary-50 border-l-4 border-primary-500'
                          : ''
                      }`}
                    >
                      <p className="font-medium truncate">{jobLabel}</p>
                      <p className="text-ink-faint text-xs">
                        {statusLabel(job.status, job.queue_position)}
                        {job.device_assigned ? ` on ${job.device_assigned}` : ''}
                      </p>
                    </button>
                    <button
                      type="button"
                      title="Kill and delete job"
                      aria-label={`Kill and delete ${jobLabel}`}
                      disabled={isRowKilling || !isRowSelectable}
                      onClick={(event) => {
                        event.stopPropagation()
                        requestKillJob(job)
                      }}
                      className="px-3 text-ink-faint hover:text-red-400 hover:bg-red-950/40 disabled:opacity-50"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </li>
                )
              })}
            </ul>
          </>
        )}
      </Card>

      <KillJobConfirmDialog
        open={pendingKill !== null}
        jobLabel={pendingKill?.label ?? ''}
        jobCount={pendingKill?.jobIds.length ?? 1}
        onCancel={() => setPendingKill(null)}
        onConfirm={(neverShowAgain) => {
          if (neverShowAgain) {
            persistSkipKillJobConfirm(true)
          }
          const jobIds = pendingKill?.jobIds ?? []
          setPendingKill(null)
          if (jobIds.length > 0) {
            void executeKill(jobIds)
          }
        }}
      />
    </>
  )
}
