import { useState } from 'react'
import { Trash2 } from 'lucide-react'
import Card from '@components/Card'
import KillJobConfirmDialog, {
  persistSkipKillJobConfirm,
  shouldSkipKillJobConfirm,
} from '@components/KillJobConfirmDialog'

export interface ComputeJobListItem {
  job_id: string
  status: string
  model_name?: string
  dataset_name?: string
  benchmark_name?: string
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
}

function statusLabel(status: string, queuePosition?: number | null) {
  if (status === 'queued' && queuePosition) return `queued (#${queuePosition})`
  return status
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
}: ComputeJobListProps) {
  const [pendingKill, setPendingKill] = useState<{ jobId: string; label: string } | null>(null)
  const [killingJobId, setKillingJobId] = useState<string | null>(null)

  const executeKill = async (jobId: string) => {
    setKillingJobId(jobId)
    try {
      await killJob(jobId)
      onJobKilled(jobId)
      setPendingKill(null)
    } catch (err: unknown) {
      const message =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        (err as Error)?.message ||
        'Failed to delete job'
      onKillError(message)
    } finally {
      setKillingJobId(null)
    }
  }

  const requestKill = (job: ComputeJobListItem) => {
    const label = getJobTitle(job)
    if (shouldSkipKillJobConfirm()) {
      void executeKill(job.job_id)
      return
    }
    setPendingKill({ jobId: job.job_id, label })
  }

  return (
    <>
      <Card title="Jobs">
        {!jobs?.length ? (
          <p className="text-slate-500 py-4 text-center text-sm">{emptyMessage}</p>
        ) : (
          <ul className="divide-y divide-slate-200 max-h-96 overflow-y-auto">
            {jobs.map((job) => (
              <li key={job.job_id} className="flex items-stretch">
                <button
                  type="button"
                  onClick={() => onSelectJob(job.job_id)}
                  className={`flex-1 text-left px-3 py-2 text-sm hover:bg-slate-50 ${
                    selectedJobId === job.job_id
                      ? 'bg-primary-50 border-l-4 border-primary-500'
                      : ''
                  }`}
                >
                  <p className="font-medium truncate">{getJobTitle(job)}</p>
                  <p className="text-slate-500 text-xs">
                    {statusLabel(job.status, job.queue_position)}
                    {job.device_assigned ? ` on ${job.device_assigned}` : ''}
                  </p>
                </button>
                <button
                  type="button"
                  title="Kill and delete job"
                  aria-label={`Kill and delete ${getJobTitle(job)}`}
                  disabled={killingJobId === job.job_id}
                  onClick={(event) => {
                    event.stopPropagation()
                    requestKill(job)
                  }}
                  className="px-3 text-slate-400 hover:text-red-600 hover:bg-red-50 disabled:opacity-50"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <KillJobConfirmDialog
        open={pendingKill !== null}
        jobLabel={pendingKill?.label ?? ''}
        isLoading={killingJobId !== null}
        onCancel={() => {
          if (killingJobId === null) {
            setPendingKill(null)
          }
        }}
        onConfirm={(neverShowAgain) => {
          if (neverShowAgain) {
            persistSkipKillJobConfirm(true)
          }
          if (pendingKill) {
            void executeKill(pendingKill.jobId)
          }
        }}
      />
    </>
  )
}
