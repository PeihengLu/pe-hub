import { useEffect, useState } from 'react'

const SKIP_CONFIRM_STORAGE_KEY = 'pe-hub.kill-job.skip-confirm'

export function shouldSkipKillJobConfirm(): boolean {
  try {
    return localStorage.getItem(SKIP_CONFIRM_STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

export function persistSkipKillJobConfirm(skip: boolean): void {
  try {
    if (skip) {
      localStorage.setItem(SKIP_CONFIRM_STORAGE_KEY, '1')
    } else {
      localStorage.removeItem(SKIP_CONFIRM_STORAGE_KEY)
    }
  } catch {
    // ignore storage errors
  }
}

interface KillJobConfirmDialogProps {
  open: boolean
  jobLabel: string
  isLoading?: boolean
  onConfirm: (neverShowAgain: boolean) => void
  onCancel: () => void
}

export default function KillJobConfirmDialog({
  open,
  jobLabel,
  isLoading = false,
  onConfirm,
  onCancel,
}: KillJobConfirmDialogProps) {
  const [neverShowAgain, setNeverShowAgain] = useState(false)

  useEffect(() => {
    if (open) {
      setNeverShowAgain(false)
    }
  }, [open])

  if (!open) {
    return null
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <button
        type="button"
        aria-label="Close dialog"
        className="absolute inset-0 bg-slate-900/50"
        onClick={isLoading ? undefined : onCancel}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="kill-job-dialog-title"
        className="relative w-full max-w-md rounded-xl border border-slate-200 bg-white p-6 shadow-xl"
      >
        <h3 id="kill-job-dialog-title" className="text-lg font-semibold text-slate-900">
          Kill and delete job?
        </h3>
        <p className="mt-3 text-sm text-slate-600">
          This will stop <span className="font-medium text-slate-900">{jobLabel}</span> if it is
          still queued or running, then permanently delete its log files and other artifacts from
          this run.
        </p>
        <p className="mt-2 text-sm text-slate-600">
          Registered model weights from a completed training job are kept in the weights registry
          and are not removed.
        </p>
        <label className="mt-4 inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
          <input
            type="checkbox"
            checked={neverShowAgain}
            onChange={(e) => setNeverShowAgain(e.target.checked)}
            disabled={isLoading}
          />
          Don&apos;t show this again
        </label>
        <div className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={isLoading}
            className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onConfirm(neverShowAgain)}
            disabled={isLoading}
            className="rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 disabled:bg-slate-400"
          >
            {isLoading ? 'Deleting…' : 'Kill & delete'}
          </button>
        </div>
      </div>
    </div>
  )
}
