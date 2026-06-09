export const ACTIVE_JOB_STATUSES = new Set(['queued', 'running'])
export const TERMINAL_JOB_STATUSES = new Set(['succeeded', 'failed', 'cancelled'])

export function jobStatusRefetchInterval(
  data: { data?: { status?: string } } | undefined
): number | false {
  const status = data?.data?.status
  if (!status) return 2000
  return ACTIVE_JOB_STATUSES.has(status) ? 2000 : false
}
