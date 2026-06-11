export const ACTIVE_JOB_STATUSES = new Set(['queued', 'running', 'stopping'])
export const TERMINAL_JOB_STATUSES = new Set(['succeeded', 'failed', 'cancelled', 'skipped'])

export function jobStatusRefetchInterval(
  data: { data?: { status?: string } } | undefined
): number | false {
  const status = data?.data?.status
  if (!status) return 2000
  return ACTIVE_JOB_STATUSES.has(status) ? 2000 : false
}

export function sortJobsByCreatedAt<T extends { created_at?: string }>(jobs: T[]): T[] {
  return [...jobs].sort((left, right) => {
    const leftTime = left.created_at ? Date.parse(left.created_at) : 0
    const rightTime = right.created_at ? Date.parse(right.created_at) : 0
    return rightTime - leftTime
  })
}

export function formatJobCreatedAtLocal(createdAt?: string): string {
  if (!createdAt) return ''
  const date = new Date(createdAt)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString()
}

export function formatJobListTitle(label: string, createdAt?: string): string {
  const timestamp = formatJobCreatedAtLocal(createdAt)
  return timestamp ? `${timestamp} · ${label}` : label
}
