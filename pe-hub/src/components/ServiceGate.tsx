import type { ReactNode } from 'react'
import { type ServiceId } from '@config/services'
import { useServiceHealth } from '@context/ServiceHealthProvider'
import LoadingSpinner from '@components/LoadingSpinner'
import ServiceUnavailable from '@components/ServiceUnavailable'

interface ServiceGateProps {
  serviceId: ServiceId
  children: ReactNode
}

export default function ServiceGate({ serviceId, children }: ServiceGateProps) {
  const { health } = useServiceHealth()
  const state = health[serviceId]

  if (state.status === 'checking') {
    return <LoadingSpinner message={`Connecting to ${serviceId}...`} />
  }

  if (state.status === 'down') {
    return <ServiceUnavailable serviceId={serviceId} />
  }

  return <>{children}</>
}
