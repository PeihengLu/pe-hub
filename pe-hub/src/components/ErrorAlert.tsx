import { AlertCircle } from 'lucide-react'

interface ErrorAlertProps {
  message: string
  onDismiss?: () => void
}

export default function ErrorAlert({ message, onDismiss }: ErrorAlertProps) {
  return (
    <div className="bg-red-950/40 border border-red-800/50 rounded-lg p-4 flex items-start gap-3">
      <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        <p className="text-red-300 font-medium">Error</p>
        <p className="text-red-300 text-sm mt-1">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-red-400 hover:text-red-300 flex-shrink-0"
        >
          ✕
        </button>
      )}
    </div>
  )
}
