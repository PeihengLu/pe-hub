import { AlertCircle } from 'lucide-react'

interface ErrorAlertProps {
  message: string
  onDismiss?: () => void
}

export default function ErrorAlert({ message, onDismiss }: ErrorAlertProps) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
      <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        <p className="text-red-800 font-medium">Error</p>
        <p className="text-red-700 text-sm mt-1">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-red-600 hover:text-red-700 flex-shrink-0"
        >
          ✕
        </button>
      )}
    </div>
  )
}
