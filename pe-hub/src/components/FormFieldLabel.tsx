import { Info } from 'lucide-react'
import clsx from 'clsx'

interface FormFieldLabelProps {
  children: string
  hint: string
  className?: string
}

export default function FormFieldLabel({ children, hint, className }: FormFieldLabelProps) {
  return (
    <span className={clsx('inline-flex items-center gap-1.5', className)}>
      {children}
      <span className="relative inline-flex group">
        <button
          type="button"
          className="inline-flex rounded-full text-slate-400 hover:text-slate-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
          aria-label={`More information about ${children}`}
        >
          <Info className="w-4 h-4 shrink-0" aria-hidden />
        </button>
        <span
          role="tooltip"
          className="pointer-events-none absolute left-0 bottom-full z-30 mb-2 w-72 max-w-[min(18rem,calc(100vw-2rem))] rounded-md bg-slate-800 px-3 py-2 text-xs font-normal leading-snug text-white opacity-0 shadow-lg transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100"
        >
          {hint}
        </span>
      </span>
    </span>
  )
}
