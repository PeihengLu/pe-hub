import clsx from 'clsx'

interface CardProps {
  children: React.ReactNode
  className?: string
  title?: string
}

export default function Card({ children, className, title }: CardProps) {
  return (
    <div className={clsx('bg-white rounded-lg shadow-md border border-slate-200', className)}>
      {title && <h2 className="px-6 py-4 text-xl font-bold border-b border-slate-200">{title}</h2>}
      <div className="p-6">{children}</div>
    </div>
  )
}
