import clsx from 'clsx'

interface CardProps {
  children: React.ReactNode
  className?: string
  title?: string
}

export default function Card({ children, className, title }: CardProps) {
  return (
    <div className={clsx('bg-surface rounded-lg shadow-md border border-line', className)}>
      {title && (
        <h2 className="px-6 py-4 text-xl font-bold text-ink border-b border-line">{title}</h2>
      )}
      <div className="p-6">{children}</div>
    </div>
  )
}
