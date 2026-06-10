import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown } from 'lucide-react'
import clsx from 'clsx'

export interface SelectMenuOption {
  value: string
  label: string
  disabled?: boolean
}

interface SelectMenuProps {
  value: string
  onChange: (value: string) => void
  options: SelectMenuOption[]
  disabled?: boolean
  placeholder?: string
  className?: string
  'aria-label'?: string
}

/**
 * In-document dropdown rendered in a portal. Avoids native &lt;select&gt; popups that
 * misalign on Linux/WSL and escapes overflow clipping from parent cards.
 */
export default function SelectMenu({
  value,
  onChange,
  options,
  disabled = false,
  placeholder = 'Select…',
  className,
  'aria-label': ariaLabel,
}: SelectMenuProps) {
  const [open, setOpen] = useState(false)
  const [menuStyle, setMenuStyle] = useState<CSSProperties>({})
  const rootRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLUListElement>(null)
  const listId = useId()
  const selected = options.find((option) => option.value === value)

  const updateMenuPosition = () => {
    const button = buttonRef.current
    if (!button) return
    const rect = button.getBoundingClientRect()
    setMenuStyle({
      position: 'fixed',
      top: rect.bottom + 4,
      left: rect.left,
      width: rect.width,
      zIndex: 9999,
    })
  }

  useLayoutEffect(() => {
    if (!open) return undefined
    updateMenuPosition()
    const onLayoutChange = () => updateMenuPosition()
    window.addEventListener('resize', onLayoutChange)
    window.addEventListener('scroll', onLayoutChange, true)
    return () => {
      window.removeEventListener('resize', onLayoutChange)
      window.removeEventListener('scroll', onLayoutChange, true)
    }
  }, [open])

  useEffect(() => {
    if (!open) return undefined
    const onPointerDown = (event: MouseEvent) => {
      const target = event.target as Node
      if (rootRef.current?.contains(target) || menuRef.current?.contains(target)) return
      setOpen(false)
    }
    // Defer so the opening click does not immediately close the menu.
    const timer = window.setTimeout(() => {
      document.addEventListener('mousedown', onPointerDown)
    }, 0)
    return () => {
      window.clearTimeout(timer)
      document.removeEventListener('mousedown', onPointerDown)
    }
  }, [open])

  useEffect(() => {
    if (!open) return undefined
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [open])

  const choose = (next: string) => {
    onChange(next)
    setOpen(false)
  }

  const menu =
    open &&
    createPortal(
      <ul
        ref={menuRef}
        id={listId}
        role="listbox"
        aria-label={ariaLabel}
        style={menuStyle}
        className="max-h-60 overflow-y-auto rounded-lg border border-slate-200 bg-white py-1 shadow-lg"
      >
        {options.length === 0 ? (
          <li className="px-3 py-2 text-sm text-slate-500">No options available</li>
        ) : (
          options.map((option) => (
            <li key={option.value} role="option" aria-selected={option.value === value}>
              <button
                type="button"
                disabled={option.disabled}
                onClick={() => choose(option.value)}
                className={clsx(
                  'w-full px-3 py-2 text-left text-sm',
                  option.disabled
                    ? 'cursor-not-allowed text-slate-400'
                    : option.value === value
                      ? 'bg-primary-50 text-primary-800'
                      : 'text-slate-900 hover:bg-slate-50'
                )}
              >
                {option.label}
              </button>
            </li>
          ))
        )}
      </ul>,
      document.body
    )

  return (
    <div ref={rootRef} className={clsx('relative', className)}>
      <button
        ref={buttonRef}
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        aria-label={ariaLabel}
        onClick={() => {
          if (!disabled) {
            if (!open) updateMenuPosition()
            setOpen((prev) => !prev)
          }
        }}
        className={clsx(
          'flex w-full items-center justify-between gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-left text-sm',
          disabled ? 'cursor-not-allowed bg-slate-100 text-slate-500' : 'hover:border-slate-400'
        )}
      >
        <span className={clsx('truncate', !selected && 'text-slate-400')}>
          {selected?.label ?? placeholder}
        </span>
        <ChevronDown
          className={clsx('h-4 w-4 shrink-0 text-slate-500 transition-transform', open && 'rotate-180')}
        />
      </button>
      {menu}
    </div>
  )
}
