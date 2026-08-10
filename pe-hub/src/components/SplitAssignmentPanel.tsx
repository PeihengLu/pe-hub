import {
  SPLIT_STRATEGIES,
  type SplitStrategy,
} from '@apps/database/config/exportAttributes'

interface SplitAssignmentPanelProps {
  splitStrategy: SplitStrategy
  onSplitStrategyChange: (value: SplitStrategy) => void
  trainPct: string
  onTrainPctChange: (value: string) => void
  valPct: string
  onValPctChange: (value: string) => void
  testPct: string
  onTestPctChange: (value: string) => void
  cvFolds: string
  onCvFoldsChange: (value: string) => void
  useOriginalFold: boolean
  onUseOriginalFoldChange: (value: boolean) => void
  originalFoldTestValue: string
  onOriginalFoldTestValueChange: (value: string) => void
  splitRandomState: string
  onSplitRandomStateChange: (value: string) => void
  description?: string
  excludeNone?: boolean
}

export default function SplitAssignmentPanel({
  splitStrategy,
  onSplitStrategyChange,
  trainPct,
  onTrainPctChange,
  valPct,
  onValPctChange,
  testPct,
  onTestPctChange,
  cvFolds,
  onCvFoldsChange,
  useOriginalFold,
  onUseOriginalFoldChange,
  originalFoldTestValue,
  onOriginalFoldTestValueChange,
  splitRandomState,
  onSplitRandomStateChange,
  description,
  excludeNone = false,
}: SplitAssignmentPanelProps) {
  const strategies = excludeNone
    ? SPLIT_STRATEGIES.filter((item) => item.value !== 'none')
    : SPLIT_STRATEGIES

  return (
    <div className="space-y-4">
      {description ? (
        <p className="text-sm text-slate-600">{description}</p>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        {strategies.map((item) => (
          <label
            key={item.value}
            className={`cursor-pointer rounded-lg border p-4 transition-all ${
              splitStrategy === item.value
                ? 'border-primary-500 bg-primary-50 ring-1 ring-primary-500'
                : 'border-slate-200 hover:border-slate-300'
            }`}
          >
            <div className="flex items-start gap-3">
              <input
                type="radio"
                name="split-strategy"
                value={item.value}
                checked={splitStrategy === item.value}
                onChange={() => onSplitStrategyChange(item.value)}
                className="mt-1"
              />
              <div>
                <p className="font-medium text-slate-900">{item.label}</p>
                <p className="text-xs text-slate-500 mt-1">{item.description}</p>
              </div>
            </div>
          </label>
        ))}
      </div>

      {(splitStrategy === 'holdout_2' || splitStrategy === 'holdout_3') && (
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-sm text-slate-700">
            Train %
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={trainPct}
              onChange={(e) => onTrainPctChange(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
            />
          </label>
          {splitStrategy === 'holdout_3' && (
            <label className="text-sm text-slate-700">
              Val %
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={valPct}
                onChange={(e) => onValPctChange(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
              />
            </label>
          )}
          <label className="text-sm text-slate-700">
            Test %
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={testPct}
              onChange={(e) => onTestPctChange(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
            />
          </label>
        </div>
      )}

      {splitStrategy === 'cv' && (
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="text-sm text-slate-700">
            CV folds
            <input
              type="number"
              min={2}
              step={1}
              value={cvFolds}
              onChange={(e) => onCvFoldsChange(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
            />
          </label>
          <label className="text-sm text-slate-700">
            Test % (optional holdout)
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={testPct}
              onChange={(e) => onTestPctChange(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
              placeholder="Leave empty for no holdout"
            />
          </label>
        </div>
      )}

      {splitStrategy !== 'none' && (
        <div className="space-y-3">
          <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={useOriginalFold}
              onChange={(e) => onUseOriginalFoldChange(e.target.checked)}
            />
            Use author original_fold when available
          </label>
          {useOriginalFold && (
            <label className="block text-sm text-slate-700 max-w-xs">
              Designate test fold
              <input
                type="number"
                step={1}
                value={originalFoldTestValue}
                onChange={(e) => onOriginalFoldTestValueChange(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
              />
              <span className="mt-1 block text-xs text-slate-500">
                For Train / test / holdout, rows with this original_fold value become the test
                partition; remaining rows are split into train and val. For CV, this is optional:
                when unset or unmatched, all author CV folds are used for evaluation. Use -1 for
                DeepPrime-style held-out test; use 0–4 for PRIDICT2 to hold out a single fold when
                needed (match run_N weights).
              </span>
            </label>
          )}
        </div>
      )}

      {splitStrategy !== 'none' && (
        <label className="block text-sm text-slate-700 max-w-xs">
          Random seed
          <input
            type="number"
            min={0}
            step={1}
            value={splitRandomState}
            onChange={(e) => onSplitRandomStateChange(e.target.value)}
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
          />
        </label>
      )}
    </div>
  )
}
