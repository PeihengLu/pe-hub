import { useMemo, type ReactNode } from 'react'

export type SchedulerName = 'none' | 'step' | 'cosine' | 'exponential'

export interface HyperparameterFormState {
  epochs: string
  batchSize: string
  learningRate: string
  weightDecay: string
  gradClip: string
  scheduler: SchedulerName
  schedulerStepSize: string
  schedulerGamma: string
  schedulerTMax: string
  earlyStoppingPatience: string
  earlyStoppingMinDelta: string
  loadPretrained: boolean
  trainEnsemble: boolean
  reshuffleEachEpoch: boolean
  pridictLossFunc: string
  pridictNumEpochs: string
  opedEpochNum: string
}

export const DEFAULT_HYPERPARAMETERS: HyperparameterFormState = {
  epochs: '5',
  batchSize: '128',
  learningRate: '0.0001',
  weightDecay: '0',
  gradClip: '1',
  scheduler: 'none',
  schedulerStepSize: '10',
  schedulerGamma: '0.95',
  schedulerTMax: '10',
  earlyStoppingPatience: '10',
  earlyStoppingMinDelta: '0',
  loadPretrained: true,
  trainEnsemble: false,
  reshuffleEachEpoch: true,
  pridictLossFunc: 'KLDloss',
  pridictNumEpochs: '20',
  opedEpochNum: '100',
}

function Field({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <label className="block text-sm text-slate-700">
      {label}
      <div className="mt-1">{children}</div>
    </label>
  )
}

const inputClass =
  'w-full rounded-md border border-slate-300 px-3 py-2 text-sm'

interface TrainingHyperparametersPanelProps {
  modelName: string
  values: HyperparameterFormState
  onChange: (values: HyperparameterFormState) => void
}

export function buildHyperparametersPayload(
  modelName: string,
  values: HyperparameterFormState
): Record<string, unknown> {
  const schedulerName = values.scheduler === 'none' ? undefined : values.scheduler
  const scheduler_kwargs =
    values.scheduler === 'step'
      ? {
          step_size: Number(values.schedulerStepSize),
          gamma: Number(values.schedulerGamma),
        }
      : values.scheduler === 'cosine'
        ? {
            t_max: Number(values.schedulerTMax),
            eta_min: 0,
          }
        : values.scheduler === 'exponential'
          ? { gamma: Number(values.schedulerGamma) }
          : undefined

  const common = {
    batch_size: Number(values.batchSize),
    lr: Number(values.learningRate),
    weight_decay: Number(values.weightDecay),
    grad_clip: Number(values.gradClip),
    early_stopping_patience: Number(values.earlyStoppingPatience),
    early_stopping_min_delta: Number(values.earlyStoppingMinDelta),
    reshuffle_each_epoch: values.reshuffleEachEpoch,
    ...(schedulerName ? { scheduler: schedulerName, scheduler_kwargs } : { scheduler: 'none' }),
  }

  if (modelName === 'deepprime') {
    return {
      ...common,
      epochs: Number(values.epochs),
      load_pretrained: values.loadPretrained,
      train_ensemble: values.trainEnsemble,
    }
  }

  if (modelName === 'oped') {
    return {
      ...common,
      epoch_num: Number(values.opedEpochNum),
      scheduler: schedulerName ?? 'step',
      scheduler_kwargs:
        scheduler_kwargs ?? { step_size: Number(values.schedulerStepSize), gamma: Number(values.schedulerGamma) },
    }
  }

  if (modelName === 'pridict2') {
    return {
      ...common,
      num_epochs: Number(values.pridictNumEpochs),
      loss_func: values.pridictLossFunc,
    }
  }

  return common
}

export default function TrainingHyperparametersPanel({
  modelName,
  values,
  onChange,
}: TrainingHyperparametersPanelProps) {
  const patch = (partial: Partial<HyperparameterFormState>) => {
    onChange({ ...values, ...partial })
  }

  const schedulerFields = useMemo(() => {
    if (values.scheduler === 'step') {
      return (
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Step size">
            <input
              type="number"
              min={1}
              value={values.schedulerStepSize}
              onChange={(e) => patch({ schedulerStepSize: e.target.value })}
              className={inputClass}
            />
          </Field>
          <Field label="Gamma">
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={values.schedulerGamma}
              onChange={(e) => patch({ schedulerGamma: e.target.value })}
              className={inputClass}
            />
          </Field>
        </div>
      )
    }
    if (values.scheduler === 'cosine') {
      return (
        <Field label="T max">
          <input
            type="number"
            min={1}
            value={values.schedulerTMax}
            onChange={(e) => patch({ schedulerTMax: e.target.value })}
            className={inputClass}
          />
        </Field>
      )
    }
    if (values.scheduler === 'exponential') {
      return (
        <Field label="Gamma">
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={values.schedulerGamma}
            onChange={(e) => patch({ schedulerGamma: e.target.value })}
            className={inputClass}
          />
        </Field>
      )
    }
    return null
  }, [values])

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        {modelName === 'pridict2' ? (
          <Field label="Epochs">
            <input
              type="number"
              min={1}
              value={values.pridictNumEpochs}
              onChange={(e) => patch({ pridictNumEpochs: e.target.value })}
              className={inputClass}
            />
          </Field>
        ) : modelName === 'oped' ? (
          <Field label="Epochs">
            <input
              type="number"
              min={1}
              value={values.opedEpochNum}
              onChange={(e) => patch({ opedEpochNum: e.target.value })}
              className={inputClass}
            />
          </Field>
        ) : (
          <Field label="Epochs">
            <input
              type="number"
              min={1}
              value={values.epochs}
              onChange={(e) => patch({ epochs: e.target.value })}
              className={inputClass}
            />
          </Field>
        )}

        <Field label="Batch size">
          <input
            type="number"
            min={1}
            value={values.batchSize}
            onChange={(e) => patch({ batchSize: e.target.value })}
            className={inputClass}
          />
        </Field>

        <Field label="Learning rate">
          <input
            type="number"
            min={0}
            step="any"
            value={values.learningRate}
            onChange={(e) => patch({ learningRate: e.target.value })}
            className={inputClass}
          />
        </Field>

        <Field label="Weight decay">
          <input
            type="number"
            min={0}
            step="any"
            value={values.weightDecay}
            onChange={(e) => patch({ weightDecay: e.target.value })}
            className={inputClass}
          />
        </Field>

        <Field label="Gradient clip">
          <input
            type="number"
            min={0}
            step="any"
            value={values.gradClip}
            onChange={(e) => patch({ gradClip: e.target.value })}
            className={inputClass}
          />
        </Field>

        <Field label="LR scheduler">
          <select
            value={values.scheduler}
            onChange={(e) => patch({ scheduler: e.target.value as SchedulerName })}
            className={inputClass}
          >
            <option value="none">None</option>
            <option value="step">Step</option>
            <option value="cosine">Cosine annealing</option>
            <option value="exponential">Exponential</option>
          </select>
        </Field>
      </div>

      {schedulerFields}

      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Early stopping patience">
          <input
            type="number"
            min={0}
            value={values.earlyStoppingPatience}
            onChange={(e) => patch({ earlyStoppingPatience: e.target.value })}
            className={inputClass}
          />
        </Field>
        <Field label="Early stopping min delta">
          <input
            type="number"
            min={0}
            step="any"
            value={values.earlyStoppingMinDelta}
            onChange={(e) => patch({ earlyStoppingMinDelta: e.target.value })}
            className={inputClass}
          />
        </Field>
      </div>

      {modelName === 'deepprime' && (
        <div className="space-y-2">
          <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={values.loadPretrained}
              onChange={(e) => patch({ loadPretrained: e.target.checked })}
            />
            Load pretrained weights before fine-tuning
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={values.trainEnsemble}
              onChange={(e) => patch({ trainEnsemble: e.target.checked })}
            />
            Fine-tune all ensemble members (not just the first)
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={values.reshuffleEachEpoch}
              onChange={(e) => patch({ reshuffleEachEpoch: e.target.checked })}
            />
            Reshuffle training batches each epoch
          </label>
        </div>
      )}

      {modelName === 'pridict2' && (
        <Field label="Loss function">
          <select
            value={values.pridictLossFunc}
            onChange={(e) => patch({ pridictLossFunc: e.target.value })}
            className={inputClass}
          >
            <option value="KLDloss">KLD loss</option>
            <option value="MSEloss">MSE loss</option>
          </select>
        </Field>
      )}
    </div>
  )
}
