import { useMemo, type ReactNode } from 'react'
import type { WeightSet } from '@apps/ensemble/services/api'
import {
  defaultSchedulerForModel,
  epochBudgetForModel,
  schedulerFormFieldsFor,
  type SchedulerName,
} from '@apps/ensemble/config/schedulerDefaults'

export type { SchedulerName }

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
  pretrainedWeightId: string
  reshuffleEachEpoch: boolean
  pridictLossFunc: string
  pridictNumEpochs: string
  opedEpochNum: string
  dpHiddenSize: string
  dpNumLayers: string
  opedEmbeddingSize: string
  opedFfnDim: string
  opedEncoderLayers: string
  opedNhead: string
  opedDropout: string
  pridict2EmbedDim: string
  pridict2ZDim: string
  pridict2NumHiddenLayers: string
  pridict2Dropout: string
}

export const DEFAULT_HYPERPARAMETERS: HyperparameterFormState = {
  epochs: '5',
  batchSize: '128',
  learningRate: '0.0001',
  weightDecay: '0',
  gradClip: '1',
  ...schedulerFormFieldsFor('none'),
  earlyStoppingPatience: '10',
  earlyStoppingMinDelta: '0',
  loadPretrained: false,
  pretrainedWeightId: '',
  reshuffleEachEpoch: true,
  pridictLossFunc: 'KLDloss',
  pridictNumEpochs: '20',
  opedEpochNum: '100',
  dpHiddenSize: '128',
  dpNumLayers: '1',
  opedEmbeddingSize: '64',
  opedFfnDim: '2048',
  opedEncoderLayers: '6',
  opedNhead: '8',
  opedDropout: '0.1',
  pridict2EmbedDim: '64',
  pridict2ZDim: '64',
  pridict2NumHiddenLayers: '1',
  pridict2Dropout: '0.1',
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
  'w-full rounded-md border border-slate-300 px-3 py-2 text-sm disabled:bg-slate-100 disabled:text-slate-500 disabled:cursor-not-allowed'

interface TrainingHyperparametersPanelProps {
  modelName: string
  values: HyperparameterFormState
  onChange: (values: HyperparameterFormState) => void
  availableWeights?: WeightSet[]
  weightsLoading?: boolean
}

function buildArchitecturePayload(
  modelName: string,
  values: HyperparameterFormState
): Record<string, unknown> {
  if (modelName === 'deepprime') {
    return {
      hidden_size: Number(values.dpHiddenSize),
      num_layers: Number(values.dpNumLayers),
    }
  }
  if (modelName === 'oped') {
    const ffnDim = Number(values.opedFfnDim)
    const encoderLayers = Number(values.opedEncoderLayers)
    return {
      embedding_size: Number(values.opedEmbeddingSize),
      hidden_size: [ffnDim, ffnDim, ffnDim],
      num_encoder_layers: [encoderLayers, encoderLayers, encoderLayers],
      nhead: Number(values.opedNhead),
      drop_out: Number(values.opedDropout),
    }
  }
  if (modelName === 'pridict2') {
    return {
      embed_dim: Number(values.pridict2EmbedDim),
      z_dim: Number(values.pridict2ZDim),
      num_hidden_layers: Number(values.pridict2NumHiddenLayers),
      p_dropout: Number(values.pridict2Dropout),
    }
  }
  return {}
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
    load_pretrained: values.loadPretrained,
    ...(values.loadPretrained ? { freezing: true } : {}),
    ...(values.loadPretrained && values.pretrainedWeightId
      ? { weights: values.pretrainedWeightId }
      : {}),
    ...(schedulerName ? { scheduler: schedulerName, scheduler_kwargs } : { scheduler: 'none' }),
  }

  const architecture = values.loadPretrained
    ? {}
    : buildArchitecturePayload(modelName, values)

  if (modelName === 'deepprime') {
    return {
      ...common,
      ...architecture,
      epochs: Number(values.epochs),
    }
  }

  if (modelName === 'oped') {
    return {
      ...common,
      ...architecture,
      epoch_num: Number(values.opedEpochNum),
      scheduler: schedulerName ?? 'step',
      scheduler_kwargs:
        scheduler_kwargs ?? { step_size: Number(values.schedulerStepSize), gamma: Number(values.schedulerGamma) },
    }
  }

  if (modelName === 'pridict2') {
    return {
      ...common,
      ...architecture,
      num_epochs: Number(values.pridictNumEpochs),
      loss_func: values.pridictLossFunc,
    }
  }

  return common
}

function defaultWeightHint(modelName: string): string {
  if (modelName === 'deepprime') {
    return 'Leave blank to use the default checkpoint for the selected cell line and PE system.'
  }
  if (modelName === 'oped') {
    return 'Leave blank to use the bundled OPED checkpoint.'
  }
  return 'Select a checkpoint when fine-tuning. Multi-head PRIDICT2 runs require a cell-type suffix.'
}

export default function TrainingHyperparametersPanel({
  modelName,
  values,
  onChange,
  availableWeights = [],
  weightsLoading = false,
}: TrainingHyperparametersPanelProps) {
  const patch = (partial: Partial<HyperparameterFormState>) => {
    onChange({ ...values, ...partial })
  }

  const architectureDisabled = values.loadPretrained
  const hasRegisteredWeights = availableWeights.length > 0

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
            onChange={(e) => {
              const scheduler = e.target.value as SchedulerName
              patch(
                schedulerFormFieldsFor(scheduler, {
                  epochBudget: epochBudgetForModel(modelName, values),
                })
              )
            }}
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

      <div className="space-y-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
        <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
          <input
            type="checkbox"
            checked={values.loadPretrained}
            onChange={(e) =>
              patch({
                loadPretrained: e.target.checked,
                ...(e.target.checked ? {} : { pretrainedWeightId: '' }),
              })
            }
          />
          Fine-tune from pretrained weights
        </label>
        {values.loadPretrained && (
          <p className="text-xs text-slate-500">
            Representation layers stay frozen; only the output head is trained.
          </p>
        )}

        {values.loadPretrained && (
          <Field label="Pretrained checkpoint">
            <select
              value={values.pretrainedWeightId}
              onChange={(e) => patch({ pretrainedWeightId: e.target.value })}
              disabled={weightsLoading || !hasRegisteredWeights}
              className={inputClass}
            >
              <option value="">
                {weightsLoading
                  ? 'Loading checkpoints…'
                  : hasRegisteredWeights
                    ? 'Default checkpoint for this model'
                    : 'No registered checkpoints'}
              </option>
              {availableWeights.map((weight) => (
                <option key={weight.id} value={weight.id}>
                  {weight.label}
                  {weight.source === 'vendor' ? ' [vendor]' : ''}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-slate-500">{defaultWeightHint(modelName)}</p>
            {modelName === 'deepprime' && (
              <p className="mt-1 text-xs text-slate-500">
                All ensemble members in the selected checkpoint are fine-tuned together.
              </p>
            )}
          </Field>
        )}
      </div>

      <div>
        <h4 className="text-sm font-semibold text-slate-900 mb-1">Model architecture</h4>
        {architectureDisabled && (
          <p className="text-xs text-slate-500 mb-3">
            Architecture is fixed by the selected checkpoint during fine-tuning.
          </p>
        )}
        {!architectureDisabled && modelName === 'deepprime' && (
          <p className="text-xs text-slate-500 mb-3">
            GRU size applies when training from scratch without pretrained weights.
          </p>
        )}

        {modelName === 'deepprime' && (
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="GRU hidden size">
              <input
                type="number"
                min={1}
                value={values.dpHiddenSize}
                onChange={(e) => patch({ dpHiddenSize: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
            <Field label="GRU layers">
              <input
                type="number"
                min={1}
                value={values.dpNumLayers}
                onChange={(e) => patch({ dpNumLayers: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
          </div>
        )}

        {modelName === 'oped' && (
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Embedding size">
              <input
                type="number"
                min={1}
                value={values.opedEmbeddingSize}
                onChange={(e) => patch({ opedEmbeddingSize: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
            <Field label="FFN dim (per branch)">
              <input
                type="number"
                min={1}
                value={values.opedFfnDim}
                onChange={(e) => patch({ opedFfnDim: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
            <Field label="Encoder layers (per branch)">
              <input
                type="number"
                min={1}
                value={values.opedEncoderLayers}
                onChange={(e) => patch({ opedEncoderLayers: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
            <Field label="Attention heads">
              <input
                type="number"
                min={1}
                value={values.opedNhead}
                onChange={(e) => patch({ opedNhead: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
            <Field label="Dropout">
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={values.opedDropout}
                onChange={(e) => patch({ opedDropout: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
          </div>
        )}

        {modelName === 'pridict2' && (
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Embed dim">
              <input
                type="number"
                min={1}
                value={values.pridict2EmbedDim}
                onChange={(e) => patch({ pridict2EmbedDim: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
            <Field label="Latent dim (z)">
              <input
                type="number"
                min={1}
                value={values.pridict2ZDim}
                onChange={(e) => patch({ pridict2ZDim: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
            <Field label="RNN hidden layers">
              <input
                type="number"
                min={1}
                value={values.pridict2NumHiddenLayers}
                onChange={(e) => patch({ pridict2NumHiddenLayers: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
            <Field label="Dropout">
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={values.pridict2Dropout}
                onChange={(e) => patch({ pridict2Dropout: e.target.value })}
                disabled={architectureDisabled}
                className={inputClass}
              />
            </Field>
          </div>
        )}
      </div>

      {modelName === 'deepprime' && (
        <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
          <input
            type="checkbox"
            checked={values.reshuffleEachEpoch}
            onChange={(e) => patch({ reshuffleEachEpoch: e.target.checked })}
          />
          Reshuffle training batches each epoch
        </label>
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
