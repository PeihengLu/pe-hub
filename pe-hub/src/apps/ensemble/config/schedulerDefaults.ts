export type SchedulerName = 'none' | 'step' | 'cosine' | 'exponential'

/**
 * Scheduler kwargs aligned with pe_common.training.build_lr_scheduler and
 * pe-ensemble model wrappers (OPED defaults to step; DeepPrime to none).
 */
export const SCHEDULER_KWARG_DEFAULTS: Record<
  SchedulerName,
  { schedulerStepSize?: string; schedulerGamma?: string; schedulerTMax?: string }
> = {
  none: {},
  step: {
    schedulerStepSize: '10',
    schedulerGamma: '0.95',
  },
  cosine: {
    schedulerTMax: '10',
  },
  exponential: {
    schedulerGamma: '0.98',
  },
}

/** Wrapper default scheduler type per built-in model. */
export function defaultSchedulerForModel(modelName: string): SchedulerName {
  if (modelName === 'oped') {
    return 'step'
  }
  return 'none'
}

export function epochBudgetForModel(
  modelName: string,
  values: { epochs: string; opedEpochNum: string; pridictNumEpochs: string }
): string {
  if (modelName === 'oped') {
    return values.opedEpochNum
  }
  if (modelName === 'pridict2') {
    return values.pridictNumEpochs
  }
  return values.epochs
}

/** Form fields to apply when the user picks a scheduler (or model changes). */
export function schedulerFormFieldsFor(
  scheduler: SchedulerName,
  options?: { epochBudget?: string }
): {
  scheduler: SchedulerName
  schedulerStepSize: string
  schedulerGamma: string
  schedulerTMax: string
} {
  const kwargs = SCHEDULER_KWARG_DEFAULTS[scheduler]
  const tMax =
    scheduler === 'cosine' && options?.epochBudget?.trim()
      ? options.epochBudget.trim()
      : (kwargs.schedulerTMax ?? '10')

  return {
    scheduler,
    schedulerStepSize: kwargs.schedulerStepSize ?? '10',
    schedulerGamma: kwargs.schedulerGamma ?? '0.95',
    schedulerTMax: tMax,
  }
}
