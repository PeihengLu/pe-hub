import type { ExportFormat } from '@apps/database/config/exportAttributes'

export const MODEL_EXPORT_FORMAT: Record<string, ExportFormat> = {
  deepprime: 'deepprime',
  pridict2: 'pridict2',
  oped: 'oped',
}

export function exportFormatForModel(modelName: string): ExportFormat {
  return MODEL_EXPORT_FORMAT[modelName] ?? 'std'
}
