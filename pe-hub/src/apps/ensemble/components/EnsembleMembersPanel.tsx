import { Plus, Trash2 } from 'lucide-react'
import Card from '@components/Card'
import SelectMenu from '@components/SelectMenu'
import type { Model, WeightSet } from '@apps/ensemble/services/api'
import type { CombineMethod } from '@apps/ensemble/config/combineMethods'
import { combineMethodOption } from '@apps/ensemble/config/combineMethods'

export interface EnsembleMemberRow {
  id: string
  modelName: string
  weightId: string
  memberWeight: string
}

interface EnsembleMembersPanelProps {
  models: Model[] | undefined
  weightSetsByModel: Record<string, WeightSet[] | undefined>
  weightsLoadingModel: string | null
  onLoadWeights: (modelName: string) => void
  members: EnsembleMemberRow[]
  onMembersChange: (members: EnsembleMemberRow[]) => void
  combine: CombineMethod
  onCombineChange: (value: CombineMethod) => void
  combineOptions: Record<string, unknown>
  onCombineOptionsChange: (options: Record<string, unknown>) => void
  combineOptionsList: Array<{ id: CombineMethod; label: string; description: string }>
}

function newMemberId(): string {
  return `member-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

export function createDefaultMember(modelName = 'deepprime'): EnsembleMemberRow {
  return {
    id: newMemberId(),
    modelName,
    weightId: '',
    memberWeight: '1',
  }
}

export default function EnsembleMembersPanel({
  models,
  weightSetsByModel,
  weightsLoadingModel,
  onLoadWeights,
  members,
  onMembersChange,
  combine,
  onCombineChange,
  combineOptions,
  onCombineOptionsChange,
  combineOptionsList,
}: EnsembleMembersPanelProps) {
  const selectedCombine = combineMethodOption(combine)

  const updateMember = (id: string, patch: Partial<EnsembleMemberRow>) => {
    onMembersChange(
      members.map((member) => (member.id === id ? { ...member, ...patch } : member))
    )
  }

  const removeMember = (id: string) => {
    if (members.length <= 2) return
    onMembersChange(members.filter((member) => member.id !== id))
  }

  const addMember = () => {
    const fallbackModel = members[members.length - 1]?.modelName ?? models?.[0]?.name ?? 'deepprime'
    onMembersChange([...members, createDefaultMember(fallbackModel)])
  }

  return (
    <Card title="Ensemble members & fusion">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">Combine method</label>
          <SelectMenu
            value={combine}
            onChange={(value) => onCombineChange(value as CombineMethod)}
            aria-label="Combine method"
            options={combineOptionsList.map((option) => ({
              value: option.id,
              label: option.label,
            }))}
          />
          {selectedCombine && (
            <p className="mt-2 text-sm text-slate-600">{selectedCombine.description}</p>
          )}
        </div>

        {selectedCombine?.needsTrimCount && (
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Trim count (per end)
            </label>
            <input
              type="number"
              min={0}
              step={1}
              value={String(combineOptions.trim_count ?? 1)}
              onChange={(event) =>
                onCombineOptionsChange({
                  ...combineOptions,
                  trim_count: Number(event.target.value),
                })
              }
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
        )}

        <div className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-slate-900">Members</h3>
            <button
              type="button"
              onClick={addMember}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              <Plus className="w-4 h-4" />
              Add member
            </button>
          </div>

          {members.map((member, index) => {
            const weightSets = weightSetsByModel[member.modelName]
            const hasWeights = (weightSets?.length ?? 0) > 0
            const isLoadingWeights = weightsLoadingModel === member.modelName

            return (
              <div
                key={member.id}
                className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-3"
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-medium text-slate-800">Member {index + 1}</p>
                  <button
                    type="button"
                    onClick={() => removeMember(member.id)}
                    disabled={members.length <= 2}
                    className="text-slate-400 hover:text-red-600 disabled:opacity-40"
                    aria-label={`Remove member ${index + 1}`}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Model</label>
                  <SelectMenu
                    value={member.modelName}
                    onChange={(value) => {
                      updateMember(member.id, { modelName: value, weightId: '' })
                      onLoadWeights(value)
                    }}
                    aria-label={`Member ${index + 1} model`}
                    options={
                      models?.map((model) => ({
                        value: model.name,
                        label: `${model.name} — ${model.description}`,
                      })) ?? []
                    }
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Weight set <span className="text-red-600">*</span>
                  </label>
                  <SelectMenu
                    value={member.weightId}
                    onChange={(value) => updateMember(member.id, { weightId: value })}
                    disabled={isLoadingWeights || !hasWeights}
                    aria-label={`Member ${index + 1} weight set`}
                    placeholder={
                      isLoadingWeights
                        ? 'Loading weight sets…'
                        : hasWeights
                          ? 'Select a weight set…'
                          : 'No weights registered'
                    }
                    options={
                      weightSets?.map((weight) => ({
                        value: weight.id,
                        label: `${weight.label}${weight.source === 'vendor' ? ' [vendor]' : ''}`,
                      })) ?? []
                    }
                  />
                </div>

                {selectedCombine?.needsMemberWeights && (
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Member weight
                    </label>
                    <input
                      type="number"
                      min={0}
                      step="any"
                      value={member.memberWeight}
                      onChange={(event) =>
                        updateMember(member.id, { memberWeight: event.target.value })
                      }
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                    />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </Card>
  )
}
