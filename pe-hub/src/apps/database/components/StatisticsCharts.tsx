import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Sector,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { PieSectorDataItem } from 'recharts/types/polar/Pie'
import type { Statistics } from '@apps/database/services/peDbApi'
import Card from '@components/Card'

const STUDY_COLORS = [
  '#4E79A7',
  '#F28E2B',
  '#E15759',
  '#76B7B2',
  '#59A14F',
  '#B07AA1',
  '#FF9DA7',
  '#9C755F',
]

const CATEGORY_COLORS = [
  '#64748B',
  '#0EA5E9',
  '#14B8A6',
  '#8B5CF6',
  '#F59E0B',
  '#EF4444',
  '#22C55E',
  '#EC4899',
  '#6366F1',
  '#84CC16',
]

type StatRow = { study: string; count: number; category: string }

function formatLabel(value: string | number): string {
  return String(value).replace(/_/g, ' ')
}

function formatCount(value: number): string {
  return value.toLocaleString()
}

function collectStudies(stats: Statistics): string[] {
  const names = new Set<string>()
  const series = [
    stats.edit_type,
    stats.edit_length,
    stats.pegRNA_delivery_method,
    stats.pe_delivery_method,
    stats.edit_scope,
    stats.experimental_method,
    stats.target_context,
  ]
  for (const rows of series) {
    for (const row of rows ?? []) {
      if (row.study) names.add(row.study)
    }
  }
  return Array.from(names).sort()
}

function studyColorMap(studies: string[]): Record<string, string> {
  return Object.fromEntries(
    studies.map((study, index) => [
      study,
      STUDY_COLORS[index % STUDY_COLORS.length],
    ])
  )
}

function toCategoryRows(
  rows: Array<Record<string, string | number>> | undefined,
  categoryKey: string
): StatRow[] {
  return (rows ?? []).map((row) => ({
    study: String(row.study),
    count: Number(row.count) || 0,
    category: formatLabel(row[categoryKey] as string | number),
  }))
}

function StudyLegend({
  studies,
  colors,
}: {
  studies: string[]
  colors: Record<string, string>
}) {
  if (studies.length === 0) return null
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm px-4 py-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
        Study
      </p>
      <div className="flex flex-wrap gap-x-4 gap-y-2">
        {studies.map((study) => (
          <div key={study} className="flex items-center gap-2 text-sm text-slate-700">
            <span
              className="inline-block h-3 w-3 rounded-sm shrink-0"
              style={{ backgroundColor: colors[study] }}
            />
            <span>{study}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function EmptyChart() {
  return (
    <div className="h-64 flex items-center justify-center text-sm text-slate-500">
      No data
    </div>
  )
}

function StudyCompositionTooltip({
  active,
  payload,
  studyColors,
  compositionByCategory,
}: {
  active?: boolean
  payload?: Array<{ name?: string; value?: number; payload?: { name: string; value: number } }>
  studyColors: Record<string, string>
  compositionByCategory: Record<string, Array<{ study: string; count: number }>>
}) {
  if (!active || !payload?.length) return null
  const category = String(payload[0].name ?? payload[0].payload?.name ?? '')
  const total = Number(payload[0].value ?? payload[0].payload?.value ?? 0)
  const composition = compositionByCategory[category] ?? []

  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 shadow-lg text-sm max-w-xs">
      <p className="font-semibold text-slate-900 mb-1">{category}</p>
      <p className="text-slate-500 mb-2">Total: {formatCount(total)}</p>
      <ul className="space-y-1">
        {composition.map(({ study, count }) => {
          const share = total > 0 ? (count / total) * 100 : 0
          return (
            <li key={study} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 text-slate-700">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-sm shrink-0"
                  style={{ backgroundColor: studyColors[study] }}
                />
                {study}
              </span>
              <span className="tabular-nums text-slate-600 whitespace-nowrap">
                {formatCount(count)} ({share.toFixed(1)}%)
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function renderActivePieShape(props: PieSectorDataItem) {
  const {
    cx = 0,
    cy = 0,
    innerRadius = 0,
    outerRadius = 0,
    startAngle,
    endAngle,
    fill,
  } = props
  return (
    <g>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 10}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
        stroke="#ffffff"
        strokeWidth={2}
        style={{ filter: 'brightness(1.08)', cursor: 'pointer' }}
      />
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={outerRadius + 12}
        outerRadius={outerRadius + 16}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
        opacity={0.35}
      />
    </g>
  )
}

function CategoryPieChart({
  title,
  rows,
  studyColors,
}: {
  title: string
  rows: StatRow[]
  studyColors: Record<string, string>
}) {
  const [activeIndex, setActiveIndex] = useState<number | undefined>(undefined)

  const { pieData, compositionByCategory, grandTotal } = useMemo(() => {
    const totals = new Map<string, number>()
    const composition = new Map<string, Map<string, number>>()

    for (const row of rows) {
      totals.set(row.category, (totals.get(row.category) ?? 0) + row.count)
      if (!composition.has(row.category)) composition.set(row.category, new Map())
      const byStudy = composition.get(row.category)!
      byStudy.set(row.study, (byStudy.get(row.study) ?? 0) + row.count)
    }

    const categories = Array.from(totals.keys()).sort((a, b) => a.localeCompare(b))
    const pieData = categories.map((name) => ({
      name,
      value: totals.get(name) ?? 0,
    }))
    const compositionByCategory = Object.fromEntries(
      categories.map((category) => [
        category,
        Array.from(composition.get(category)?.entries() ?? [])
          .map(([study, count]) => ({ study, count }))
          .sort((a, b) => b.count - a.count || a.study.localeCompare(b.study)),
      ])
    )
    const grandTotal = pieData.reduce((sum, row) => sum + row.value, 0)
    return { pieData, compositionByCategory, grandTotal }
  }, [rows])

  return (
    <Card title={title}>
      {pieData.length === 0 || grandTotal === 0 ? (
        <EmptyChart />
      ) : (
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius="70%"
                activeIndex={activeIndex}
                activeShape={renderActivePieShape}
                onMouseEnter={(_, index) => setActiveIndex(index)}
                onMouseLeave={() => setActiveIndex(undefined)}
                label={({ name, percent }) =>
                  `${name} (${((percent ?? 0) * 100).toFixed(0)}%)`
                }
                labelLine
              >
                {pieData.map((entry, index) => {
                  const isDimmed =
                    activeIndex !== undefined && activeIndex !== index
                  return (
                    <Cell
                      key={entry.name}
                      fill={CATEGORY_COLORS[index % CATEGORY_COLORS.length]}
                      stroke="#ffffff"
                      strokeWidth={1}
                      fillOpacity={isDimmed ? 0.45 : 1}
                      style={{ cursor: 'pointer', outline: 'none' }}
                    />
                  )
                })}
              </Pie>
              <Tooltip
                content={
                  <StudyCompositionTooltip
                    studyColors={studyColors}
                    compositionByCategory={compositionByCategory}
                  />
                }
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  )
}

function EditLengthBarChart({
  rows,
  studies,
  studyColors,
}: {
  rows: Array<{ study: string; edit_length: number; count: number }>
  studies: string[]
  studyColors: Record<string, string>
}) {
  const data = useMemo(() => {
    const byLength = new Map<number, Record<string, number | string>>()
    for (const row of rows) {
      const key = row.edit_length
      if (!byLength.has(key)) {
        byLength.set(key, { edit_length: key })
      }
      const entry = byLength.get(key)!
      entry[row.study] = (Number(entry[row.study]) || 0) + row.count
    }
    return Array.from(byLength.entries())
      .sort(([a], [b]) => a - b)
      .map(([, value]) => value)
  }, [rows])

  return (
    <Card title="Edit length">
      {data.length === 0 ? (
        <EmptyChart />
      ) : (
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 12, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
              <XAxis
                dataKey="edit_length"
                tick={{ fill: '#64748B', fontSize: 12 }}
                label={{
                  value: 'Edit length',
                  position: 'insideBottom',
                  offset: -2,
                  fill: '#64748B',
                  fontSize: 12,
                }}
              />
              <YAxis
                tick={{ fill: '#64748B', fontSize: 12 }}
                tickFormatter={(value: number) => formatCount(value)}
                label={{
                  value: 'Count',
                  angle: -90,
                  position: 'insideLeft',
                  fill: '#64748B',
                  fontSize: 12,
                }}
              />
              <Tooltip
                formatter={(value: number, name: string) => [formatCount(value), name]}
                labelFormatter={(label) => `Edit length ${label}`}
                contentStyle={{
                  borderRadius: 8,
                  borderColor: '#E2E8F0',
                  fontSize: 13,
                }}
              />
              {studies.map((study) => (
                <Bar
                  key={study}
                  dataKey={study}
                  stackId="study"
                  fill={studyColors[study]}
                  maxBarSize={28}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  )
}

export default function StatisticsCharts({ stats }: { stats: Statistics }) {
  const studies = useMemo(() => collectStudies(stats), [stats])
  const colors = useMemo(() => studyColorMap(studies), [studies])

  const pieCharts = [
    {
      title: 'Edit type',
      rows: toCategoryRows(stats.edit_type, 'edit_type'),
    },
    {
      title: 'Edit scope',
      rows: toCategoryRows(stats.edit_scope, 'edit_scope'),
    },
    {
      title: 'Experimental method',
      rows: toCategoryRows(stats.experimental_method, 'experimental_method'),
    },
    {
      title: 'Target context',
      rows: toCategoryRows(stats.target_context, 'target_context'),
    },
    {
      title: 'pegRNA delivery method',
      rows: toCategoryRows(stats.pegRNA_delivery_method, 'delivery_method'),
    },
    {
      title: 'PE delivery method',
      rows: toCategoryRows(stats.pe_delivery_method, 'delivery_method'),
    },
  ]

  return (
    <div className="space-y-4">
      <StudyLegend studies={studies} colors={colors} />
      <EditLengthBarChart
        rows={stats.edit_length ?? []}
        studies={studies}
        studyColors={colors}
      />
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {pieCharts.map((chart) => (
          <CategoryPieChart
            key={chart.title}
            title={chart.title}
            rows={chart.rows}
            studyColors={colors}
          />
        ))}
      </div>
    </div>
  )
}
