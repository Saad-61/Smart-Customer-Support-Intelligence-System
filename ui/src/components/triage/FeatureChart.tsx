import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts'
import { Info, HelpCircle } from 'lucide-react'
import {
  Tooltip as UiTooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import type { ExplanationFeature } from '../../types/api'

interface FeatureChartProps {
  features: ExplanationFeature[]
  category: string
}

interface TooltipPayload {
  payload: ExplanationFeature
}

const CustomTooltip: React.FC<{
  active?: boolean
  payload?: TooltipPayload[]
}> = ({ active, payload }) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload
    const isPositive = data.contribution >= 0
    return (
      <div className="bg-[#101217] border border-border p-3 rounded text-xs font-mono shadow-xl space-y-1">
        <div className="font-bold text-foreground border-b border-border/80 pb-1">
          TOKEN: &quot;{data.feature}&quot;
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">Contribution:</span>
          <span className={isPositive ? 'text-[#f87171] font-bold' : 'text-[#9ca3af]'}>
            {data.contribution.toFixed(4)}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">Class Weight:</span>
          <span className="text-foreground">{data.weight.toFixed(4)}</span>
        </div>
        <div className="text-[11px] text-muted-foreground pt-0.5">
          {isPositive ? '↑ Pushes towards predicted category' : '↓ Pushes away from category'}
        </div>
      </div>
    )
  }
  return null
}

export const FeatureChart: React.FC<FeatureChartProps> = ({ features, category }) => {
  if (!features || features.length === 0) {
    return null
  }

  const chartData = [...features].map((item) => ({
    feature: item.feature.length > 20 ? `${item.feature.slice(0, 18)}...` : item.feature,
    contribution: parseFloat(item.contribution.toFixed(4)),
    weight: item.weight,
    rawFeature: item.feature,
  }))

  return (
    <TooltipProvider delayDuration={150}>
      <div className="border border-border bg-[#16181f] rounded-lg p-5 space-y-3 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border pb-2.5">
          <div>
            <h3 className="text-sm font-semibold text-foreground tracking-tight">
              Feature Attributions (Explainability)
            </h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Salient n-gram tokens influencing the [{category}] classification.
            </p>
          </div>

          <UiTooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-help self-start sm:self-auto"
              >
                <Info className="w-3.5 h-3.5 text-[#d13639]" />
                <span>Hyperplane Attribution</span>
                <HelpCircle className="w-3 h-3 opacity-70" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-xs bg-[#101217] border-border text-xs leading-relaxed text-foreground p-3">
              <p className="font-semibold text-foreground mb-1">Linear Classifier Attribution</p>
              Computes feature contribution as (token TF-IDF value * model class weight). Positive tokens increase probability of the predicted category.
            </TooltipContent>
          </UiTooltip>
        </div>

        <div className="h-60 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 5, right: 20, left: 30, bottom: 5 }}
            >
              <XAxis
                type="number"
                stroke="#4b5563"
                tick={{ fill: '#8c95a3', fontSize: 10, fontFamily: 'DM Mono' }}
                tickLine={{ stroke: '#292d38' }}
                axisLine={{ stroke: '#292d38' }}
              />
              <YAxis
                type="category"
                dataKey="feature"
                stroke="#4b5563"
                tick={{ fill: '#f0f2f5', fontSize: 11, fontFamily: 'DM Mono' }}
                tickLine={false}
                axisLine={{ stroke: '#292d38' }}
                width={95}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(209, 54, 57, 0.06)' }} />
              <ReferenceLine x={0} stroke="#292d38" strokeWidth={1.5} />
              <Bar dataKey="contribution" radius={[0, 3, 3, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.contribution >= 0 ? '#d13639' : '#475569'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-between text-[11px] font-mono text-muted-foreground border-t border-border/60 pt-2">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#d13639] inline-block" /> Positive Support
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#475569] inline-block" /> Opposing Feature
            </span>
          </div>
          <span>Contribution = TF-IDF * Class Weight</span>
        </div>
      </div>
    </TooltipProvider>
  )
}
