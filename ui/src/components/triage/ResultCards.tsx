import React from 'react'
import { Clock, ShieldCheck, HelpCircle } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { AnimatedNumber } from '../bits/AnimatedNumber'
import type { PredictionResponse } from '../../types/api'

interface ResultCardsProps {
  prediction: PredictionResponse
}

export const ResultCards: React.FC<ResultCardsProps> = ({ prediction }) => {
  const getPriorityStyle = (priority: string) => {
    switch (priority.toUpperCase()) {
      case 'HIGH':
        return {
          textColor: 'text-[#f87171]',
          bgColor: 'bg-[#d13639]/15',
          borderColor: 'border-[#d13639]/50',
          badgeText: 'CRITICAL / IMMEDIATE ATTENTION',
        }
      case 'MEDIUM':
        return {
          textColor: 'text-[#fbbf24]',
          bgColor: 'bg-[#d97706]/15',
          borderColor: 'border-[#d97706]/50',
          badgeText: 'STANDARD QUEUE SLA (12H)',
        }
      case 'LOW':
        return {
          textColor: 'text-[#34d399]',
          bgColor: 'bg-[#10b981]/15',
          borderColor: 'border-[#10b981]/50',
          badgeText: 'ROUTINE INQUIRY QUEUE SLA (24H)',
        }
      default:
        return {
          textColor: 'text-foreground',
          bgColor: 'bg-muted/10',
          borderColor: 'border-border',
          badgeText: 'UNASSIGNED QUEUE',
        }
    }
  }

  const pStyle = getPriorityStyle(prediction.priority)

  return (
    <TooltipProvider delayDuration={150}>
      <div className="space-y-4">
        {/* Top Meta Bar */}
        <div className="flex items-center justify-between text-xs font-mono text-muted-foreground px-1">
          <span className="font-semibold text-foreground tracking-wide">
            PREDICTION SUMMARY
          </span>
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Clock className="w-3.5 h-3.5" />
            <span>Latency:</span>
            <strong className="text-foreground">
              <AnimatedNumber value={prediction.processing_time_ms} decimals={1} suffix=" ms" />
            </strong>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Category Classification Card */}
          <Card className="bg-[#16181f] border-border rounded-lg shadow-sm">
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-border/80 pb-2.5">
                <span className="text-xs text-muted-foreground uppercase font-medium">
                  Issue Category
                </span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      className="flex items-center gap-1 text-[11px] font-mono text-muted-foreground hover:text-foreground transition-colors cursor-help"
                    >
                      <ShieldCheck className="w-3.5 h-3.5 text-[#d13639]" />
                      <span>PLATT-CALIBRATED</span>
                      <HelpCircle className="w-3 h-3 ml-0.5 opacity-70" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-xs bg-[#101217] border-border text-xs leading-relaxed text-foreground p-3">
                    <p className="font-semibold text-[#d13639] mb-1">Platt Calibration (Sigmoid Scaling)</p>
                    Fits a logistic regression map over raw SVM decision margins to yield realistic, reliable probabilities that match actual error rates.
                  </TooltipContent>
                </Tooltip>
              </div>

              <div>
                <div className="text-xl font-bold text-foreground tracking-tight">
                  {prediction.category}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Calibrated Posterior Confidence:
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="text-muted-foreground">CONFIDENCE</span>
                  <span className="text-base font-bold text-[#d13639]">
                    <AnimatedNumber
                      value={prediction.category_confidence * 100}
                      decimals={1}
                      suffix="%"
                    />
                  </span>
                </div>
                <div className="w-full bg-[#202430] h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-[#d13639] h-full transition-all duration-700 ease-out rounded-full"
                    style={{ width: `${Math.min(prediction.category_confidence * 100, 100)}%` }}
                  />
                </div>
              </div>

              <div className="text-[11px] font-mono text-muted-foreground border-t border-border/60 pt-2.5 flex items-center justify-between">
                <span>LinearSVC + 3-Fold Platt</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="text-foreground underline decoration-dotted underline-offset-4 cursor-help flex items-center gap-1">
                      AdaECE: 0.40%
                      <HelpCircle className="w-3 h-3 text-muted-foreground" />
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-xs bg-[#101217] border-border text-xs leading-relaxed text-foreground p-3">
                    <p className="font-semibold text-foreground mb-1">Adaptive Expected Calibration Error (AdaECE)</p>
                    Measures whether a predicted 90% confidence actually corresponds to 90% real-world accuracy across equal-frequency bins. An AdaECE of 0.40% indicates near-perfect probability alignment.
                  </TooltipContent>
                </Tooltip>
              </div>
            </CardContent>
          </Card>

          {/* Priority Prediction Card */}
          <Card className="bg-[#16181f] border-border rounded-lg shadow-sm">
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-border/80 pb-2.5">
                <span className="text-xs text-muted-foreground uppercase font-medium">
                  Triage Priority
                </span>
                <div className="flex items-center gap-1 text-[11px] font-mono text-muted-foreground">
                  <span className="w-2 h-2 rounded-full bg-[#d13639]" />
                  <span>XGBOOST CLASSIFIER</span>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-3">
                  <div
                    className={`text-xl font-bold tracking-wide px-3 py-1 rounded border ${pStyle.borderColor} ${pStyle.bgColor} ${pStyle.textColor}`}
                  >
                    {prediction.priority}
                  </div>
                </div>
                <div className="text-xs text-muted-foreground mt-2">
                  {pStyle.badgeText}
                </div>
              </div>

              {/* Priority Confidence Bar */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="text-muted-foreground">PRIORITY CONFIDENCE</span>
                  <span className="text-base font-bold text-foreground">
                    <AnimatedNumber
                      value={prediction.priority_confidence * 100}
                      decimals={1}
                      suffix="%"
                    />
                  </span>
                </div>
                <div className="w-full bg-[#202430] h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-[#f0f2f5] h-full transition-all duration-700 ease-out rounded-full"
                    style={{ width: `${Math.min(prediction.priority_confidence * 100, 100)}%` }}
                  />
                </div>
              </div>

              <div className="text-[11px] font-mono text-muted-foreground border-t border-border/60 pt-2.5 flex items-center justify-between">
                <span>TruncatedSVD + OneHot + XGB</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="text-foreground underline decoration-dotted underline-offset-4 cursor-help flex items-center gap-1">
                      Weighted F1: 82.3%
                      <HelpCircle className="w-3 h-3 text-muted-foreground" />
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-xs bg-[#101217] border-border text-xs leading-relaxed text-foreground p-3">
                    <p className="font-semibold text-foreground mb-1">Weighted F1 Score</p>
                    Harmonic mean of precision and recall calculated across priority tiers, weighted by the number of tickets in each class. Accounts for real-world support ticket class imbalance.
                  </TooltipContent>
                </Tooltip>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </TooltipProvider>
  )
}
