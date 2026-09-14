import React from 'react'
import { AlertTriangle, UserCheck } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'

interface OodBannerProps {
  uncertain: boolean
  confidence: number
}

export const OodBanner: React.FC<OodBannerProps> = ({ uncertain, confidence }) => {
  if (!uncertain) return null

  return (
    <Alert className="rounded-lg border-[#d97706]/60 bg-[#241a08] text-foreground p-4">
      <div className="flex items-start gap-3">
        <AlertTriangle className="h-5 w-5 text-[#f59e0b] shrink-0 mt-0.5" />
        <div className="space-y-1">
          <AlertTitle className="text-xs font-mono font-bold uppercase tracking-wider text-[#f59e0b] flex items-center gap-2">
            <span>OUT-OF-DISTRIBUTION GUARDRAIL // LOW CONFIDENCE</span>
          </AlertTitle>
          <AlertDescription className="text-xs text-[#e2d5c3] leading-relaxed">
            The Platt-calibrated prediction confidence (
            <strong className="text-white">{(confidence * 100).toFixed(1)}%</strong>
            ) is below the 50.0% safety threshold. This ticket exhibits ambiguous phrasing or novel vocabulary outside the model's training distribution.
            <div className="mt-1.5 flex items-center gap-1.5 text-[#f59e0b] font-medium">
              <UserCheck className="w-3.5 h-3.5" />
              <span>Recommended Action: Route to human tier-2 support queue for manual verification.</span>
            </div>
          </AlertDescription>
        </div>
      </div>
    </Alert>
  )
}
