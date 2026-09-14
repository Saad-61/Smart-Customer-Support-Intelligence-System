import React, { useState } from 'react'
import { Header } from './components/layout/Header'
import { TicketForm } from './components/triage/TicketForm'
import { ResultCards } from './components/triage/ResultCards'
import { OodBanner } from './components/triage/OodBanner'
import { FeatureChart } from './components/triage/FeatureChart'
import { SimilarTickets } from './components/triage/SimilarTickets'
import { FadeContent } from './components/bits/FadeContent'
import { predictTicket } from './api/client'
import type { PredictionResponse, TicketRequest } from './types/api'
import { AlertCircle } from 'lucide-react'

export const App: React.FC = () => {
  const [loadingPrediction, setLoadingPrediction] = useState<boolean>(false)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [predictionError, setPredictionError] = useState<string | null>(null)

  const handleTicketSubmit = async (request: TicketRequest) => {
    setLoadingPrediction(true)
    setPredictionError(null)
    try {
      const result = await predictTicket(request)
      setPrediction(result)
    } catch (err: any) {
      setPredictionError(
        err.message || 'An error occurred while communicating with the triage inference service.'
      )
    } finally {
      setLoadingPrediction(false)
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans">
      <Header />

      <main className="flex-1 max-w-4xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Ticket Input Panel */}
        <TicketForm
          onSubmit={handleTicketSubmit}
          loading={loadingPrediction}
        />

        {/* Error Notification */}
        {predictionError && (
          <div className="p-4 border border-[#d13639]/50 bg-[#d13639]/10 rounded-lg text-foreground text-xs flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-[#d13639] shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-semibold text-[#d13639] uppercase tracking-wider">
                Inference Service Error
              </div>
              <p className="text-muted-foreground">{predictionError}</p>
              <div className="text-[11px] text-muted-foreground pt-1 font-mono">
                Verify that the API server is active on{' '}
                <code className="text-[#d13639]">
                  {import.meta.env.VITE_API_URL || 'http://localhost:8000'}
                </code>
              </div>
            </div>
          </div>
        )}

        {/* Prediction Results */}
        {prediction && (
          <FadeContent duration={350} className="space-y-6">
            {/* OOD banner if confidence < 0.50 */}
            <OodBanner
              uncertain={prediction.uncertain}
              confidence={prediction.category_confidence}
            />

            {/* Primary Category & Priority cards with tooltips */}
            <ResultCards prediction={prediction} />

            {/* Linear Hyperplane Feature Attributions */}
            <FeatureChart
              features={prediction.explanation}
              category={prediction.category}
            />

            {/* Dense Semantic Retrieval Table */}
            <SimilarTickets tickets={prediction.similar_tickets} />
          </FadeContent>
        )}
      </main>
    </div>
  )
}

export default App
