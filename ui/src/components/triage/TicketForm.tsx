import { useState } from 'react'
import { Send, Loader2, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import type { TicketRequest } from '../../types/api'

interface TicketFormProps {
  onSubmit: (request: TicketRequest) => void
  loading: boolean
}

const PRESET_SCENARIOS = [
  {
    label: 'Billing / Charge',
    text: 'I was charged twice on my credit card for the in-game bundle yesterday, but only received the points once. Order ref #89421.',
  },
  {
    label: 'Crash / Technical',
    text: 'Game crashes directly to desktop with an unexpected critical system error right after the latest patch update.',
  },
  {
    label: 'Account / Ban',
    text: 'My account was unauthorizedly accessed from another country and received a suspension for third-party software. I have recovered my email.',
  },
  {
    label: 'Player Behavior',
    text: 'A player was intentionally throwing the ranked match, refusing to participate, and using aggressive insults in team chat.',
  },
  {
    label: 'Ambiguous Query',
    text: 'hey i need help with some stuff',
  },
]

export const TicketForm: React.FC<TicketFormProps> = ({ onSubmit, loading }) => {
  const [text, setText] = useState(
    'I was charged twice on my credit card for the in-game bundle yesterday, but only received the points once. Order ref #89421.'
  )
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!text.trim()) {
      setError('Please provide player complaint details.')
      return
    }
    setError(null)
    onSubmit({
      ticket_text: text.trim(),
      product: 'League of Legends',
      previous_tickets: 0,
    })
  }

  const handleApplyPreset = (scenarioText: string) => {
    setText(scenarioText)
    setError(null)
  }

  return (
    <div className="border border-border bg-[#16181f] rounded-lg p-5 lg:p-6 space-y-4 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h2 className="text-sm font-semibold text-foreground tracking-tight">
            Incoming Player Support Ticket
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Submit an unclassified player ticket for instant automated triage.
          </p>
        </div>

        {/* Quick test scenarios */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[11px] font-mono text-muted-foreground mr-1">SCENARIOS:</span>
          {PRESET_SCENARIOS.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => handleApplyPreset(p.text)}
              className="text-xs font-medium px-2.5 py-1 rounded border border-border bg-[#1d2029] text-muted-foreground hover:text-foreground hover:border-[#d13639] transition-colors"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <label htmlFor="ticket-text" className="text-muted-foreground font-medium">
              Complaint Narrative
            </label>
            <span className="text-muted-foreground text-[11px] font-mono">
              {text.length} characters
            </span>
          </div>
          <Textarea
            id="ticket-text"
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste player support issue description, error codes, payment issues, or ban appeal details..."
            className="w-full bg-[#101217] border-border text-foreground placeholder:text-[#586273] font-sans focus-visible:ring-[#d13639] text-sm resize-y min-h-[110px] rounded-md"
          />
        </div>

        {error && (
          <div className="flex items-center gap-2 text-xs text-[#d13639] bg-[#d13639]/10 border border-[#d13639]/30 p-2.5 rounded-md">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="pt-1 flex items-center justify-end">
          <Button
            type="submit"
            disabled={loading}
            className="w-full sm:w-auto px-6 py-2 bg-[#d13639] hover:bg-[#b82e31] text-white font-medium text-xs uppercase tracking-wider transition-all shadow-sm rounded-md"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                <span>TRIAGING TICKET...</span>
              </>
            ) : (
              <>
                <Send className="w-3.5 h-3.5 mr-2" />
                <span>ANALYZE TICKET</span>
              </>
            )}
          </Button>
        </div>
      </form>
    </div>
  )
}
