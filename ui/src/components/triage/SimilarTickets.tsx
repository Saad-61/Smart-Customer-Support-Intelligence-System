import React from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import type { SimilarTicket } from '../../types/api'

interface SimilarTicketsProps {
  tickets: SimilarTicket[]
}

export const SimilarTickets: React.FC<SimilarTicketsProps> = ({ tickets }) => {
  if (!tickets || tickets.length === 0) {
    return null
  }

  const getPriorityBadgeClass = (priority: string) => {
    switch (priority?.toUpperCase()) {
      case 'HIGH':
        return 'text-[#f87171] border-[#d13639]/40 bg-[#d13639]/10'
      case 'MEDIUM':
        return 'text-[#fbbf24] border-[#d97706]/40 bg-[#d97706]/10'
      case 'LOW':
        return 'text-[#34d399] border-[#10b981]/40 bg-[#10b981]/10'
      default:
        return 'text-muted-foreground border-border bg-muted/10'
    }
  }

  return (
    <div className="border border-border bg-[#16181f] rounded-lg overflow-hidden shadow-sm">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground tracking-tight">
            Semantically Similar Historical Tickets
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Top historical matches retrieved via 384-dimensional dense neural embeddings (all-MiniLM-L6-v2).
          </p>
        </div>
        <span className="text-xs font-mono text-muted-foreground px-2 py-0.5 rounded bg-[#101217] border border-border">
          {tickets.length} MATCHES
        </span>
      </div>

      <div className="overflow-x-auto">
        <Table>
          <TableHeader className="bg-[#101217] border-b border-border text-xs font-medium">
            <TableRow className="border-border hover:bg-transparent">
              <TableHead className="text-muted-foreground py-2.5 w-24">Similarity</TableHead>
              <TableHead className="text-muted-foreground py-2.5 w-32">Ticket ID</TableHead>
              <TableHead className="text-muted-foreground py-2.5 w-36">Product</TableHead>
              <TableHead className="text-muted-foreground py-2.5 w-44">Historical Category</TableHead>
              <TableHead className="text-muted-foreground py-2.5 w-24">Priority</TableHead>
              <TableHead className="text-muted-foreground py-2.5">Complaint Excerpt</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody className="text-xs">
            {tickets.map((t, idx) => {
              const matchPct = (t.similarity * 100).toFixed(1)
              const isHighMatch = t.similarity >= 0.70
              return (
                <TableRow
                  key={t.ticket_id || idx}
                  className="border-border hover:bg-[#1a1d26] transition-colors"
                >
                  <TableCell className="font-mono font-medium py-3">
                    <span
                      className={`px-2 py-0.5 rounded text-[11px] border ${
                        isHighMatch
                          ? 'border-[#d13639]/40 text-[#f87171] bg-[#d13639]/10'
                          : 'border-border text-muted-foreground bg-[#101217]'
                      }`}
                    >
                      {matchPct}%
                    </span>
                  </TableCell>
                  <TableCell className="text-foreground font-mono py-3">
                    #{t.ticket_id}
                  </TableCell>
                  <TableCell className="text-muted-foreground py-3">
                    {t.product || 'League of Legends'}
                  </TableCell>
                  <TableCell className="text-foreground font-medium py-3">
                    {t.category}
                  </TableCell>
                  <TableCell className="py-3">
                    <span
                      className={`px-2 py-0.5 rounded border text-[10px] font-bold font-mono uppercase ${getPriorityBadgeClass(
                        t.priority
                      )}`}
                    >
                      {t.priority}
                    </span>
                  </TableCell>
                  <TableCell className="text-muted-foreground font-sans text-xs py-3 max-w-sm truncate">
                    &quot;{t.preview}&quot;
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
      <div className="p-2.5 border-t border-border/60 text-[11px] font-mono text-muted-foreground flex items-center justify-between bg-[#12141a]">
        <span>Dense Index: Cosine Similarity on Sentence Transformer Embeddings</span>
        <span>K = {tickets.length}</span>
      </div>
    </div>
  )
}
