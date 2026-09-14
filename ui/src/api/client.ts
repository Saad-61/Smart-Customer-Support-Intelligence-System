import type { HealthResponse, PredictionResponse, SimilarTicket, TicketRequest } from '../types/api'

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/+$/, '') || 'http://localhost:8000'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

export async function predictTicket(payload: TicketRequest): Promise<PredictionResponse> {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    let errorDetail = 'Inference request failed'
    try {
      const err = await response.json()
      if (err.detail) {
        errorDetail = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)
      }
    } catch {
      errorDetail = response.statusText || errorDetail
    }
    throw new ApiError(response.status, errorDetail)
  }

  return response.json()
}

export async function getSimilarTickets(query_text: string, top_k = 5): Promise<SimilarTicket[]> {
  const response = await fetch(`${BASE_URL}/similar`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ query_text, top_k }),
  })

  if (!response.ok) {
    throw new ApiError(response.status, 'Failed to retrieve similar tickets')
  }

  const data = await response.json()
  return data.results || []
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${BASE_URL}/health`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    throw new ApiError(response.status, 'Backend health check failed')
  }

  return response.json()
}
