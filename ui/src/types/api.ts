export interface TicketRequest {
  ticket_text: string;
  product: string;
  previous_tickets: number;
}

export interface SimilarTicket {
  ticket_id: string;
  similarity: number;
  preview: string;
  category: string;
  priority: string;
  product: string;
}

export interface ExplanationFeature {
  feature: string;
  weight: number;
  contribution: number;
}

export interface PredictionResponse {
  category: string;
  priority: string;
  category_confidence: number;
  priority_confidence: number;
  calibrated_note: string;
  similar_tickets: SimilarTicket[];
  explanation: ExplanationFeature[];
  uncertain: boolean;
  processing_time_ms: number;
}

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  device: string;
  gpu_name: string | null;
  loaded_models: Record<string, string>;
  total_indexed_tickets: number;
}

export interface HistoryItem {
  id: string;
  timestamp: number;
  ticket_text: string;
  product: string;
  previous_tickets: number;
  category: string;
  priority: string;
  confidence: number;
  uncertain: boolean;
}
