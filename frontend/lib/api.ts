/**
 * Types and fetch helpers for the Cascade API.
 *
 * Types are written by hand against the Pydantic schemas rather than generated,
 * because generation would add a build step and a dependency for six shapes that
 * change rarely.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export type Tier = "T1" | "T2" | "T3";
export type Subject = "math" | "science" | "history" | "language" | "general";
export type Level = "school" | "undergrad";
export type Outcome = "accepted" | "escalated" | "provider_error";

export interface RoutingStep {
  tier: Tier;
  provider: string;
  model: string;
  latency_ms: number;
  verifier_score: number | null;
  outcome: Outcome;
  detail: string | null;
}

export interface AskResponse {
  answer: string;
  trace: RoutingStep[];
  predicted_tier: Tier;
  final_tier: Tier;
  escalated: boolean;
  prediction_confidence: number;
  prediction_source: string;
  estimated_cost_usd: number;
  baseline_cost_usd: number;
  total_latency_ms: number;
  cached: boolean;
  request_id: string;
  warnings: string[];
}

export interface ApiError {
  error_code: string;
  message: string;
  request_id: string;
}

export interface RequestSummary {
  id: string;
  created_at: string;
  question: string;
  subject: string;
  predicted_tier: Tier;
  final_tier: Tier;
  escalated: boolean;
  estimated_cost_usd: number;
  baseline_cost_usd: number;
  total_latency_ms: number;
  cached: boolean;
}

export interface RequestPage {
  items: RequestSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface TierStats {
  tier: Tier;
  count: number;
  escalation_rate: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
}

export interface MetricsResponse {
  total_requests: number;
  escalation_rate: number;
  estimated_total_cost_usd: number;
  estimated_baseline_cost_usd: number;
  estimated_saving_usd: number;
  estimated_saving_percent: number;
  cache_hit_rate: number;
  tier_distribution: Record<string, number>;
  tier_stats: TierStats[];
  feedback_positive: number;
  feedback_negative: number;
  cost_basis: string;
}

export interface ModelHealth {
  provider: string;
  model: string;
  tier: Tier;
  advertised: boolean;
  circuit_open: boolean;
}

export interface DriftEvent {
  provider: string;
  model: string;
  kind: string;
  detail: string;
}

export interface ProviderHealth {
  provider: string;
  reachable: boolean;
  model_count: number;
  last_checked: string | null;
  last_error: string | null;
}

export interface CatalogResponse {
  providers: ProviderHealth[];
  models: ModelHealth[];
  recent_drift: DriftEvent[];
}

/** Thrown with the message the API supplied, so the UI can show what happened. */
export class CascadeApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly requestId: string,
  ) {
    super(message);
    this.name = "CascadeApiError";
  }
}

async function parseError(response: Response): Promise<never> {
  let body: Partial<ApiError> = {};
  try {
    body = (await response.json()) as ApiError;
  } catch {
    // A non-JSON error body means something upstream of the app failed. Fall
    // through to a generic message rather than showing the reader raw HTML.
  }
  throw new CascadeApiError(
    body.message ?? "Something went wrong. Please try again.",
    body.error_code ?? `http_${response.status}`,
    body.request_id ?? "",
  );
}

export async function askQuestion(
  question: string,
  subject: Subject = "general",
  level: Level = "school",
): Promise<AskResponse> {
  const response = await fetch(`${API_BASE}/api/v1/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, subject, level }),
  });
  if (!response.ok) return parseError(response);
  return (await response.json()) as AskResponse;
}

export async function sendFeedback(
  requestId: string,
  helpful: boolean,
): Promise<void> {
  await fetch(`${API_BASE}/api/v1/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: requestId, helpful }),
  });
}

/**
 * Server-side fetches for the history and metrics pages.
 *
 * `no-store` because both are live views of a system whose whole point is that
 * routing changes as providers change; a cached dashboard would misreport health.
 */
export async function fetchRequests(params: {
  tier?: string;
  escalated?: string;
  limit?: number;
}): Promise<RequestPage | null> {
  const query = new URLSearchParams();
  query.set("limit", String(params.limit ?? 50));
  if (params.tier) query.set("tier", params.tier);
  if (params.escalated) query.set("escalated", params.escalated);

  try {
    const response = await fetch(`${API_BASE}/api/v1/requests?${query}`, {
      cache: "no-store",
    });
    if (!response.ok) return null;
    return (await response.json()) as RequestPage;
  } catch {
    return null;
  }
}

export async function fetchMetrics(): Promise<MetricsResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/metrics`, {
      cache: "no-store",
    });
    if (!response.ok) return null;
    return (await response.json()) as MetricsResponse;
  } catch {
    return null;
  }
}

export async function fetchCatalog(): Promise<CatalogResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/catalog`, {
      cache: "no-store",
    });
    if (!response.ok) return null;
    return (await response.json()) as CatalogResponse;
  } catch {
    return null;
  }
}

/** Cost figures are tiny; fixed notation would render every one as $0.00. */
export function formatCost(usd: number): string {
  if (usd === 0) return "$0";
  if (usd < 0.0001) return `$${usd.toExponential(2)}`;
  return `$${usd.toFixed(6)}`;
}

export function formatLatency(ms: number): string {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export const TIER_LABEL: Record<Tier, string> = {
  T1: "Small model",
  T2: "Mid model",
  T3: "Strongest model",
};
