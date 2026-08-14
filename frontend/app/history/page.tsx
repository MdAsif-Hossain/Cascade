import Link from "next/link";
import { fetchRequests, formatCost, formatLatency, type Tier } from "@/lib/api";

export const dynamic = "force-dynamic";

const TIER_TEXT: Record<Tier, string> = {
  T1: "text-t1",
  T2: "text-t2",
  T3: "text-t3",
};

const FILTERS = [
  { label: "All", tier: undefined, escalated: undefined },
  { label: "T1", tier: "T1", escalated: undefined },
  { label: "T2", tier: "T2", escalated: undefined },
  { label: "T3", tier: "T3", escalated: undefined },
  { label: "Escalated", tier: undefined, escalated: "true" },
];

export default async function HistoryPage({
  searchParams,
}: {
  searchParams: { tier?: string; escalated?: string };
}) {
  const page = await fetchRequests({
    tier: searchParams.tier,
    escalated: searchParams.escalated,
  });

  return (
    <>
      <h1 className="font-display text-3xl font-semibold tracking-tight">History</h1>
      <p className="mt-2 text-muted">Questions asked, and how each one was routed.</p>

      <nav aria-label="Filter" className="mt-6 flex flex-wrap gap-2">
        {FILTERS.map((f) => {
          const params = new URLSearchParams();
          if (f.tier) params.set("tier", f.tier);
          if (f.escalated) params.set("escalated", f.escalated);
          const href = params.toString() ? `/history?${params}` : "/history";
          const active =
            (searchParams.tier ?? "") === (f.tier ?? "") &&
            (searchParams.escalated ?? "") === (f.escalated ?? "");
          return (
            <Link
              key={f.label}
              href={href}
              aria-current={active ? "page" : undefined}
              className={`rounded-md border px-3 py-1 font-mono text-xs uppercase tracking-wider ${
                active ? "border-ink bg-ink text-paper" : "border-ink/15 text-muted"
              }`}
            >
              {f.label}
            </Link>
          );
        })}
      </nav>

      {page === null ? (
        <p className="mt-10 rounded-lg border border-t3/40 bg-t3/5 px-4 py-3 text-sm">
          Could not load history. The API may be starting up — free hosting sleeps
          when idle, so the first request can take a moment.
        </p>
      ) : page.items.length === 0 ? (
        <p className="mt-10 border-t border-ink/10 pt-6 text-sm text-muted">
          Nothing here yet.{" "}
          <Link href="/" className="text-ink underline underline-offset-4">
            Ask your first question
          </Link>
          .
        </p>
      ) : (
        <ul className="mt-8 divide-y divide-ink/10 border-t border-ink/10">
          {page.items.map((item) => (
            <li key={item.id} className="py-4">
              <div className="flex items-start justify-between gap-4">
                <p className="flex-1 leading-snug">{item.question}</p>
                <span
                  className={`shrink-0 font-mono text-xs font-medium ${TIER_TEXT[item.final_tier]}`}
                >
                  {item.final_tier}
                </span>
              </div>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-muted">
                <span>{new Date(item.created_at).toLocaleString()}</span>
                <span>{formatLatency(item.total_latency_ms)}</span>
                <span>{formatCost(item.estimated_cost_usd)} est.</span>
                {item.predicted_tier !== item.final_tier && (
                  <span className="text-t2">
                    predicted {item.predicted_tier} → {item.final_tier}
                  </span>
                )}
                {item.cached && <span className="text-t1">cached</span>}
              </div>
            </li>
          ))}
        </ul>
      )}

      {page && page.total > page.items.length && (
        <p className="mt-6 font-mono text-xs text-muted">
          Showing {page.items.length} of {page.total}.
        </p>
      )}
    </>
  );
}
