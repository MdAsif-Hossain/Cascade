import { fetchCatalog, fetchMetrics, formatCost, type Tier } from "@/lib/api";

export const dynamic = "force-dynamic";

const TIER_BG: Record<Tier, string> = { T1: "bg-t1", T2: "bg-t2", T3: "bg-t3" };
const TIER_TEXT: Record<Tier, string> = {
  T1: "text-t1",
  T2: "text-t2",
  T3: "text-t3",
};

function Stat({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note?: string;
}) {
  return (
    <div className="rounded-lg border border-ink/10 bg-white/50 px-4 py-3">
      <p className="font-mono text-[11px] uppercase tracking-widest text-muted">
        {label}
      </p>
      <p className="mt-1 font-mono text-xl">{value}</p>
      {note && <p className="mt-0.5 text-[11px] text-muted">{note}</p>}
    </div>
  );
}

export default async function MetricsPage() {
  const [metrics, catalog] = await Promise.all([fetchMetrics(), fetchCatalog()]);

  if (!metrics) {
    return (
      <>
        <h1 className="font-display text-3xl font-semibold tracking-tight">Metrics</h1>
        <p className="mt-6 rounded-lg border border-t3/40 bg-t3/5 px-4 py-3 text-sm">
          Could not load metrics. The API may be waking up — free hosting sleeps
          when idle. Try again in a moment.
        </p>
      </>
    );
  }

  const totalRouted = Object.values(metrics.tier_distribution).reduce(
    (a, b) => a + b,
    0,
  );

  return (
    <>
      <h1 className="font-display text-3xl font-semibold tracking-tight">Metrics</h1>
      <p className="mt-2 text-muted">
        How routing has performed. {metrics.cost_basis}
      </p>

      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Questions" value={String(metrics.total_requests)} />
        <Stat
          label="Est. saved"
          value={`${metrics.estimated_saving_percent.toFixed(1)}%`}
          note={`${formatCost(metrics.estimated_saving_usd)} vs always strongest`}
        />
        <Stat
          label="Escalated"
          value={`${(metrics.escalation_rate * 100).toFixed(1)}%`}
          note="answers the check sent up a level"
        />
        <Stat
          label="Cache hits"
          value={`${(metrics.cache_hit_rate * 100).toFixed(1)}%`}
        />
      </div>

      <section className="mt-10">
        <h2 className="font-mono text-xs uppercase tracking-widest text-muted">
          Which model answered
        </h2>
        {totalRouted === 0 ? (
          <p className="mt-3 text-sm text-muted">No questions answered yet.</p>
        ) : (
          <>
            <div className="mt-4 flex h-3 overflow-hidden rounded-full">
              {(["T1", "T2", "T3"] as Tier[]).map((tier) => {
                const share = (metrics.tier_distribution[tier] ?? 0) / totalRouted;
                if (share === 0) return null;
                return (
                  <div
                    key={tier}
                    className={TIER_BG[tier]}
                    style={{ width: `${share * 100}%` }}
                    title={`${tier}: ${metrics.tier_distribution[tier]}`}
                  />
                );
              })}
            </div>
            <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs">
              {(["T1", "T2", "T3"] as Tier[]).map((tier) => (
                <div key={tier} className="flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${TIER_BG[tier]}`} />
                  <dt className={TIER_TEXT[tier]}>{tier}</dt>
                  <dd className="text-muted">
                    {metrics.tier_distribution[tier] ?? 0} (
                    {(
                      ((metrics.tier_distribution[tier] ?? 0) / totalRouted) *
                      100
                    ).toFixed(0)}
                    %)
                  </dd>
                </div>
              ))}
            </dl>
          </>
        )}
      </section>

      <section className="mt-10">
        <h2 className="font-mono text-xs uppercase tracking-widest text-muted">
          Speed by tier
        </h2>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[420px] border-collapse font-mono text-xs">
            <thead>
              <tr className="border-b border-ink/15 text-left text-muted">
                <th className="py-2 font-normal">Tier</th>
                <th className="py-2 font-normal">Answers</th>
                <th className="py-2 font-normal">P50</th>
                <th className="py-2 font-normal">P95</th>
                <th className="py-2 font-normal">Escalated</th>
              </tr>
            </thead>
            <tbody>
              {metrics.tier_stats.map((s) => (
                <tr key={s.tier} className="border-b border-ink/5">
                  <td className={`py-2 ${TIER_TEXT[s.tier]}`}>{s.tier}</td>
                  <td className="py-2">{s.count}</td>
                  <td className="py-2">{s.p50_latency_ms} ms</td>
                  <td className="py-2">{s.p95_latency_ms} ms</td>
                  <td className="py-2">{(s.escalation_rate * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {catalog && (
        <section className="mt-10">
          <h2 className="font-mono text-xs uppercase tracking-widest text-muted">
            Provider health
          </h2>
          <ul className="mt-4 space-y-2">
            {catalog.providers.map((p) => (
              <li
                key={p.provider}
                className="flex flex-wrap items-center gap-x-3 font-mono text-xs"
              >
                <span
                  className={`h-2 w-2 rounded-full ${p.reachable ? "bg-t1" : "bg-t3"}`}
                />
                <span className="text-ink">{p.provider}</span>
                <span className="text-muted">{p.model_count} models</span>
                {!p.reachable && <span className="text-t3">unreachable</span>}
              </li>
            ))}
          </ul>

          {catalog.models.some((m) => !m.advertised || m.circuit_open) && (
            <ul className="mt-4 space-y-1 text-xs text-muted">
              {catalog.models
                .filter((m) => !m.advertised || m.circuit_open)
                .map((m) => (
                  <li key={`${m.provider}/${m.model}`}>
                    <span className={TIER_TEXT[m.tier]}>{m.tier}</span>{" "}
                    {m.provider}/{m.model} —{" "}
                    {!m.advertised ? "no longer listed" : "temporarily skipped"}
                  </li>
                ))}
            </ul>
          )}

          <h3 className="mt-8 font-mono text-xs uppercase tracking-widest text-muted">
            Catalog changes
          </h3>
          {catalog.recent_drift.length === 0 ? (
            <p className="mt-3 text-sm text-muted">
              No changes seen since this service started.
            </p>
          ) : (
            <ul className="mt-3 space-y-1 text-xs text-muted">
              {catalog.recent_drift.map((d, i) => (
                <li key={i}>{d.detail}</li>
              ))}
            </ul>
          )}
        </section>
      )}

      <section className="mt-10 border-t border-ink/10 pt-5">
        <p className="font-mono text-xs text-muted">
          Student feedback: {metrics.feedback_positive} helpful,{" "}
          {metrics.feedback_negative} not
        </p>
      </section>
    </>
  );
}
