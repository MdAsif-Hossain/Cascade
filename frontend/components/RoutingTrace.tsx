import type { RoutingStep, Tier } from "@/lib/api";
import { formatLatency, TIER_LABEL } from "@/lib/api";

/**
 * The routing trace — the one memorable idea in the interface.
 *
 * A question descending a ladder of models, drawn as a ladder. Each step sits at
 * a height matching its tier, so an escalation is a visible step *upward* rather
 * than a footnote. Everything else on the page stays quiet so this reads first.
 */

const TIER_ROW: Record<Tier, number> = { T3: 0, T2: 1, T1: 2 };

const TIER_STYLE: Record<Tier, { dot: string; text: string; border: string }> = {
  T1: { dot: "bg-t1", text: "text-t1", border: "border-t1" },
  T2: { dot: "bg-t2", text: "text-t2", border: "border-t2" },
  T3: { dot: "bg-t3", text: "text-t3", border: "border-t3" },
};

function outcomeLabel(step: RoutingStep): string {
  switch (step.outcome) {
    case "accepted":
      return "Answered";
    case "escalated":
      return "Sent up a level";
    case "provider_error":
      return "Unavailable";
  }
}

export function RoutingTrace({ steps }: { steps: RoutingStep[] }) {
  if (steps.length === 0) return null;

  return (
    <section aria-labelledby="trace-heading" className="mt-10">
      <h2
        id="trace-heading"
        className="font-mono text-xs uppercase tracking-widest text-muted"
      >
        How this was answered
      </h2>

      {/* The ladder. Three rows, strongest at the top, so upward means harder. */}
      <div className="mt-4 overflow-x-auto">
        <div className="min-w-[520px]">
          {([2, 1, 0] as const).map((row) => {
            const tier = (Object.keys(TIER_ROW) as Tier[]).find(
              (t) => TIER_ROW[t] === row,
            )!;
            return (
              <div key={tier} className="flex items-stretch">
                <div className="w-28 shrink-0 border-r border-ink/10 py-3 pr-3 text-right">
                  <span
                    className={`font-mono text-xs font-medium ${TIER_STYLE[tier].text}`}
                  >
                    {tier}
                  </span>
                  <span className="ml-2 hidden text-xs text-muted sm:inline">
                    {TIER_LABEL[tier]}
                  </span>
                </div>

                <div className="relative flex flex-1 items-center gap-3 py-3 pl-4">
                  {steps.map((step, index) =>
                    step.tier === tier ? (
                      <StepCard key={index} step={step} index={index} />
                    ) : (
                      <div
                        key={index}
                        className="h-px w-32 shrink-0"
                        aria-hidden="true"
                      />
                    ),
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* A plain-language reading of the same journey, which is also what a
          screen reader announces — the ladder above is decorative for them. */}
      <ol className="mt-4 space-y-1 text-sm text-muted">
        {steps.map((step, index) => (
          <li key={index}>
            <span className="font-mono text-xs">{step.tier}</span>{" "}
            {step.provider === "-" ? (
              <>no model in this tier was available</>
            ) : (
              <>
                {step.model} answered in {formatLatency(step.latency_ms)}
                {step.verifier_score !== null && (
                  <> and scored {step.verifier_score.toFixed(2)} when checked</>
                )}
              </>
            )}
            . {outcomeLabel(step)}.
          </li>
        ))}
      </ol>
    </section>
  );
}

function StepCard({ step, index }: { step: RoutingStep; index: number }) {
  const style = TIER_STYLE[step.tier];
  const failed = step.outcome === "provider_error";

  return (
    <div
      className={`w-32 shrink-0 rounded-md border bg-white/60 px-2 py-2 ${
        failed ? "border-dashed border-muted/50" : style.border
      }`}
      aria-hidden="true"
    >
      <div className="flex items-center gap-1.5">
        <span
          className={`h-1.5 w-1.5 rounded-full ${failed ? "bg-muted" : style.dot}`}
        />
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted">
          step {index + 1}
        </span>
      </div>
      <p className="mt-1 truncate font-mono text-[11px] text-ink" title={step.model}>
        {step.provider === "-" ? "unavailable" : step.model}
      </p>
      <p className="mt-0.5 font-mono text-[10px] text-muted">
        {failed ? "—" : formatLatency(step.latency_ms)}
        {step.verifier_score !== null && ` · ${step.verifier_score.toFixed(2)}`}
      </p>
    </div>
  );
}
