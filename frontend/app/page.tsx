"use client";

import { useState } from "react";
import { RoutingTrace } from "@/components/RoutingTrace";
import {
  askQuestion,
  CascadeApiError,
  formatCost,
  formatLatency,
  sendFeedback,
  type AskResponse,
  type Subject,
} from "@/lib/api";

const SUBJECTS: Subject[] = ["general", "math", "science", "history", "language"];

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [subject, setSubject] = useState<Subject>("general");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [rated, setRated] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (question.trim().length < 3 || pending) return;

    setPending(true);
    setError(null);
    setResult(null);
    setRated(false);
    try {
      setResult(await askQuestion(question.trim(), subject));
    } catch (err) {
      setError(
        err instanceof CascadeApiError
          ? err.message
          : "Could not reach Cascade. Check your connection and try again.",
      );
    } finally {
      setPending(false);
    }
  }

  async function rate(helpful: boolean) {
    if (!result || rated) return;
    setRated(true);
    await sendFeedback(result.request_id, helpful);
  }

  return (
    <>
      <h1 className="font-display text-3xl font-semibold tracking-tight">
        Ask a question
      </h1>
      <p className="mt-2 max-w-xl text-muted">
        Cascade sends it to the cheapest model likely to get it right, has a second
        model check the answer, and moves up a level only if that check fails.
      </p>

      <form onSubmit={onSubmit} className="mt-8">
        <label htmlFor="question" className="sr-only">
          Your question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onSubmit(e);
          }}
          rows={3}
          maxLength={2000}
          placeholder="Why does ice float on water?"
          className="w-full resize-y rounded-lg border border-ink/15 bg-white px-4 py-3 text-lg leading-relaxed placeholder:text-muted/60"
        />

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <label htmlFor="subject" className="font-mono text-xs uppercase tracking-widest text-muted">
            Subject
          </label>
          <select
            id="subject"
            value={subject}
            onChange={(e) => setSubject(e.target.value as Subject)}
            className="rounded-md border border-ink/15 bg-white px-2 py-1.5 font-mono text-xs"
          >
            {SUBJECTS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>

          <button
            type="submit"
            disabled={pending || question.trim().length < 3}
            className="ml-auto rounded-md bg-ink px-5 py-2 font-medium text-paper disabled:opacity-40"
          >
            {pending ? "Thinking…" : "Ask"}
          </button>
        </div>
      </form>

      {error && (
        <div
          role="alert"
          className="mt-6 rounded-lg border border-t3/40 bg-t3/5 px-4 py-3 text-sm"
        >
          <p className="font-medium">That did not work.</p>
          <p className="mt-1 text-muted">{error}</p>
        </div>
      )}

      {!result && !error && !pending && (
        <p className="mt-12 border-t border-ink/10 pt-6 text-sm text-muted">
          Ask anything from your coursework. You will see which model answered and
          why, underneath the answer.
        </p>
      )}

      {result && (
        <article className="mt-10 border-t border-ink/10 pt-8">
          <div className="whitespace-pre-wrap text-lg leading-relaxed">
            {result.answer}
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-xs text-muted">
            <span>{formatLatency(result.total_latency_ms)}</span>
            <span>
              {formatCost(result.estimated_cost_usd)} est. ·{" "}
              {formatCost(result.baseline_cost_usd)} if always strongest
            </span>
            {result.cached && <span className="text-t1">from cache</span>}
            {result.escalated && <span className="text-t2">escalated</span>}
          </div>

          {result.warnings.length > 0 && (
            <ul className="mt-3 space-y-1 text-xs text-muted">
              {result.warnings.map((w, i) => (
                <li key={i}>Note: {w}</li>
              ))}
            </ul>
          )}

          <RoutingTrace steps={result.trace} />

          <div className="mt-8 flex items-center gap-3 border-t border-ink/10 pt-5">
            {rated ? (
              <p className="text-sm text-muted">Thanks — that helps.</p>
            ) : (
              <>
                <span className="text-sm text-muted">Was this helpful?</span>
                <button
                  onClick={() => rate(true)}
                  className="rounded-md border border-ink/15 px-3 py-1 text-sm hover:border-t1"
                >
                  Yes
                </button>
                <button
                  onClick={() => rate(false)}
                  className="rounded-md border border-ink/15 px-3 py-1 text-sm hover:border-t3"
                >
                  No
                </button>
              </>
            )}
          </div>
        </article>
      )}
    </>
  );
}
