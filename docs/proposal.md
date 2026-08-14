# Cascade — Project Proposal

## Problem

Frontier LLMs answer student questions well but cost far more per token than smaller models. Most study-assistant products call a single strong model for every question, regardless of whether the question needed it. That is wasteful, and it puts a real cost floor under any product that wants to serve students where money is the constraint.

Most questions do not need the most expensive model. Simple factual recall or basic arithmetic is answered just as correctly by a small, cheap model as by a frontier one. The problem is deciding, *before* generating an answer, which questions are simple and which are not — cheaply, and without sacrificing correctness on the hard ones.

## Approach

Cascade treats that decision as a learned classification problem, not a heuristic:

1. **A trained difficulty classifier** predicts whether a question needs a small model (T1), a mid-sized model (T2), or the strongest available free model (T3). Labels are empirical, not proxy: each training question is actually run through a T1 and a T2 model, graded against known-correct answers, and labeled by which tier first got it right. Features combine a hosted question embedding with handcrafted signals (token count, digit density, operator presence, question type, subject).
2. **A quality verifier with escalation** checks every cheap answer before it reaches the student. A fast heuristic pass catches obvious failures (empty output, refusals, language mismatch); an LLM-as-judge pass — running on a *different provider* than the answering model, to avoid self-preference bias — scores correctness and completeness. A failing score escalates the question one tier and retries, capped at two escalations.
3. **Catalog drift detection** protects the system against the instability of free provider tiers. A background poller tracks each provider's model list hourly; when a model disappears, the router removes it from tier resolution immediately rather than failing requests against a dead endpoint.

Provider adapters (Groq, Google AI Studio/Gemini, Cerebras, OpenRouter), the routing plumbing, and the frontend exist to support these three components. None of them are the contribution on their own — a thin proxy across providers is a solved problem (LiteLLM, OpenRouter itself). The decision layer on top is what's being built and evaluated here.

## Architecture

```
Next.js (Vercel) → FastAPI gateway (Render) → embedder, classifier, router,
provider adapters, verifier, escalation loop, catalog poller → Supabase Postgres
```

Full lifecycle, data contracts, and API surface are specified in `CLAUDE.md` §3, §6, §10.

## Constraints

- $0 inference budget (free provider tiers only) and $0 hosting budget (Render/Vercel/Supabase free tiers, 512 MB backend memory ceiling).
- Solo developer, final-year CS student, ~10–12 hours/week over 4 weeks.
- Cost figures reported by the product are counterfactual (computed from published pricing), always labeled as estimated, never presented as money actually spent or saved.

## Evaluation plan

The classifier is evaluated on a held-out 15% test split (touched once) with accuracy, macro-F1, a confusion matrix, and an ablation table (embeddings only / handcrafted features only / both) — the last of these is the single most important artifact in the project, since it is the evidence that the learned features are doing real work rather than the embedding alone carrying the signal. System-level metrics (quality retention vs. an always-T3 baseline, estimated cost reduction, escalation rate by predicted tier, latency percentiles per tier, cache hit rate, failover success rate, drift events observed) are collected continuously from Phase 2 onward and reported honestly in the final report, including where the system underperforms.

## Timeline

Six phases over four weeks: foundation → provider layer → classifier and verifier (the core) → service layer → frontend and deployment → evaluation and write-up. Full build order and phase gates are in `CLAUDE.md` §14. Feature scope is frozen at the end of Phase 3; no new scope is added after that point.

## Risks

- **Provider access from Bangladesh is unverified.** Phase 0 includes a hard gate: obtain keys and make one successful call to each of the four providers before any further work proceeds. If fewer than three work, the plan is revisited before continuing.
- **Free-tier rate limits and model deprecation** are treated as certainties, not edge cases — caching, backoff, and the catalog poller exist specifically to absorb this.
- **Scope discipline.** The classifier and verifier are the technical core; everything else is scaffolding. Time pressure will be resisted by cutting scaffolding polish before cutting into the core evaluation work.
