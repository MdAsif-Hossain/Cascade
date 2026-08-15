# ADR-0001 — Provider selection: three providers, not four

**Status:** Accepted
**Date:** 2026-08-14
**Phase:** 0 (foundation gate)

## Context

The original spec named four providers: Groq, Google AI Studio (Gemini), Cerebras, and OpenRouter. Phase 0 required obtaining a key for each and making one successful inference call, with an explicit gate: if fewer than three work, stop and revisit the plan.

All four keys were obtained and tested from Bangladesh on 2026-08-14. Each provider's model-list endpoint was called first (to discover the real request shape rather than assume it), then a single small chat completion.

| Provider | Model list | Inference | Evidence |
|---|---|---|---|
| Groq | 200, 15 models | 200, 122 ms | `llama-3.1-8b-instant` answered; 45 total tokens |
| Google AI Studio | 200, 37 text models | 200, 785 ms | `gemini-flash-lite-latest` answered; 8 total tokens |
| OpenRouter | 200, 411 models (15 `:free`) | 200 | `openai/gpt-oss-20b:free` answered; `is_free_tier: true`, usage 0 |
| Cerebras | 200, 3 models | **402 on all 3** | `payment_required_error` — "Payment required to access this resource" |

The Cerebras key authenticates correctly — the catalog request succeeds — but every model (`zai-glm-4.7`, `gemma-4-31b`, `gpt-oss-120b`) returns HTTP 402 on chat completion. Cerebras no longer grants free inference on this account tier. Enabling it would require billing, which violates the project's $0 inference constraint (§2).

Two further measurements were taken because they constrain later phases:

- **Groq rate limits:** 14,400 requests/day but only **6,000 tokens/minute**. Tokens, not requests, are the binding constraint on the ~2,400-call empirical labeling run in Phase 2.
- **Free models rate-limit independently.** `google/gemma-4-31b-it:free` returned an upstream 429 in the same second that three other OpenRouter free models answered normally.

## Decision

Ship **three** provider adapters: Groq, Gemini, and OpenRouter. Cerebras is dropped entirely — no adapter, no tier entry, no key in the environment template.

Embeddings are served by the **same Google AI Studio key** via `gemini-embedding-001` (verified: 3072 dimensions, HTTP 200). No separate embedding provider or key is required.

## Consequences

**Positive**

- Three providers is sufficient for the verifier's hard cross-provider constraint (§8): any answer from one provider can always be judged by one of two others.
- One less adapter to write, test, and maintain — roughly an hour returned to the classifier and verifier, which are the actual technical contribution (§1).
- The embedder needs no new signup, no new key, and no new failure mode.

**Negative**

- Failover breadth within a tier is narrower. If two of three providers are simultaneously rate-limited, a tier can fail to resolve. The circuit breaker and tier fallback must handle this rather than assume a healthy provider always exists.
- The project depends on Google for both generation and embeddings. A Google outage degrades two subsystems at once. This is accepted: the alternative is a second embedding signup for a component that is called once per request and cached.

**Neutral**

- `CLAUDE.md` §3 and §5 were updated to remove Cerebras, so the spec continues to describe the system as actually built.
- Two provider-behaviour findings from this gate are carried into the Phase 1 contract test suite: reasoning models return `content: null` with `finish_reason: "length"` when the token budget is too small to escape the reasoning phase, and free-tier models return upstream 429s independently of one another.
