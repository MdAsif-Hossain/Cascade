# CLAUDE.md — Cascade

> Persistent instructions for Claude Code. Read this fully at the start of every session.
> If a request conflicts with this file, say so before acting.

---

## 1. What this project is

**Cascade** is an AI study assistant that answers student questions at a fraction of normal inference cost.

A student asks a question. Cascade predicts how hard it is, routes it to the cheapest model likely to answer it correctly, checks the answer, and escalates to a stronger model only when the check fails.

**Product promise:** frontier-quality tutoring at a small fraction of frontier cost, so it works where money is the barrier.

**Engineering thesis:** most questions do not need the most expensive model. Deciding *which* ones do is a learnable problem.

### The three things that make this project

1. **A trained difficulty classifier.** Not keyword rules. A real model with accuracy figures, a confusion matrix, and an ablation study.
2. **A quality verifier with escalation.** The cheap answer is checked; failures retry on a stronger tier. This is what makes cost reduction safe.
3. **Catalog drift detection.** Free-tier providers silently delete models. Cascade polls provider catalogs and fails over before requests break.

**If these three are not built, the project has failed** — regardless of how polished everything else is. Provider adapters, routing plumbing, and UI are scaffolding around these three. Do not let scaffolding consume the schedule.

### Anti-goal

Do not rebuild LiteLLM or OpenRouter. A proxy that forwards requests to several providers is a solved problem and adds nothing. What is being built here is the *decision layer* on top.

---

## 2. Hard constraints

| Constraint | Value |
|---|---|
| Inference budget | $0. Free provider tiers only. |
| Hosting budget | $0. Render free + Vercel free + Supabase free. |
| Backend memory ceiling | 512 MB (Render free tier) |
| Timeline | 4 weeks, roughly 10–12 hours per week |
| Owner | Solo developer, final-year CS student |

**Consequences that are not negotiable:**

- No local transformer models in the backend. Embeddings come from a hosted API. The only model artifact shipped is a scikit-learn classifier (kilobytes).
- Never store uploaded or generated files on Render's disk. It is ephemeral.
- Cost figures are **counterfactual**, computed from published per-token pricing. Every surface that displays cost must label it as estimated. Never imply money was actually spent or saved.
- Assume every free tier will rate-limit you. Cache aggressively, back off, and degrade gracefully.

---

## 3. Architecture

```
Next.js (Vercel)
      │  HTTPS
      ▼
FastAPI gateway (Render, Docker)
      │
      ├─ Embedder ──────────► Gemini embedding API (gemini-embedding-001)
      ├─ Difficulty classifier (sklearn, local .joblib)
      ├─ Router ────────────► tier selection + provider choice
      ├─ Provider adapters ─► Groq │ Gemini │ OpenRouter
      ├─ Verifier ──────────► LLM-as-judge (different provider than answerer)
      ├─ Escalation loop
      └─ Catalog poller (background) ─► provider model lists
      │
      ▼
Supabase Postgres  (requests, traces, catalog snapshots, feedback)
```

### Request lifecycle

1. `POST /api/v1/ask` receives `{question, subject?, level?}`
2. Cache lookup on `sha256(normalized_question)`. Hit → return immediately, `cached: true`.
3. Embed the question.
4. Classifier predicts tier: `T1 | T2 | T3`.
5. Router selects a healthy provider serving that tier.
6. Adapter calls the provider. On failure, fail over within the tier.
7. Verifier scores the answer. Pass → return. Fail → escalate one tier, repeat (max 2 escalations).
8. Persist a full trace row. Return answer plus routing metadata.

**The routing metadata is part of the product, not debug output.** The student sees which model answered, whether it was escalated, and why. Transparency is the feature.

---

## 4. Stack — use exactly these

**Backend**
- Python 3.11, FastAPI, Pydantic v2
- `httpx` (async) for all provider calls — never `requests`
- SQLAlchemy 2.0 + Alembic migrations
- SQLite locally, Supabase Postgres in production
- `scikit-learn` + `joblib` for the classifier
- `structlog` for structured JSON logging
- `tenacity` for retry/backoff

**Frontend**
- Next.js 14 (App Router), TypeScript strict mode
- Tailwind CSS, shadcn/ui
- No state library. Server components plus `useState`.

**Testing / CI**
- `pytest`, `pytest-asyncio`, `respx` (mock httpx — never hit real providers in tests)
- `ruff` (lint + format), `mypy` in strict mode on `app/`
- GitHub Actions on every push

**Deployment**
- Backend: Docker on Render
- Frontend: Vercel
- Keep-alive ping every 10 minutes against `/health`

Do not add dependencies beyond this list without asking. Every addition needs a stated reason.

---

## 5. Repository layout

```
cascade/
├── CLAUDE.md
├── README.md
├── docs/
│   ├── proposal.md
│   ├── final-report.md
│   ├── architecture.png
│   ├── adr/                    # architecture decision records
│   └── screenshots/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/v1/routes/      # ask.py, metrics.py, health.py, feedback.py
│   │   ├── core/               # config.py, logging.py, errors.py, cache.py
│   │   ├── providers/          # base.py, groq.py, gemini.py, openrouter.py
│   │   ├── routing/            # classifier.py, router.py, tiers.py
│   │   ├── verification/       # verifier.py, heuristics.py
│   │   ├── catalog/            # poller.py, drift.py
│   │   ├── schemas/
│   │   └── db/                 # models.py, session.py
│   ├── artifacts/              # classifier.joblib, metadata.json
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── ml/
│   ├── data/{raw,labeled}/
│   ├── notebooks/
│   ├── train_classifier.py
│   ├── label_empirical.py
│   ├── eval/{run_eval.py,ablation.py}
│   └── results/
├── frontend/
├── .github/workflows/ci.yml
└── .env.example
```

---

## 6. Data contracts

Define these in `schemas/` before writing any logic that touches them.

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    subject: Literal["math","science","history","language","general"] = "general"
    level: Literal["school","undergrad"] = "school"

class RoutingStep(BaseModel):
    tier: Literal["T1","T2","T3"]
    provider: str
    model: str
    latency_ms: int
    verifier_score: float | None
    outcome: Literal["accepted","escalated","provider_error"]

class AskResponse(BaseModel):
    answer: str
    trace: list[RoutingStep]
    predicted_tier: Literal["T1","T2","T3"]
    final_tier: Literal["T1","T2","T3"]
    escalated: bool
    estimated_cost_usd: float      # counterfactual
    baseline_cost_usd: float       # cost if T3 had been used directly
    total_latency_ms: int
    cached: bool
    request_id: str
```

### Tier definitions (`routing/tiers.py`)

Tiers are capability bands, not providers. Each tier lists candidate models across providers with published per-token prices. When a provider disappears, the tier still resolves.

- **T1** — small fast models. Factual recall, definitions, simple arithmetic.
- **T2** — mid models. Multi-step reasoning, explanation, short derivations.
- **T3** — strongest available free models. Hard multi-step math, subtle reasoning.

---

## 7. The classifier — build this properly

This carries the largest share of the project's technical credibility. Do not shortcut it.

### Labels

Use **empirical difficulty**, not proxy difficulty. Proxy labels (question length, dataset tags) are weak and indefensible. Empirical labels are generated by measuring what actually happens:

1. Take ~1,200 questions from public datasets: ARC-Easy, ARC-Challenge, GSM8K, SciQ, OpenBookQA. These have known-correct answers.
2. Run every question through a T1 model and a T2 model.
3. Grade each answer against ground truth automatically.
4. Label: T1 correct → `T1`. T1 wrong, T2 correct → `T2`. Both wrong → `T3`.

This label *is* the routing decision, which is why it beats any proxy. Write this as `ml/label_empirical.py`, run it in batches respecting rate limits, and cache every response to disk so it never re-runs.

### Model

- Features: question embedding + handcrafted features (token count, digit density, presence of math operators, question-word type, subject one-hot)
- Estimator: `LogisticRegression` first. Only try `GradientBoostingClassifier` if logistic underperforms, and report both.
- Split: 70/15/15 stratified. The test split is touched **once**, at the end.
- Persist to `backend/artifacts/classifier.joblib` with a `metadata.json` recording training date, dataset sizes, and metrics.

### Required outputs for the report

- Accuracy, macro-F1, per-class precision/recall
- Confusion matrix (saved as PNG)
- **Ablation table**: embeddings only / handcrafted only / both. This is the single most important artifact in the whole project.
- Calibration: predicted-tier confidence vs. actual escalation rate

---

## 8. The verifier

Two stages, cheap first.

**Heuristics** (no model call): empty or truncated output, refusal patterns, answer language mismatch, absurd length.

**LLM-as-judge:** a cheap model scores the answer 0–1 on correctness and completeness given the question.

**Critical rule:** the judge must run on a **different provider** than the model that produced the answer. Self-preference bias is well documented, and using the same family invalidates the measurement. Encode this as a hard constraint in the router and cover it with a test.

Escalate when `score < 0.7`. Cap at 2 escalations. Log every escalation with its reason.

---

## 9. Catalog drift detection

A background task polls each provider's model-list endpoint hourly and writes a snapshot to `catalog_snapshots`.

On diff:
- Model disappeared → mark unhealthy, remove from tier resolution, log a warning
- New model appeared → log for manual review, never auto-adopt
- Provider unreachable 3 times consecutively → circuit-break for 15 minutes

Surface drift events on the metrics dashboard. This is the feature that proves the system is built for the real world rather than a demo.

---

## 10. API surface

```
POST /api/v1/ask                    ask a question
GET  /api/v1/requests               paginated history
GET  /api/v1/requests/{id}          single trace
POST /api/v1/feedback               thumbs up/down on an answer
GET  /api/v1/metrics                aggregate stats for the dashboard
GET  /api/v1/catalog                current provider/model health
GET  /health                        liveness (used by keep-alive)
```

Every endpoint: Pydantic request and response models, so `/docs` is accurate and usable as documentation.

**Errors** are always `{"error_code": str, "message": str, "request_id": str}`. Never leak a stack trace. Never return a bare 500.

---

## 11. Frontend

Three routes. Build the quality floor without announcing it: responsive to mobile, visible keyboard focus, `prefers-reduced-motion` respected.

**`/` — Ask.** A single input, prominent. Answer renders below with the routing trace beneath it.

**`/history` — past questions**, filterable by tier and escalation status.

**`/metrics` — the dashboard.** Cost saved vs. baseline, tier distribution, escalation rate, P50/P95 latency by tier, provider health, drift events.

### Design direction

The subject is a question descending a ladder of models. That journey is the interface's one memorable idea.

**Signature element — the routing trace.** A horizontal ladder under each answer showing every step: tier, model, latency, verifier score, and outcome. An escalation is drawn as a visible step upward, not a footnote. This is the element the whole page is built around; keep everything else quiet.

**Palette — tier colors carry meaning, they are not decoration.** The same three colors identify tiers everywhere: trace, badges, charts, history rows.

```
paper    #F2F4F1   cool off-white, green cast
ink      #14171A
muted    #6B7280
T1       #4B7B6E   teal
T2       #C08A2E   ochre
T3       #8B4A6B   plum
```

**Type — three roles.**
- Display: `Fraunces` — headings only, used with restraint
- Body: `Public Sans` — answers and prose
- Utility: `JetBrains Mono` — the trace, metrics, model names. Telemetry should read as telemetry.

**Explicitly avoid:** cream background with terracotta accent; near-black with acid green; broadsheet hairline-rule layouts. These are current AI-design defaults and a reviewer will recognise them.

**Copy rules.** Active voice, sentence case. Name things the way a student would: "Checked by a second model," not "verifier pass." Errors state what happened and what to do next. The empty state on `/` invites a first question rather than apologising for having no data.

---

## 12. Git discipline — read this before every commit

Version control is graded and cannot be fixed retroactively. Follow these exactly.

- **One logical change per commit.** Never bundle a feature, a refactor, and a test fix together. If a commit message needs "and", split it.
- **Conventional commits:** `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`
- **Commit at every green test run.** Do not accumulate a day of work into one commit.
- **Branch per feature**, PR into `main`, squash-merge with a descriptive title. Solo development is not an excuse to skip this — the PR history is the design record.
- **Never commit** `.env`, API keys, `data/raw/`, `node_modules/`, `.joblib` files larger than 5 MB, or notebook outputs.
- **No attribution trailers.** Never add `Co-Authored-By:` lines, `🤖 Generated with Claude Code` footers, or any other tool attribution to commit messages or PR descriptions. The repository owner is the sole author of record.
- Tag milestones: `v0.1-proposal`, `v0.5-mvp`, `v1.0-final`

**Target: 40+ commits across 20+ distinct calendar days.** A grader looks at the contribution graph. Density matters as much as content.

---

## 13. Code standards

- Type hints on every function. `mypy --strict` must pass on `app/`.
- No bare `except:`. Catch specific exceptions and log with context.
- No secrets in code. Everything through `core/config.py` via Pydantic `BaseSettings`.
- No business logic in route handlers — routes validate, delegate to a service, and serialise.
- Every provider adapter subclasses `providers/base.py` and satisfies the shared contract test suite.
- Structured logging with a `request_id` propagated through the full lifecycle.
- Docstrings on public functions explaining *why*, not *what*.

### Test requirements

- **Write the provider contract test suite before writing any adapter.** With four providers, plausible-looking adapters will differ subtly in token counting and error handling. One shared suite catches this. This is not optional and not reorderable.
- Mock all HTTP with `respx`. Tests must pass offline with no API keys set.
- Cover: each adapter against the contract, classifier load and predict, router tier resolution, verifier escalation logic, judge-provider-differs-from-answerer, cache hit/miss, drift detection, every endpoint's happy path and error path.
- Target 35+ tests. CI must be green before any merge to `main`.

---

## 14. Build order

Complete each phase before starting the next. Do not work ahead.

### Phase 0 — Foundation
- [x] Repo, structure, `.gitignore`, `.env.example`, this file committed
- [x] GitHub Actions CI with one passing test
- [x] Verify provider access from Bangladesh — **3/4 pass.** Groq, Gemini, OpenRouter work; Cerebras returns HTTP 402 on every model and is dropped. See `docs/adr/0001-provider-selection.md`.
- [x] `docs/proposal.md` (1–2 pages)

### Phase 1 — Provider layer
- [ ] `providers/base.py` contract
- [ ] **Contract test suite, written first**
- [ ] Three adapters, all passing the suite
- [ ] Retry, backoff, circuit breaker
- [ ] Tier definitions with published pricing table
- [ ] Milestone: one question routed to a hardcoded tier returns an answer

### Phase 2 — Intelligence (the core)
- [ ] `label_empirical.py` — generate labels, cache all responses
- [ ] `train_classifier.py` — features, training, persistence
- [ ] Ablation study and confusion matrix
- [ ] Verifier: heuristics + LLM-as-judge with cross-provider constraint
- [ ] Escalation loop
- [ ] Milestone: classifier metrics and ablation table exist as real numbers

### Phase 3 — Service
- [ ] Database schema and migrations
- [ ] All endpoints with full validation and error handling
- [ ] Caching, structured logging
- [ ] Catalog poller and drift detection
- [ ] Milestone: full API working end to end locally

### Phase 4 — Interface and deployment
- [ ] Three frontend routes per the design direction above
- [ ] Docker build, Render deploy, Vercel deploy, CORS, keep-alive
- [ ] Milestone: public URL works from a phone on mobile data

### Phase 5 — Evidence
- [ ] Final evaluation on the held-out test split — run once
- [ ] README: pitch, live URL, architecture diagram, metrics table, setup, screenshots
- [ ] ADRs for the four biggest decisions
- [ ] `docs/final-report.md` (3–5 pages)
- [ ] Tag `v1.0-final`

**Feature freeze at the end of Phase 3.** No new scope after that point.

---

## 15. Numbers the final report needs

Collect these as you build, not at the end.

| Metric | Source |
|---|---|
| Classifier accuracy, macro-F1, confusion matrix | Phase 2 |
| Ablation: embeddings / handcrafted / both | Phase 2 |
| Quality retention vs. always-T3 baseline | Phase 2 |
| Estimated cost reduction (labelled counterfactual) | Phase 2 |
| Escalation rate by predicted tier | Phase 3 |
| P50 / P95 latency per tier | Phase 3 |
| Cache hit rate | Phase 3 |
| Failover success rate | Phase 3 |
| Drift events observed | Phase 3+ |

**Report honestly.** If cost reduction is 60% rather than 90%, write 60%. If the classifier confuses T2 and T3, say so and show the confusion matrix. A limitations section that names real weaknesses reads as competence; one that claims everything worked reads as untested.

---

## 16. Working rules for Claude Code

- **Ask before deviating** from this file. Do not silently substitute a library, skip a phase, or restructure directories.
- **Do not write large amounts of code in one pass.** Small increments, tests, commit, repeat.
- **Never fabricate metrics.** If a number has not been measured, write `TBD`. Never place a plausible-looking figure in the README or report.
- **If a provider's API differs from expectation, stop and report** rather than guessing at the request shape.
- **Flag scope creep.** If a request would add work beyond this spec, say so and estimate the cost before starting.
- When ambiguity arises, prefer the option that produces a measurable number over the one that produces a feature.
