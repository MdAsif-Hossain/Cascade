# Cascade

An AI study assistant that routes each question to the cheapest model likely to answer it correctly, has a **different provider** check the answer, and escalates only when that check fails.

Built to run entirely on free provider tiers — $0 inference, $0 hosting.

Capstone project for the Ostad AI Engineering programme.

**Live:** https://cascade-red-eight.vercel.app
**API:** https://cascade-api-5w68.onrender.com ([docs](https://cascade-api-5w68.onrender.com/docs))

Both are on free tiers, so the first request after a quiet spell can take up to 50 seconds while the backend wakes.

---

## What it does

```
question → difficulty classifier → cheapest capable tier → answer
                                          ↓
                          verifier (heuristics, then a judge on
                          a different provider than the answerer)
                                          ↓
                          pass → return    fail → escalate one tier
```

Every answer ships with its routing trace — which model answered, how long it took, what the checker scored it, and whether it was escalated. That transparency is a product feature, not debug output.

## Key results

All figures measured, not estimated. Reproduce with `ml/eval/run_eval.py`.

### The honest headline

**The difficulty classifier did not beat the majority-class baseline.** On the held-out test split it scored **83.33% accuracy against a baseline of 84.00%** — marginally *worse* than always guessing the most common tier.

| Metric | Value |
|---|---|
| Test accuracy | 0.8333 |
| Majority-class baseline | 0.8400 |
| Improvement over baseline | **−0.0067** |
| Macro-F1 | 0.3550 |
| Test split size | 150 questions |

Per class:

| Tier | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T1 | 0.85 | 0.98 | 0.91 | 126 |
| T2 | 0.50 | 0.09 | 0.15 | 22 |
| T3 | 0.00 | 0.00 | 0.00 | 2 |

It catches 2 of 22 T2 questions and 0 of 2 T3 questions.

It is not degenerate, though — it does discriminate. On the live site it routes "What is photosynthesis?" to T1 at 0.99 confidence and "Prove that the square root of 2 is irrational" to T2 at 0.90. Those are anecdotes; the table above is the measurement. See [`docs/final-report.md`](docs/final-report.md) for why it fails and what that means for the design.

### Ablation — the study that did work

Validation split. This is the artifact that answers whether each feature family earns its place.

| Model | Features | Dims | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| logistic | handcrafted | 22 | 0.396 | 0.230 |
| logistic | embedding | 256 | 0.638 | 0.302 |
| logistic | both | 278 | 0.671 | 0.304 |
| boosting | handcrafted | 22 | 0.846 | 0.334 |
| boosting | embedding | 256 | 0.805 | 0.393 |
| **boosting** | **both** | **278** | **0.819** | **0.423** |

Macro-F1 rises monotonically — handcrafted 0.334, embeddings 0.393, both 0.423 — so both families carry signal and the combination is additive. Gradient boosting beats logistic regression on every feature set. No configuration beats the baseline on accuracy; boosting on handcrafted features alone ties it exactly, which is what a model that has learned to always say T1 looks like.

### Empirical labels

1,249 questions from ARC-Easy, ARC-Challenge, GSM8K, SciQ and OpenBookQA, each run through a real T1 model and, on failure, a real T2 model, then graded against known-correct answers.

| Label | Count | Share |
|---|---:|---:|
| T1 — small model was enough | 1,052 | 84.2% |
| T2 — needed a mid model | 179 | 14.3% |
| T3 — both failed | 18 | 1.4% |

**84.2% of questions were answered correctly by the cheapest model.** That is the project's central premise confirmed — and simultaneously the reason the classifier struggles, since there is very little minority class to learn from.

### Cost

| Metric | Value |
|---|---|
| Estimated cost reduction vs. always-T3 | 97.6% |
| Predicted tier sufficient | 86.0% |
| Under-predicted (verifier must escalate) | 14.0% |

**These are counterfactual.** Every call ran on a free tier; no money was spent. Costs come from published per-token list prices retrieved from OpenRouter's API on 2026-08-14. The 97.6% figure is largely a consequence of the classifier predicting T1 almost always, and must be read together with the 14% under-prediction rate that the verifier then has to absorb.

### Calibration

Expected calibration error **0.1258** on validation. The model is overconfident where it matters most: in the 0.8–1.0 band, which holds 136 of 149 validation questions, it claims 0.946 and is right 0.824 of the time. Confidence is therefore *not* used as a routing input.

### Measured provider latency

| Provider | Model | Latency |
|---|---|---|
| Groq | `llama-3.1-8b-instant` | 126 ms |
| Gemini | `gemini-flash-lite-latest` | 908 ms |
| OpenRouter | `openai/gpt-oss-20b:free` | 27.5 s |

Tier candidate ordering follows measured latency, not price.

## Screenshots

The routing trace under an answer — the element the interface is built around. An escalation is drawn as a visible step upward, not a footnote.

![Answer with routing trace](docs/screenshots/ask-answered.png)

| History | Metrics |
|---|---|
| ![History](docs/screenshots/history.png) | ![Metrics](docs/screenshots/metrics.png) |

Captured from the live site.

## Architecture

![Cascade architecture](docs/architecture.png)

```
Next.js (Vercel)
      │  HTTPS
      ▼
FastAPI gateway (Render, Docker)
      ├─ Embedder ──────────► gemini-embedding-001
      ├─ Difficulty classifier (sklearn, 565 KB .joblib)
      ├─ Router ────────────► tier selection + failover + circuit breaker
      ├─ Provider adapters ─► Groq │ Gemini │ OpenRouter
      ├─ Verifier ──────────► heuristics, then cross-provider LLM judge
      ├─ Escalation loop
      └─ Catalog poller ────► drift detection
      ▼
Supabase Postgres
```

### Request lifecycle

1. `POST /api/v1/ask` receives the question
2. Cache lookup on `sha256(normalised question + subject + level)` — a hit returns immediately
3. The question is embedded and the classifier predicts a tier
4. The router picks a healthy model in that tier, failing over within it on error
5. The verifier screens the answer with heuristics, then a judge on a **different provider**
6. Score below 0.7 escalates one tier and repeats, capped at two escalations
7. The full trace is persisted and returned alongside the answer

### Tiers

Capability bands, not providers — each lists candidates across several providers so a tier still resolves when one disappears. Every model was verified callable before being listed, because Gemini advertises models that return 404 (see ADR-0002).

| Tier | For | Preferred model |
|---|---|---|
| T1 | recall, definitions, simple arithmetic | `groq/llama-3.1-8b-instant` |
| T2 | multi-step reasoning, short derivations | `groq/llama-3.3-70b-versatile` |
| T3 | hard multi-step maths, subtle reasoning | `gemini/gemini-3.5-flash` |

## Stack

**Backend** — Python 3.11, FastAPI, Pydantic v2, `httpx` (async), SQLAlchemy 2.0, scikit-learn, `structlog`, `tenacity`
**Frontend** — Next.js 14 App Router, TypeScript strict, Tailwind
**Data** — SQLite locally, Supabase Postgres in production
**Testing** — `pytest`, `respx`, `ruff`, `mypy --strict`, GitHub Actions

## Layout

```
backend/app/
  api/v1/routes/   ask, metrics, health, requests, feedback, catalog
  core/            config, logging, errors, cache
  providers/       base contract, groq, gemini, openrouter, embeddings
  routing/         classifier, router, tiers, features, breaker
  verification/    verifier, heuristics
  catalog/         poller, drift
  services/        ask (the escalation loop)
ml/
  label_empirical.py   empirical difficulty labelling
  train_classifier.py  training and persistence
  eval/                ablation, calibration, final evaluation
frontend/app/      ask, history, metrics
```

## Deployment

| Component | Host | Notes |
|---|---|---|
| Frontend | Vercel | root directory `frontend`, one env var: `NEXT_PUBLIC_API_BASE` |
| API | Render | Docker, `backend/Dockerfile`, build context `backend`, free tier |
| Database | Supabase | Postgres via the session pooler (port 5432 — the direct host is IPv6-only) |
| Keep-alive | cron-job.org | pings `/health` every 10 minutes so the free tier doesn't sleep |

Render needs `GROQ_API_KEY`, `GOOGLE_AI_STUDIO_API_KEY`, `OPENROUTER_API_KEY`, `DATABASE_URL` and `ENVIRONMENT`. Vercel needs only `NEXT_PUBLIC_API_BASE` — provider keys must never go there, since anything prefixed `NEXT_PUBLIC_` is compiled into the browser bundle. Full notes in [`ops/README.md`](ops/README.md).

## Setup

```bash
# Backend
cd backend
python -m venv .venv && .venv/Scripts/activate    # or source .venv/bin/activate
pip install -r requirements-dev.txt
cp ../.env.example ../.env                        # then add your API keys
uvicorn app.main:app --reload                     # http://127.0.0.1:8000/docs

# Frontend
cd frontend
npm install
npm run dev                                       # http://localhost:3000

# Reproduce the ML pipeline
cd ml
python label_empirical.py      # generates labels, caches every response
python train_classifier.py     # trains and persists the classifier
python eval/ablation.py        # ablation table
python eval/calibration.py     # calibration curve
python eval/run_eval.py        # final test evaluation — run once
```

Requires free API keys for [Groq](https://console.groq.com), [Google AI Studio](https://aistudio.google.com), and [OpenRouter](https://openrouter.ai). See `.env.example`.

## Tests

323 tests. `ruff` and `mypy --strict` clean. All HTTP is mocked with `respx`, so the suite passes offline with no API keys set — which is also what proves nothing quietly reaches the network.

```bash
cd backend && pytest -q
```

The provider contract suite (`tests/providers/contract.py`) is 28 shared assertions every adapter must satisfy, written before any adapter existed. It caught three real bugs, documented in [ADR-0002](docs/adr/0002-catalog-listing-is-not-callability.md).

Two tests worth pointing at:

- `test_verifier.py::TestCrossProviderConstraint` — asserts the judge never runs on the provider that produced the answer, in both directions
- `test_features.py::test_answer_options_do_not_change_the_features` — a regression guard for a train/inference feature leak that shipped once already

## What didn't work

The classifier ties a majority-class baseline. The ablation is a real finding, the verifier demonstrably absorbs the 14% of questions routed too cheaply, and catalog drift caught Gemini advertising three models that 404 — but the headline component underperformed, and [the report](docs/final-report.md) says so in §3 and lists ten limitations in §7.

## Docs

- [`docs/final-report.md`](docs/final-report.md) — full write-up, including what did not work ([.docx](docs/Cascade-Final-Report.docx))
- [`docs/proposal.md`](docs/proposal.md) — original proposal
- [`docs/adr/`](docs/adr/) — architecture decision records
  - [0001](docs/adr/0001-provider-selection.md) — why three providers, not four
  - [0002](docs/adr/0002-catalog-listing-is-not-callability.md) — a listed model is not a callable model
  - [0003](docs/adr/0003-empirical-labels-and-pacing.md) — empirical labels, and pacing by tokens
  - [0004](docs/adr/0004-verification-before-cost-saving.md) — verification is what makes cheapness safe
- [`ops/README.md`](ops/README.md) — deployment notes
- [`ml/results/`](ml/results/) — raw evaluation output: ablation, calibration, test results, confusion matrix

## Licence

Coursework project, submitted as the capstone for the Ostad AI Engineering programme.
