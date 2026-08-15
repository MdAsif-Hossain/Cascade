<div align="center">

# Cascade

**An AI study assistant that routes each question to the cheapest model that can answer it — then has a *different* provider check the answer before it reaches the student.**

Frontier-quality tutoring at a fraction of frontier cost. Running end to end on free tiers: **$0 inference, $0 hosting.**

[![CI](https://github.com/MdAsif-Hossain/Cascade/actions/workflows/ci.yml/badge.svg)](https://github.com/MdAsif-Hossain/Cascade/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-323-4B7B6E)
![mypy](https://img.shields.io/badge/mypy-strict-4B7B6E)
![Python](https://img.shields.io/badge/python-3.11-3776AB)
![Next.js](https://img.shields.io/badge/Next.js-14-000000)

### [→ Try it live](https://cascade-red-eight.vercel.app) · [API docs](https://cascade-api-5w68.onrender.com/docs) · [Full report](docs/final-report.md)

*First request may take ~50s while the free-tier backend wakes.*

</div>

---

![Answer with routing trace](docs/screenshots/ask-answered.png)

<div align="center"><i>Every answer ships with its routing trace — which model replied, how fast, what the checker scored it. Transparency is the feature, not debug output.</i></div>

---

## The idea

Most student questions don't need an expensive model. **I measured this: 84.2% of 1,249 benchmark questions were answered correctly by the cheapest model I had access to.** The hard part isn't saving money — it's knowing *which* questions you can be cheap about, without silently giving someone a worse answer.

Cascade's answer to that is three components:

```
question → difficulty classifier → cheapest capable tier → answer
                                          ↓
                          verifier (heuristics, then an LLM judge
                          on a different provider than the answerer)
                                          ↓
                          pass → return    fail → escalate one tier
```

## What makes this interesting

**A classifier trained on empirical labels, not proxies.** Every training label was produced by actually running the question through a real T1 model and a real T2 model and grading the output against ground truth. The label *is* the routing decision, which is why it beats guessing from question length or dataset tags.

**A verifier that can't cheat.** Models systematically prefer their own output, so the judge is structurally barred from running on the provider that produced the answer — `exclude_providers` is passed into the router before selection, not checked afterwards. Two tests assert it in both directions.

**Catalog drift detection that earned itself.** Gemini advertises `gemini-2.5-flash`, `gemini-2.5-pro` and `gemini-2.5-flash-lite` in its model list. All three return HTTP 404 when called. A tier table built from the catalog was silently unroutable — [ADR-0002](docs/adr/0002-catalog-listing-is-not-callability.md).

**A published negative result.** The classifier ties a majority-class baseline. That's in the README, the report headline, and the abstract — not buried. More on that below.

## Results

All measured. Reproduce with `ml/eval/run_eval.py`.

### The ablation — the finding I'd defend

Does each feature family earn its keep, or is one of them decoration?

| Model | Features | Dims | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| logistic | handcrafted | 22 | 0.396 | 0.230 |
| logistic | embedding | 256 | 0.638 | 0.302 |
| logistic | both | 278 | 0.671 | 0.304 |
| boosting | handcrafted | 22 | 0.846 | 0.334 |
| boosting | embedding | 256 | 0.805 | 0.393 |
| **boosting** | **both** | **278** | **0.819** | **0.423** |

Macro-F1 climbs monotonically — 0.334 → 0.393 → 0.423 — so both families carry signal and combining them is genuinely additive. That's what justifies putting an embedding API call on the hot path.

### The headline result is negative

**On the held-out test split the classifier scored 83.33% against an 84.00% majority-class baseline** — marginally worse than always guessing the most common tier. It finds 2 of 22 T2 questions and 0 of 2 T3.

| Tier | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T1 | 0.85 | 0.98 | 0.91 | 126 |
| T2 | 0.50 | 0.09 | 0.15 | 22 |
| T3 | 0.00 | 0.00 | 0.00 | 2 |

Why: the labels are 84% / 14% / 1.4%, so there's almost no minority class to learn from — and whether a model gets a question right may be a property of *that model's training*, not of anything readable in the question text.

**What makes the failure survivable is the verifier.** 14% of questions get routed too cheaply; those are caught by a second model from a different provider and escalated. Cost reduction without that check would just be quality reduction. The component I'd filed as plumbing turned out to be the one holding the system up.

The report states this in §3, discloses that the test split was evaluated twice and why, and lists ten limitations in §8.

### Cost and latency

| Metric | Value |
|---|---|
| Estimated cost reduction vs. always-T3 | 97.6% |
| Predicted tier sufficient | 86.0% |
| Groq / Gemini / OpenRouter latency | 126 ms / 908 ms / 27.5 s |

Cost figures are **counterfactual** — computed from published per-token list prices. Every call ran on a free tier; no money was spent. And 97.6% is largely a consequence of the classifier predicting T1 almost always, which is a caveat I state rather than a number I lead with.

## Architecture

![Cascade architecture](docs/architecture.png)

**Request lifecycle:** cache lookup on `sha256(question + subject + level)` → embed → classify → route to a healthy model in that tier (failing over within it) → heuristic screen → cross-provider judge → escalate if scored below 0.7, capped at two → persist the full trace.

**Tiers are capability bands, not providers**, so a tier still resolves when a provider disappears. Every model was verified callable before being listed.

| Tier | For | Preferred model |
|---|---|---|
| T1 | recall, definitions, simple arithmetic | `groq/llama-3.1-8b-instant` |
| T2 | multi-step reasoning, short derivations | `groq/llama-3.3-70b-versatile` |
| T3 | hard multi-step maths, subtle reasoning | `gemini/gemini-3.5-flash` |

## Engineering

**323 tests**, `ruff` and `mypy --strict` clean, CI on every push. The suite passes **offline with no API keys set** — which is what proves nothing quietly reaches the network.

**The provider contract suite was written before any adapter existed.** 28 shared assertions that all three adapters must satisfy. Three providers implemented independently will look correct and behave differently — different token-usage keys, different error envelopes — and those divergences don't crash, they corrupt your cost metric. The suite caught exactly that.

Some bugs worth reading about, all documented:

| Bug | Why it mattered |
|---|---|
| Reasoning models return `content: null` with `finish_reason: length` | HTTP 200 carrying no answer. Returning it as empty would make the verifier score a provider truncation as question difficulty |
| OpenRouter reports upstream 502s **inside** an HTTP 200 | Classified as a permanent fault when it was a transient outage that should fail over |
| Rate limits are metered in **tokens**, not requests | Groq allows 14,400 req/day but 6,000 tokens/min. A token-budget pacer took the labelling run from 9 errors in 60 to **1 in 1,249** |
| `num_choices` train/inference leak | 76% of training rows had 4 answer options; a real question has 0. Fixed, with a regression test |

## Stack

**Backend** Python 3.11 · FastAPI · Pydantic v2 · httpx (async) · SQLAlchemy 2.0 · scikit-learn · structlog · tenacity
**Frontend** Next.js 14 App Router · TypeScript strict · Tailwind
**Data** SQLite locally · Supabase Postgres in production
**Testing** pytest · respx · ruff · mypy --strict · GitHub Actions
**Deploy** Docker on Render · Vercel · cron keep-alive

## Run it locally

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # .venv/Scripts/activate on Windows
pip install -r requirements-dev.txt
cp ../.env.example ../.env                          # add your free API keys
uvicorn app.main:app --reload                       # → http://127.0.0.1:8000/docs

# Frontend
cd frontend && npm install && npm run dev           # → http://localhost:3000

# Reproduce the ML pipeline
cd ml
python label_empirical.py     # empirical labels, caches every response
python train_classifier.py    # train + persist
python eval/ablation.py       # ablation table
python eval/run_eval.py       # final test evaluation — run once
```

Needs free keys from [Groq](https://console.groq.com), [Google AI Studio](https://aistudio.google.com) and [OpenRouter](https://openrouter.ai).

## Repository

```
backend/app/
  api/v1/routes/   ask, metrics, health, requests, feedback, catalog
  core/            config, logging, errors, cache
  providers/       base contract, groq, gemini, openrouter, embeddings
  routing/         classifier, router, tiers, features, breaker
  verification/    verifier, heuristics
  catalog/         poller, drift
  services/        ask — the escalation loop
ml/                labelling, training, ablation, calibration, evaluation
frontend/app/      ask, history, metrics
docs/adr/          architecture decision records
```

## Documentation

| | |
|---|---|
| [Final report](docs/final-report.md) ([.docx](docs/Cascade-Final-Report.docx)) | Full write-up, including what didn't work |
| [ADR-0001](docs/adr/0001-provider-selection.md) | Why three providers, not four |
| [ADR-0002](docs/adr/0002-catalog-listing-is-not-callability.md) | A listed model is not a callable model |
| [ADR-0003](docs/adr/0003-empirical-labels-and-pacing.md) | Empirical labels, and pacing by tokens |
| [ADR-0004](docs/adr/0004-verification-before-cost-saving.md) | Verification is what makes cheapness safe |
| [ops/](ops/README.md) | Deployment notes |
| [ml/results/](ml/results/) | Raw evaluation output |

---

<div align="center">

Capstone project · Ostad AI Engineering · **[Md. Asif Hossain](https://github.com/MdAsif-Hossain)**

</div>
