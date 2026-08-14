# Cascade

An AI study assistant that routes each question to the cheapest model likely to answer it correctly, has a **different provider** check the answer, and escalates only when that check fails.

Built to run entirely on free provider tiers — $0 inference, $0 hosting.

**Status:** Phases 0–4 complete. Live URL pending deploy.

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

**The difficulty classifier did not beat the majority-class baseline.** On the held-out test split it scored **84.00% accuracy against a baseline of 84.00%** — a difference of exactly zero. It learned to predict T1 almost always, which is what predicting the majority class does.

| Metric | Value |
|---|---|
| Test accuracy | 0.840 |
| Majority-class baseline | 0.840 |
| Improvement over baseline | **+0.000** |
| Macro-F1 | 0.358 |
| Test split size | 150 questions |

Per class:

| Tier | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T1 | 0.86 | 0.98 | 0.92 | 126 |
| T2 | 0.67 | 0.09 | 0.16 | 22 |
| T3 | 0.00 | 0.00 | 0.00 | 2 |

It catches 2 of 22 T2 questions and 0 of 2 T3 questions. See `docs/final-report.md` for why, and what that means for the design.

### Ablation — the study that did work

Validation split. This is the artifact that answers whether each feature family earns its place.

| Model | Features | Dims | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| logistic | handcrafted | 23 | 0.389 | 0.227 |
| logistic | embedding | 256 | 0.638 | 0.302 |
| logistic | both | 279 | 0.671 | 0.304 |
| boosting | handcrafted | 23 | 0.832 | 0.329 |
| boosting | embedding | 256 | 0.805 | 0.393 |
| **boosting** | **both** | **279** | **0.812** | **0.420** |

Macro-F1 rises monotonically — handcrafted 0.329, embeddings 0.393, both 0.420 — so both families carry signal and the combination is additive. Gradient boosting beats logistic regression on every feature set. No configuration beats the baseline on raw accuracy.

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

Expected calibration error **0.1185** on validation. The model is overconfident in its 0.7–0.8 band (predicted 0.747, actual 0.375). Confidence is therefore *not* used as a routing input.

### Measured provider latency

| Provider | Model | Latency |
|---|---|---|
| Groq | `llama-3.1-8b-instant` | 126 ms |
| Gemini | `gemini-flash-lite-latest` | 908 ms |
| OpenRouter | `openai/gpt-oss-20b:free` | 27.5 s |

Tier candidate ordering follows measured latency, not price.

## Architecture

```
Next.js (Vercel)
      │  HTTPS
      ▼
FastAPI gateway (Render, Docker)
      ├─ Embedder ──────────► gemini-embedding-001
      ├─ Difficulty classifier (sklearn, 569 KB .joblib)
      ├─ Router ────────────► tier selection + failover + circuit breaker
      ├─ Provider adapters ─► Groq │ Gemini │ OpenRouter
      ├─ Verifier ──────────► heuristics, then cross-provider LLM judge
      ├─ Escalation loop
      └─ Catalog poller ────► drift detection
      ▼
Supabase Postgres
```

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

305 tests. `ruff` and `mypy --strict` clean. All HTTP is mocked with `respx`, so the suite passes offline with no API keys set.

```bash
cd backend && pytest -q
```

The provider contract suite (`tests/providers/contract.py`) is 28 shared assertions every adapter must satisfy — written before any adapter existed. It caught three real bugs, documented in `docs/adr/0002-catalog-listing-is-not-callability.md`.

## Docs

- [`docs/final-report.md`](docs/final-report.md) — full write-up, including what did not work
- [`docs/proposal.md`](docs/proposal.md) — original proposal
- [`docs/adr/`](docs/adr/) — architecture decision records
  - [0001](docs/adr/0001-provider-selection.md) — why three providers, not four
  - [0002](docs/adr/0002-catalog-listing-is-not-callability.md) — a listed model is not a callable model
  - [0003](docs/adr/0003-empirical-labels-and-pacing.md) — empirical labels, and pacing by tokens
  - [0004](docs/adr/0004-verification-before-cost-saving.md) — verification is what makes cheapness safe
- [`ops/README.md`](ops/README.md) — deployment notes
