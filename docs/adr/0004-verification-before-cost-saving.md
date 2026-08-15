# ADR-0004 — Verification is what makes cost reduction safe

**Status:** Accepted
**Date:** 2026-08-14
**Phase:** 2 (intelligence)

## Context

Routing questions to cheaper models saves money by definition. On its own, that is not a result — it is a quality reduction with a cost figure attached. Any system can be made arbitrarily cheap by always using the weakest model; what makes cheapness *defensible* is evidence that answers did not get worse.

The classifier will be wrong sometimes. Its predictions come from a question's text, before any answer exists, so it cannot know that a particular question happens to trip a small model up. Something has to catch that after the fact.

Three design questions had to be settled.

**Who judges?** An LLM scoring an answer is standard practice, but self-preference bias is well documented: models systematically rate their own outputs, and outputs from their own family, more highly. A judge sharing the answerer's provider produces numbers that look like quality measurement and are not.

**What happens when the judge is unavailable?** Free tiers rate-limit. If no cross-provider judge can be reached, the loop can either escalate (fail closed) or accept (fail open).

**When is a check worth paying for?** Every judge call is a second model call, spending the same free-tier quota the routing is trying to conserve.

## Decision

**Two stages, cheap first.** `verification/heuristics.py` runs with no model call and catches empty output, refusals, absurd length, and truncation. Only answers that survive reach the judge. A refusal costs nothing to detect and would be an obvious waste of a judge call.

**The judge provider is excluded before routing, not checked afterwards.** `Verifier.verify` passes `exclude_providers={answered_by_provider}` into the router, so a same-provider judge is structurally impossible rather than merely discouraged. A check applied after selection could be bypassed by a config change or a tier reshuffle; this cannot. Two tests assert it for both directions of the provider pairing.

**Fail open when no judge is reachable.** The answer already passed the cheap checks, and escalating every request during a provider outage would multiply load precisely when capacity is shortest. The verdict records `stage="judge_unavailable"` and the response carries a warning, so an unverified answer is visibly unverified rather than silently trusted.

**Escalate below 0.7, cap at two escalations.** Past the cap the best answer obtained is returned with a warning rather than an error — a checked-but-imperfect answer helps a student more than a 503 does.

## Consequences

**Positive**

- The cost claim becomes defensible. Cheap answers are checked by an unrelated model before reaching a student, so cost reduction is not silently traded against quality.
- The cross-provider rule is enforced by construction. It cannot regress without a test failing.
- The two-stage design keeps verification cheap: refused and truncated answers never reach a judge at all.

**Negative**

- Verification roughly doubles the model calls for a question that passes the heuristics. On a free tier that is real quota, and it is the main cost of the safety property.
- The judge is itself a cheap model and will sometimes be wrong. Its scores are a filter, not ground truth, and the report presents them as such.
- Failing open means some unverified answers reach students during an outage. This is a deliberate availability trade, made visible in the response rather than hidden.

**Neutral**

- The verifier's threshold (0.7) and escalation cap (2) come from the specification rather than from tuning. Tuning them would require a labelled set of good and bad answers, which this project does not have — the empirical labels measure whether a model was *correct*, not whether the judge agreed. This is named as a limitation rather than papered over with a tuned-looking number.
