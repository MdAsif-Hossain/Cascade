# ADR-0002 — A model being listed is not evidence it can be called

**Status:** Accepted
**Date:** 2026-08-14
**Phase:** 1 (provider layer)

## Context

Catalog drift detection was specified around a single signal: poll each provider's model-list endpoint, diff the snapshots, and treat a model's disappearance from the list as the failure to react to (`CLAUDE.md` §9).

While verifying that every model named in `routing/tiers.py` actually answers, three failures appeared that this design would not have caught.

**1. Listed but dead.** Gemini's `/v1beta/models` endpoint advertises `gemini-2.5-flash`, `gemini-2.5-pro`, and `gemini-2.5-flash-lite`. All three return HTTP 404 when called:

> This model models/gemini-2.5-flash is no longer available to new users.

The model never disappears from the catalog, so a listing diff reports no drift while every request to it fails. The tier had been built from the catalog and was silently unroutable.

**2. Listed, alive, but not for us.** `gemini-pro-latest` and `gemini-3.1-pro-preview` are listed and are not deprecated, but return HTTP 429 quota-exceeded on the free tier for any request. They are real models that this project can never use.

**3. Success status, failure body.** OpenRouter returned `HTTP 200` carrying `{"error": {"message": "Upstream error from Nvidia", "code": 502}}` for `nvidia/nemotron-3-ultra-550b-a55b:free`. The adapter had been treating the HTTP status as authoritative and classified this as a malformed response — a permanent, non-retryable fault — when it was a transient upstream outage that should have failed over.

## Decision

**Listing and callability are tracked as two separate facts.** A model is eligible for tier resolution only if it is both advertised *and* has answered.

1. Tier membership requires a successful live call, recorded with its date. Every model in `routing/tiers.py` was verified on 2026-08-14 before being listed there.
2. The Phase 3 catalog poller records a *callability* probe alongside each listing snapshot, and a drift event fires on either signal changing — not just on a model leaving the list.
3. Adapters inspect the response body of successful HTTP calls for an error envelope, and let the inner status code override the outer one (`providers/_http.py::_effective_status`). Three contract tests cover this for every adapter.

## Consequences

**Positive**

- The failure mode that would have been most embarrassing in a demo — a tier that resolves to a model returning 404 for every student — is now impossible to introduce silently.
- Gateway-proxied upstream outages are classified as retryable and fail over, instead of being recorded as permanent faults against a model that is fine.
- The drift dashboard gets a more honest signal. "Advertised but uncallable" is a real, observable state that a listing-only diff cannot express, and it is the state Gemini is actually in.

**Negative**

- Probing costs requests against free-tier quotas. The poller must probe on a slower cycle than it lists, and cache results, or it becomes the largest consumer of the very quota it is protecting.
- Tier definitions now carry a verification date and go stale. This is a maintenance cost accepted deliberately: a stale-but-dated table is honest, while an unverified one is confidently wrong.

**Neutral**

- Band membership in `tiers.py` is ordered by published price rather than by model name, because names do not order reliably — Google prices `gemini-3.5-flash` at four times `gemini-3.7-flash` despite the higher version number.
