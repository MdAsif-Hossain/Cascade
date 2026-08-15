"""Diffing catalog snapshots into drift events.

Pure functions, no I/O, so the rules can be tested exhaustively without a
provider or a database. The poller does the talking; this decides what a change
means.

Asymmetric on purpose (CLAUDE.md §9): a disappeared model is acted on
immediately because tier resolution depends on it, while a new model is only
logged. Auto-adopting an unknown model would put untested capability in front of
students, and the whole point of the tier table is that every entry has been
verified callable (ADR-0002).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DriftKind = Literal["model_removed", "model_added", "provider_unreachable", "provider_recovered"]


@dataclass(frozen=True, slots=True)
class Drift:
    """One observed change in a provider's catalog."""

    provider: str
    model: str
    kind: DriftKind
    detail: str

    @property
    def is_breaking(self) -> bool:
        """True when the change can break tier resolution and needs action now."""
        return self.kind in ("model_removed", "provider_unreachable")


def diff_models(provider: str, previous: list[str], current: list[str]) -> list[Drift]:
    """Compare two model lists for one provider.

    An empty ``previous`` means this is the first snapshot, which is not drift —
    reporting every model as newly added on first run would bury the real signal
    the moment the system started.
    """
    if not previous:
        return []

    before, after = set(previous), set(current)

    removed = [
        Drift(
            provider=provider,
            model=model,
            kind="model_removed",
            detail=f"{model} is no longer advertised by {provider}",
        )
        for model in sorted(before - after)
    ]
    added = [
        Drift(
            provider=provider,
            model=model,
            kind="model_added",
            detail=f"{model} appeared in {provider}'s catalog; logged for review, not adopted",
        )
        for model in sorted(after - before)
    ]
    return removed + added


def diff_reachability(provider: str, was_reachable: bool, is_reachable: bool) -> list[Drift]:
    """Report a provider changing between reachable and unreachable."""
    if was_reachable and not is_reachable:
        return [
            Drift(
                provider=provider,
                model="*",
                kind="provider_unreachable",
                detail=f"{provider} could not be reached",
            )
        ]
    if not was_reachable and is_reachable:
        return [
            Drift(
                provider=provider,
                model="*",
                kind="provider_recovered",
                detail=f"{provider} is reachable again",
            )
        ]
    return []


def affected_tier_models(drifts: list[Drift], tier_models: dict[str, list[str]]) -> list[str]:
    """Which models Cascade actually depends on are broken by these drifts.

    A provider deleting a model nobody routes to is noise. This narrows the diff
    to the entries that would make a tier fail to resolve — the only ones worth
    interrupting anyone over.
    """
    broken: list[str] = []
    for drift in drifts:
        if not drift.is_breaking:
            continue
        for model in tier_models.get(drift.provider, []):
            if drift.model in ("*", model):
                broken.append(f"{drift.provider}/{model}")
    return sorted(set(broken))
