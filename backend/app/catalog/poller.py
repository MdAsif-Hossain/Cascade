"""Background catalog polling.

Free providers delete models without notice. Polling turns that from an outage
into a logged event and a routing decision made before a student's request hits
the dead model.

Runs hourly (§9). More often would spend quota on a question whose answer changes
on the order of days; less often leaves a window where a deleted model is still
being routed to.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from app.catalog.drift import Drift, diff_models, diff_reachability
from app.core.errors import ProviderError
from app.providers.base import Provider

logger = structlog.get_logger(__name__)

POLL_INTERVAL_SECONDS = 60 * 60

# Three consecutive failures trip the provider, matching the router's breaker so
# the two do not disagree about whether a provider is usable (§9).
UNREACHABLE_THRESHOLD = 3


@dataclass
class ProviderHealth:
    """Current view of one provider, as shown on the metrics dashboard."""

    provider: str
    reachable: bool = True
    models: list[str] = field(default_factory=list)
    consecutive_failures: int = 0
    last_checked: datetime | None = None
    last_error: str | None = None

    @property
    def is_circuit_broken(self) -> bool:
        return self.consecutive_failures >= UNREACHABLE_THRESHOLD


class CatalogPoller:
    """Polls provider catalogs, records snapshots, and emits drift events."""

    def __init__(
        self,
        providers: dict[str, Provider],
        *,
        interval_seconds: float = POLL_INTERVAL_SECONDS,
        on_drift: object = None,
    ) -> None:
        self._providers = providers
        self._interval = interval_seconds
        self._on_drift = on_drift
        self._health: dict[str, ProviderHealth] = {
            name: ProviderHealth(provider=name) for name in providers
        }
        self._drifts: list[Drift] = []
        self._task: asyncio.Task[None] | None = None

    @property
    def health(self) -> dict[str, ProviderHealth]:
        return dict(self._health)

    @property
    def recent_drifts(self) -> list[Drift]:
        return list(self._drifts)

    async def poll_once(self) -> list[Drift]:
        """Poll every provider once and return the drift observed."""
        drifts: list[Drift] = []

        for name, provider in self._providers.items():
            health = self._health[name]
            previous_models = list(health.models)
            was_reachable = health.reachable

            try:
                models = await provider.list_models()
            except ProviderError as exc:
                health.consecutive_failures += 1
                health.reachable = False
                health.last_error = str(exc)
                health.last_checked = datetime.now(UTC)
                drifts.extend(diff_reachability(name, was_reachable, False))
                logger.warning(
                    "catalog_poll_failed",
                    provider=name,
                    error=exc.error_code,
                    consecutive_failures=health.consecutive_failures,
                )
                continue

            health.consecutive_failures = 0
            health.reachable = True
            health.last_error = None
            health.last_checked = datetime.now(UTC)

            drifts.extend(diff_reachability(name, was_reachable, True))
            drifts.extend(diff_models(name, previous_models, models))
            health.models = models

        for drift in drifts:
            log = logger.warning if drift.is_breaking else logger.info
            log("catalog_drift", provider=drift.provider, model=drift.model, kind=drift.kind)

        self._drifts.extend(drifts)
        # Bounded: this list backs a dashboard panel, not an audit log. The
        # database holds the durable history.
        self._drifts = self._drifts[-100:]

        if drifts and callable(self._on_drift):
            self._on_drift(drifts)

        return drifts

    async def _loop(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the poller must never kill the app
                logger.error("catalog_poll_crashed", error=str(exc))
            await asyncio.sleep(self._interval)

    def start(self) -> None:
        """Start polling in the background. Safe to call twice."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
