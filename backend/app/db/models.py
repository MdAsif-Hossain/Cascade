"""Database schema: request traces, feedback, and catalog snapshots.

Everything the metrics dashboard and the final report need has to be recoverable
from these tables. Numbers are collected as the system runs rather than
reconstructed afterwards (CLAUDE.md §15), so each request persists its full
routing trace, not just its answer.

SQLite locally, Postgres on Supabase in production. Column types are kept to the
intersection both handle identically — JSON for the trace, plain integers for
token counts — so a query written against one behaves the same on the other.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class RequestRecord(Base):
    """One answered question, with the routing decision that produced it."""

    __tablename__ = "requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    question: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(String(20), default="general")
    level: Mapped[str] = mapped_column(String(20), default="school")
    answer: Mapped[str] = mapped_column(Text)

    predicted_tier: Mapped[str] = mapped_column(String(2), index=True)
    final_tier: Mapped[str] = mapped_column(String(2), index=True)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    prediction_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    prediction_source: Mapped[str] = mapped_column(String(32), default="model")

    # Stored as JSON rather than a child table: the trace is always read whole,
    # with the request, and never queried across rows. A join would buy nothing.
    trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    baseline_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)

    feedback: Mapped[list[FeedbackRecord]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )


class FeedbackRecord(Base):
    """A thumbs up or down on an answer.

    The only signal available about whether routing actually served the student,
    as opposed to whether the judge model approved of it.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id"), index=True)
    helpful: Mapped[bool] = mapped_column(Boolean)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    request: Mapped[RequestRecord] = relationship(back_populates="feedback")


class CatalogSnapshot(Base):
    """What a provider advertised, and what was actually callable, at one moment.

    Listing and callability are separate columns because they are separate facts:
    Gemini advertises models that return 404 when called (ADR-0002). A schema
    that recorded only the model list could not represent that state.
    """

    __tablename__ = "catalog_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    provider: Mapped[str] = mapped_column(String(32), index=True)
    models: Mapped[list[str]] = mapped_column(JSON, default=list)
    reachable: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DriftEvent(Base):
    """A change between two consecutive catalog snapshots."""

    __tablename__ = "drift_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    provider: Mapped[str] = mapped_column(String(32), index=True)
    model: Mapped[str] = mapped_column(String(128))
    # "model_removed" is the one that breaks tier resolution and triggers action.
    # "model_added" is logged for review and never auto-adopted (§9).
    kind: Mapped[str] = mapped_column(String(32), index=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
