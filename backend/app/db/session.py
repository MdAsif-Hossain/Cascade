"""Database engine and session management.

Synchronous SQLAlchemy inside a thread pool rather than async: the async
Postgres drivers add a dependency and a failure mode, and the write volume here
is one row per question. The bottleneck is a several-hundred-millisecond model
call, not the database.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _connect_args(url: str) -> dict[str, object]:
    # SQLite guards connections against cross-thread use; FastAPI's threadpool
    # legitimately hands them between threads, so that guard has to be relaxed.
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def normalise_url(url: str) -> str:
    """Accept the ``postgres://`` scheme some hosts still hand out.

    SQLAlchemy 2.0 removed support for that spelling and raises an obscure
    "Can't load plugin" error. Rewriting it here means a connection string
    pasted straight from a dashboard works instead of failing at startup with a
    message that does not name the problem.
    """
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = normalise_url(get_settings().database_url)
        _engine = create_engine(url, connect_args=_connect_args(url), pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def create_tables() -> None:
    """Create any missing tables.

    Alembic owns migrations for real schema changes; this exists so a fresh
    checkout and the test suite have a working database without a migration step.
    """
    Base.metadata.create_all(bind=get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional session that commits on success and rolls back on failure."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop cached engine and factory. Used by tests switching database URLs."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
