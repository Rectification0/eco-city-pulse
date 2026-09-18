"""Engine and session lifecycle.

The engine is created lazily and cached per connection URL. That matters for
two reasons: importing ``main`` must never require a reachable database (the
test suite does exactly that), and a single pooled engine per process is the
only way connection pooling actually works.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import Settings, get_settings

# Keyed by connection URL, not by Settings: Settings is not hashable by value,
# and the URL is what actually distinguishes one engine from another. Held in a
# plain dict rather than lru_cache so the engines can be disposed on shutdown.
_ENGINES: dict[str, Engine] = {}
_ENGINE_LOCK = threading.Lock()


def get_engine(settings: Settings | None = None) -> Engine:
    """Process-wide engine for the configured database."""
    settings = settings or get_settings()
    url = settings.database_url

    engine = _ENGINES.get(url)
    if engine is not None:
        return engine

    with _ENGINE_LOCK:
        # Re-check inside the lock: two requests can race here on first use.
        engine = _ENGINES.get(url)
        if engine is None:
            engine = create_engine(
                url,
                echo=settings.db_echo,
                # The database container can be restarted out from under a
                # pooled connection; pre_ping turns that into a transparent
                # reconnect instead of a 500 on the next request.
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                future=True,
            )
            _ENGINES[url] = engine
    return engine


def get_sessionmaker(settings: Settings | None = None) -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(settings),
        autoflush=False,
        # Attributes stay readable after commit, so a route can serialise an
        # ORM object it just wrote without triggering a second query.
        expire_on_commit=False,
        class_=Session,
    )


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Transactional scope for scripts and background jobs.

    The request path uses ``api.dependencies.get_db_session`` instead; this is
    the same contract for code that has no request to hang off.
    """
    session = get_sessionmaker(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engines() -> None:
    """Drop every pooled connection. Used by tests and shutdown hooks."""
    with _ENGINE_LOCK:
        for engine in _ENGINES.values():
            engine.dispose()
        _ENGINES.clear()


__all__ = [
    "dispose_engines",
    "get_engine",
    "get_sessionmaker",
    "session_scope",
]
