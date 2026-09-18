"""Shared FastAPI dependencies.

Settings injection plus the request-scoped database session (task 1.7). Routes
annotate with ``SettingsDep`` / ``SessionDep`` and never import the engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from db.session import get_sessionmaker

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_db_session(settings: SettingsDep) -> Iterator[Session]:
    """One session per request, committed on success and rolled back on error.

    Commit lives here rather than in each route so a handler that raises after a
    partial write cannot leave the transaction half-applied. Routes that only
    read pay nothing for it -- a commit with no changes is a no-op.
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


SessionDep = Annotated[Session, Depends(get_db_session)]

__all__ = ["SessionDep", "SettingsDep", "get_db_session", "get_settings"]
