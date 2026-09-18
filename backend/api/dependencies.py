"""Shared FastAPI dependencies.

Phase 0 provides settings injection only. The database session dependency
lands with the data layer (task 1.7) and is added here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from core.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]

__all__ = ["SettingsDep", "get_settings"]
