"""v1 router aggregation.

One place to mount routers so ``main.py`` stays a thin app factory. Later
phases add their routers here:
  - ``data``  -> GET /data/sources          (task 2.10)
  - ``eda``   -> POST /eda/profile, /reduce (tasks 4.3, 6.4)
  - ``ml``    -> POST /ml/predict           (task 9.4)
"""

from fastapi import APIRouter

from api.routes import data, health

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(data.router)

__all__ = ["api_v1_router"]
