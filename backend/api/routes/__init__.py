"""v1 router aggregation.

One place to mount routers so ``main.py`` stays a thin app factory. Later
phases add their routers here:
  - ``eda``   -> POST /eda/reduce           (task 6.4)
  - ``ml``    -> POST /ml/predict           (task 9.4)
"""

from fastapi import APIRouter

from api.routes import data, eda, health

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(data.router)
api_v1_router.include_router(eda.router)

__all__ = ["api_v1_router"]
