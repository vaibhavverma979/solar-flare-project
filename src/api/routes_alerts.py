"""
routes_alerts.py — REST endpoints for nowcast/forecast alerts (Phase 5).
TODO (Phase 5): Implement /alerts/list and /alerts/{id} endpoints.
"""
from fastapi import APIRouter
router = APIRouter(prefix="/alerts", tags=["alerts"])
