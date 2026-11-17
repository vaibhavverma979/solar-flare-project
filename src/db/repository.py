"""
repository.py — Database access layer (Phase 5).

All database I/O goes through this module.  No other module imports SQLAlchemy
directly.  This abstraction means switching from SQLite to TimescaleDB or
InfluxDB requires changes only here.

TODO (Phase 5): Implement all repository methods below.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import pandas as pd



@runtime_checkable
class LightCurveRepository(Protocol):
    """Protocol that any storage backend must implement."""

    def insert_lightcurve(self, df: pd.DataFrame) -> int:
        """Bulk-insert a light-curve DataFrame; return number of rows inserted."""
        ...

    def query_lightcurve(
        self,
        instrument: str,
        start: str,
        end: str,
        energy_channel: str | None = None,
    ) -> pd.DataFrame:
        """Return all light-curve points for *instrument* in [start, end]."""
        ...


@runtime_checkable
class FlareEventRepository(Protocol):
    """Protocol for the master flare catalogue."""

    def upsert_event(self, event: dict) -> str:
        """Insert or update a flare event; return event_id."""
        ...

    def list_events(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Return the master catalogue as a DataFrame."""
        ...


class SQLiteRepository:
    """
    Concrete repository implementation using SQLAlchemy + SQLite.

    TODO (Phase 5): Implement using models.py ORM classes.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url

    def insert_lightcurve(self, df: pd.DataFrame) -> int:
        raise NotImplementedError("Phase 5 — not yet implemented")

    def query_lightcurve(self, instrument, start, end, energy_channel=None):
        raise NotImplementedError("Phase 5 — not yet implemented")

    def upsert_event(self, event: dict) -> str:
        raise NotImplementedError("Phase 5 — not yet implemented")

    def list_events(self, start=None, end=None):
        raise NotImplementedError("Phase 5 — not yet implemented")
