from datetime import datetime, timezone, timedelta
from langchain_core.tools import tool
from src.database.connection import AsyncSessionLocal
from sqlalchemy import select, and_
from src.database.models import FlareEvent, SolexsTelemetry

@tool
async def query_historical_flare_events(start_time: str, end_time: str, min_class: str) -> dict:
    """
    Executes a parameterized SQL query on the flare_events table to retrieve past events.
    Use this when a scientist asks about specific historical timestamps or flare classes.
    start_time: ISO-8601 string (e.g., '2023-01-01T00:00:00Z')
    end_time: ISO-8601 string (e.g., '2023-01-02T00:00:00Z')
    min_class: Flare class character (e.g., 'M', 'X') to search for.
    """
    try:
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        async with AsyncSessionLocal() as session:
            stmt = select(FlareEvent).where(
                and_(
                    FlareEvent.peak_timestamp >= start_dt,
                    FlareEvent.peak_timestamp <= end_dt,
                    FlareEvent.flare_class.ilike(f"{min_class}%")
                )
            )
            
            res = (await session.execute(stmt)).scalars().all()
            events = [
                {
                    "event_id": r.event_id, 
                    "peak_time": r.peak_timestamp.isoformat(),
                    "class": r.flare_class
                } 
                for r in res
            ]
            return {"status": "Success", "events": events}
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@tool
async def get_lightcurve_timeseries(minutes: int = 30) -> dict:
    """
    Returns an array of timestamps and flux values from SoLEXS for specific lightcurve analysis over the last N minutes.
    minutes: number of minutes to look back (default 30)
    """
    try:
        lookback = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        async with AsyncSessionLocal() as session:
            stmt = select(SolexsTelemetry.timestamp, SolexsTelemetry.soft_xray_flux).where(SolexsTelemetry.timestamp >= lookback).order_by(SolexsTelemetry.timestamp)
            res = await session.execute(stmt)
            data = res.all()
            timestamps = [r[0].isoformat() for r in data]
            flux = [r[1] for r in data]
            return {"status": "Success", "timestamps": timestamps, "flux": flux}
    except Exception as e:
        return {"status": "Error", "message": str(e)}
