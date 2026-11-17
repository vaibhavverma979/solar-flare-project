from fastapi import APIRouter
from sqlalchemy import text
from src.database.connection import AsyncSessionLocal

router = APIRouter(prefix="/api/insitu", tags=["Geomagnetic Status"])


@router.get("/geomagnetic-status")
async def get_geomagnetic_status():
    """
    Returns a 24-hour historical trend of Dst and Kp indices from geomagnetic_forecast.
    Falls back to simulated data if the table is empty.
    """
    async with AsyncSessionLocal() as session:
        try:
            query = text("""
                SELECT
                    forecast_target_time AS timestamp,
                    predicted_dst AS dst,
                    predicted_kp AS kp,
                    storm_category AS scale
                FROM geomagnetic_forecast
                ORDER BY forecast_target_time DESC
                LIMIT 1440
            """)
            result = await session.execute(query)
            rows = result.fetchall()

            if rows:
                data = [dict(r._mapping) for r in rows]
                return {"status": "ok", "source": "database", "data": data}
            else:
                # Fallback: simulated data when DB is empty
                import math, random
                from datetime import datetime, timedelta, timezone

                now = datetime.now(timezone.utc)
                data = []
                for i in range(24 * 4):  # 15-minute intervals
                    t = now - timedelta(minutes=15 * i)
                    dst = -10.0 + random.uniform(-5, 5)
                    kp = 1.0 + random.uniform(0, 1)
                    data.append({
                        "timestamp": t.isoformat(),
                        "dst": round(dst, 2),
                        "kp": round(kp, 2),
                        "scale": "Quiet",
                    })
                return {"status": "ok", "source": "simulated", "data": data}

        except Exception as e:
            return {"status": "error", "message": str(e), "data": []}


@router.get("/space-weather-impact")
async def get_space_weather_impact():
    """
    Returns the latest 100 rows from space_weather_impact table.
    """
    async with AsyncSessionLocal() as session:
        try:
            query = text("""
                SELECT
                    timestamp,
                    kp_index,
                    dst_index,
                    imf_b_z,
                    earth_impact_eta,
                    spectral_hardness_ratio,
                    risk
                FROM space_weather_impact
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            result = await session.execute(query)
            rows = result.fetchall()

            if rows:
                data = [dict(r._mapping) for r in rows]
                return {"status": "ok", "count": len(data), "data": data}
            else:
                return {"status": "ok", "count": 0, "data": [], "message": "Table is empty. Run seed_impact.py or start the mock generator."}

        except Exception as e:
            return {"status": "error", "message": str(e), "data": []}


@router.get("/flare-events")
async def get_flare_events():
    """
    Returns the latest 50 flare events from the flare_events table.
    """
    async with AsyncSessionLocal() as session:
        try:
            query = text("""
                SELECT
                    event_id,
                    peak_timestamp,
                    end_timestamp,
                    flare_class,
                    confidence_score,
                    neupert_flag
                FROM flare_events
                ORDER BY peak_timestamp DESC
                LIMIT 50
            """)
            result = await session.execute(query)
            rows = result.fetchall()

            if rows:
                data = [dict(r._mapping) for r in rows]
                return {"status": "ok", "count": len(data), "data": data}
            else:
                return {"status": "ok", "count": 0, "data": [], "message": "No flare events found. Run seed_impact.py or start the mock generator."}

        except Exception as e:
            return {"status": "error", "message": str(e), "data": []}
