import asyncpg
import asyncio
from langchain_core.tools import tool
from config import settings
from datetime import datetime, timedelta

async def get_db_connection():
    """Helper to get asyncpg connection from settings."""
    url = settings.timescaledb_url.replace("+asyncpg", "")
    return await asyncpg.connect(url)

@tool
async def get_nowcast_status() -> dict:
    """
    Queries the TimescaleDB `flare_events` table for any active flares detected in the last 2 hours.
    Use this tool to check for real-time solar flare events currently happening.
    """
    try:
        conn = await get_db_connection()
        try:
            query = """
            SELECT event_id, start_time, peak_time, end_time, class_estimated
            FROM flare_events
            WHERE start_time >= NOW() - INTERVAL '2 hours'
            """
            rows = await conn.fetch(query)
            if rows:
                events = [dict(row) for row in rows]
                return {"status": "Active flares found", "flares": events}
            return {"status": "No active flares detected in the last 2 hours.", "flares": []}
        except (asyncpg.exceptions.UndefinedTableError, Exception) as e:
            print(f"Using mock flare events data. DB Error: {e}")
            # Mock fallback
            return {
                "status": "Mocked active flares",
                "flares": [
                    {
                        "event_id": "FL-2026-08-16-01",
                        "start_time": (datetime.utcnow() - timedelta(minutes=45)).isoformat(),
                        "peak_time": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
                        "end_time": None,
                        "class_estimated": "M2.1"
                    }
                ]
            }
        finally:
            if 'conn' in locals():
                await conn.close()
    except Exception as e:
        print(f"Fallback triggered in get_nowcast_status: {e}")
        return {"status": "Mocked active flares (DB connection failed)", "flares": []}

@tool
async def get_forecast_predictions() -> dict:
    """
    Queries the latest PatchTST 30-minute flare probability and expected Dst/Kp indices.
    Use this tool to answer questions about predictive models and future space weather conditions.
    """
    try:
        conn = await get_db_connection()
        try:
            query = """
            SELECT flare_probability, predicted_dst, predicted_kp
            FROM forecasts
            ORDER BY timestamp DESC
            LIMIT 1
            """
            row = await conn.fetchrow(query)
            if row:
                return {
                    "flare_probability": row["flare_probability"],
                    "predicted_dst": row["predicted_dst"],
                    "predicted_kp": row["predicted_kp"]
                }
            return {"status": "No forecast predictions available."}
        except (asyncpg.exceptions.UndefinedTableError, Exception) as e:
            print(f"Using mock forecast data. DB Error: {e}")
            return {
                "flare_probability": 0.82,
                "predicted_dst": -42.5,
                "predicted_kp": 4.1,
                "status": "Mocked forecast data"
            }
        finally:
            if 'conn' in locals():
                await conn.close()
    except Exception as e:
        print(f"Fallback triggered in get_forecast_predictions: {e}")
        return {
            "flare_probability": 0.82,
            "predicted_dst": -42.5,
            "predicted_kp": 4.1,
            "status": "Mocked forecast data (DB connection failed)"
        }

@tool
async def get_current_telemetry() -> dict:
    """
    Fetches the latest 5 minutes of SoLEXS, HEL1OS, MAG, and ASPEX flux/velocity data.
    Use this tool for real-time telemetry metrics and raw solar flux values.
    """
    try:
        conn = await get_db_connection()
        try:
            query = """
            SELECT 
                MAX(solexs_flux) as max_solexs,
                MAX(hel1os_flux) as max_hel1os,
                AVG(mag_b_total) as avg_mag,
                AVG(aspex_proton_density) as avg_proton,
                AVG(aspex_solar_wind_velocity) as avg_velocity
            FROM telemetry
            WHERE timestamp >= NOW() - INTERVAL '5 minutes'
            """
            row = await conn.fetchrow(query)
            if row:
                return {
                    "solexs_flux_max": row["max_solexs"] or 0.0,
                    "hel1os_flux_max": row["max_hel1os"] or 0.0,
                    "mag_b_total_avg": row["avg_mag"] or 0.0,
                    "aspex_proton_density_avg": row["avg_proton"] or 0.0,
                    "aspex_solar_wind_velocity_avg": row["avg_velocity"] or 0.0,
                    "status": "Telemetry retrieved successfully."
                }
            return {"status": "No telemetry data found for the last 5 minutes."}
        except (asyncpg.exceptions.UndefinedTableError, Exception) as e:
            print(f"Using mock telemetry data. DB Error: {e}")
            return {
                "solexs_flux_max": 5.2e-6,
                "hel1os_flux_max": 1.8e-7,
                "mag_b_total_avg": 5.4,
                "aspex_proton_density_avg": 11.2,
                "aspex_solar_wind_velocity_avg": 420.5,
                "status": "Mocked telemetry data"
            }
        finally:
            if 'conn' in locals():
                await conn.close()
    except Exception as e:
        print(f"Fallback triggered in get_current_telemetry: {e}")
        return {
            "solexs_flux_max": 5.2e-6,
            "hel1os_flux_max": 1.8e-7,
            "mag_b_total_avg": 5.4,
            "aspex_proton_density_avg": 11.2,
            "aspex_solar_wind_velocity_avg": 420.5,
            "status": "Mocked telemetry data (DB connection failed)"
        }
