from sqlalchemy import text
from src.database.connection import AsyncSessionLocal
import pandas as pd

class CatalogService:
    """
    Enterprise catalog service that replaces the legacy in-memory caching system.
    Queries the TimescaleDB hypertables and continuous aggregates directly.
    """

    async def get_historical_lightcurve(self, start_time: str, end_time: str, bucket_interval='1 minute'):
        """
        Fetches the aggregated historical light-curve from TimescaleDB using native
        time_bucket downsampling for optimal performance.
        """
        async with AsyncSessionLocal() as session:
            # We use native raw SQL for the hypertable time_bucket function
            query = text(f"""
                SELECT time_bucket(:interval, timestamp) AS bucket,
                       AVG(soft_xray_flux) as solexs_flux
                FROM solexs_telemetry
                WHERE timestamp >= :start_time AND timestamp <= :end_time
                GROUP BY bucket
                ORDER BY bucket ASC
            """)
            
            result = await session.execute(query, {
                "interval": bucket_interval,
                "start_time": start_time,
                "end_time": end_time
            })
            
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=["timestamp", "flux"])
            return df

    async def get_flare_events(self):
        """
        Fetches the master flare catalog directly from the database instead of
        calculating it in-memory via Pandas.
        """
        async with AsyncSessionLocal() as session:
            query = text("""
                SELECT event_id, peak_timestamp as peak_time, end_timestamp as end_time, 
                       flare_class as class, confidence_score as confidence, neupert_flag as neupert_consistent
                FROM flare_events
                ORDER BY peak_timestamp DESC
                LIMIT 100
            """)
            
            result = await session.execute(query)
            rows = result.fetchall()
            
            # Map database columns to expected JSON keys
            records = [dict(row._mapping) for row in rows]
            
            # Since the database might be empty in this fallback mode, we return an empty list
            # The API cache fallback will provide demo data if this fails or is empty.
            return records

catalog_service = CatalogService()
