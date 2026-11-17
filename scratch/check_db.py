import asyncio
from sqlalchemy import text
from src.database.connection import AsyncSessionLocal

async def check_db():
    async with AsyncSessionLocal() as session:
        tables = [
            "solexs_telemetry",
            "hel1os_telemetry",
            "mag_telemetry",
            "aspex_telemetry",
            "space_weather_impact",
            "flare_events",
            "geomagnetic_forecast"
        ]
        for t in tables:
            try:
                result = await session.execute(text(f"SELECT COUNT(*) FROM {t}"))
                count = result.scalar()
                print(f"Table {t}: {count} rows")
            except Exception as e:
                print(f"Table {t} error: {e}")
                await session.rollback()

if __name__ == "__main__":
    asyncio.run(check_db())
