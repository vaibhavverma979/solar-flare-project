"""
init_db.py — Creates all TimescaleDB tables and converts time-series tables
to hypertables. Run this ONCE before starting the application.

Usage:
    python init_db.py
"""
import asyncio
import sqlalchemy as sa
from src.database.connection import engine
from src.database.models import Base


# Tables that should be TimescaleDB hypertables (keyed on 'timestamp')
HYPERTABLES = [
    "solexs_telemetry",
    "hel1os_telemetry",
    "mag_telemetry",
    "aspex_telemetry",
    "space_weather_impact",
]


async def init_db():
    # 1. Create all ORM-defined tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ All database tables created (or already exist).")

    # 2. Convert time-series tables to TimescaleDB hypertables
    async with engine.begin() as conn:
        for table in HYPERTABLES:
            try:
                await conn.execute(
                    sa.text(
                        f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE);"
                    )
                )
                print(f"✅ Hypertable ready: {table}")
            except Exception as e:
                # TimescaleDB raises a NOTICE (not an error) when the hypertable already exists
                print(f"ℹ️  Hypertable notice for {table}: {e}")

    print("\n🚀 Database initialisation complete — ready for data ingestion.")


if __name__ == "__main__":
    asyncio.run(init_db())
