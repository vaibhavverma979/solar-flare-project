import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

async def main():
    try:
        import os
        db_url = os.getenv("TIMESCALEDB_URL", "postgresql+asyncpg://user:password@localhost:5432/defaultdb?ssl=require")
        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            print('Connected to TimescaleDB!')
    except Exception as e:
        print('DB Error:', e)

if __name__ == '__main__':
    asyncio.run(main())
