import asyncio
from src.database.connection import AsyncSessionLocal
from src.database.models import SolexsTelemetry
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(SolexsTelemetry))
        print('Rows in SoLEXS:', len(res.scalars().all()))

if __name__ == '__main__':
    asyncio.run(main())
