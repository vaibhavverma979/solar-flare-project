import asyncio
from src.database.connection import AsyncSessionLocal
from src.database.models import SolexsTelemetry
from sqlalchemy import select, desc

async def main():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(SolexsTelemetry).order_by(desc(SolexsTelemetry.timestamp)).limit(1))
        obj = res.scalar_one_or_none()
        print('Latest Timestamp:', obj.timestamp if obj else None)

if __name__ == '__main__':
    asyncio.run(main())
