import asyncio
from src.database.connection import AsyncSessionLocal
from src.database.models import SolexsTelemetry
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(SolexsTelemetry).limit(1))
        obj = res.scalar_one_or_none()
        print('Timestamp:', obj.timestamp if obj else None)

if __name__ == '__main__':
    asyncio.run(main())
