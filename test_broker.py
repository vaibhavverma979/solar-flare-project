import asyncio
from src.streaming.broker import broker

async def main():
    await broker.connect()
    print("Is redis connected?", broker.redis is not None)

if __name__ == '__main__':
    asyncio.run(main())
