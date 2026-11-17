import asyncio
from src.streaming.broker import broker

async def main():
    await broker.connect()
    print("Stream Length Solexs:", await broker.redis.xlen('aditya.telemetry.solexs'))

if __name__ == '__main__':
    asyncio.run(main())
