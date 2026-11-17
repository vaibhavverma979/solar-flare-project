import asyncio
from src.streaming.consumer import telemetry_consumer
from src.streaming.broker import broker

async def main():
    await broker.connect()
    print("Consumer starting...")
    await telemetry_consumer()

if __name__ == '__main__':
    asyncio.run(main())
