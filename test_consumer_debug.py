import asyncio
import traceback
from src.streaming.consumer import telemetry_consumer, broker
from config import settings

async def main():
    try:
        await broker.connect()
        print("Connected to Redis. Starting consumer...")
        await telemetry_consumer()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
