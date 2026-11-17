import asyncio
from src.streaming.redis_client import redis_client

async def check_redis():
    await redis_client.connect()
    if not redis_client.redis:
        return
    for stream in ["aditya.telemetry.mag", "aditya.telemetry.aspex"]:
        try:
            res = await redis_client.redis.xrevrange(stream, count=2)
            print(f"Stream {stream}: {res}")
        except Exception as e:
            print(f"Error {stream}: {e}")
    await redis_client.disconnect()

if __name__ == "__main__":
    asyncio.run(check_redis())
