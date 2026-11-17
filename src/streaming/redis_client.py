import redis.asyncio as redis
from config import settings

class RedisClient:
    def __init__(self):
        self.redis = None

    async def connect(self):
        try:
            self.redis = redis.from_url(settings.redis_url, decode_responses=True)
            await self.redis.ping()
            print("Connected to Redis broker.")
        except Exception as e:
            print(f"Warning: Redis connection failed: {e}. Running in degraded mode.")
            self.redis = None

    async def disconnect(self):
        if self.redis:
            await self.redis.close()

redis_client = RedisClient()

async def get_redis():
    """Dependency for FastAPI"""
    return redis_client.redis
