import asyncio
from src.api.copilot_db_context import get_live_space_weather_context

async def main():
    res = await get_live_space_weather_context()
    print(res[0])

if __name__ == '__main__':
    asyncio.run(main())
