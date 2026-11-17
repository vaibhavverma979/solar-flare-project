import asyncio
import redis.asyncio

async def main():
    try:
        url = 'rediss://default:gQAAAAAAAaGHAAIgcDFjMzRjYzI1NWU5NjU0MTEyYWJmMjlhOTc3NmZjM2ExYw@mighty-gannet-106887.upstash.io:6379'
        print(f"Connecting to {url} ...")
        r = redis.asyncio.from_url(url, decode_responses=True)
        await r.ping()
        print('Success!')
        await r.close()
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    asyncio.run(main())
