import asyncio
from src.streaming.broker import broker

async def main():
    await broker.connect()
    print("Connected:", broker.redis is not None)
    if broker.redis:
        try:
            await broker.publish('test_stream', {'hello': 'world'})
            print('Published successfully')
        except Exception as e:
            print('Publish failed:', e)

if __name__ == '__main__':
    asyncio.run(main())
