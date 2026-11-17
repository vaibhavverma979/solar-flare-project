from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from config import settings

# Create the async engine
engine = create_async_engine(
    settings.timescaledb_url,
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)

# Create an async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db_session():
    """Dependency injection function for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        yield session
