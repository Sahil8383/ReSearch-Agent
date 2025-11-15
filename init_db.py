"""Script to initialize the database"""

import asyncio
from api.database import init_db, close_db


async def main():
    """Initialize database tables"""
    print("Initializing database...")
    try:
        await init_db()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())

