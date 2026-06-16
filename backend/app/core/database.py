from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client: AsyncIOMotorClient = None


async def connect_to_mongo():
    global client
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_db_name]
    await db.users.create_index("email", unique=True)


async def close_mongo():
    global client
    if client:
        client.close()


def get_database():
    return client[settings.mongodb_db_name]
