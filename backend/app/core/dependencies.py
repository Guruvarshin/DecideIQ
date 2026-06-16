from fastapi import Cookie, HTTPException
from bson import ObjectId
from app.core.security import decode_access_token
from app.core.database import get_database


async def get_current_user(access_token: str = Cookie(None)):
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = decode_access_token(access_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    db = get_database()
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
