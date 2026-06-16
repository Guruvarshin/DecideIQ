from fastapi import APIRouter, Depends, HTTPException, Response
from pymongo.errors import DuplicateKeyError
from app.core.database import get_database
from app.core.security import hash_password, verify_password, create_access_token
from app.core.dependencies import get_current_user
from app.core.config import settings
from app.models.user import UserRegister, UserLogin
from datetime import datetime, timezone

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "access_token"
COOKIE_MAX_AGE = 7 * 24 * 60 * 60

# Cross-site (Vercel → Render) requires SameSite=None + Secure=True.
# Locally both run on localhost so Lax + Secure=False is fine.
_IS_PROD = "localhost" not in settings.frontend_url


def _set_auth_cookie(response: Response, token: str):
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="none" if _IS_PROD else "lax",
        secure=_IS_PROD,
        max_age=COOKIE_MAX_AGE,
    )


def _validate_register_input(body: UserRegister):
    if "@" not in body.email or "." not in body.email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")


@router.post("/register", status_code=201)
async def register(body: UserRegister, response: Response):
    _validate_register_input(body)
    db = get_database()
    try:
        result = await db.users.insert_one({
            "email": body.email.lower().strip(),
            "hashed_password": hash_password(body.password),
            "created_at": datetime.now(timezone.utc),
        })
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="An account with this email already exists")
    token = create_access_token(str(result.inserted_id))
    _set_auth_cookie(response, token)
    return {"id": str(result.inserted_id), "email": body.email.lower().strip()}


@router.post("/login")
async def login(body: UserLogin, response: Response):
    db = get_database()
    user = await db.users.find_one({"email": body.email.lower().strip()})
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token(str(user["_id"]))
    _set_auth_cookie(response, token)
    return {"id": str(user["_id"]), "email": user["email"]}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(
        COOKIE_NAME,
        samesite="none" if _IS_PROD else "lax",
        secure=_IS_PROD,
    )
    return {"message": "Logged out"}


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {
        "id": str(current_user["_id"]),
        "email": current_user["email"],
        "created_at": current_user["created_at"],
    }
