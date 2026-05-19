"""Password hashing and JWT helpers.

We use bcrypt for password hashing and JWT (HS256) for bearer tokens.
The signing key comes from the JWT_SECRET env var; if unset, a random key
is generated at startup (which invalidates tokens across restarts — fine
for dev, set the env var in production).
"""

import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Header

JWT_SECRET = os.getenv("JWT_SECRET") or secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
TOKEN_TTL_DAYS = 30


def hash_password(password: str) -> str:
    """Return a bcrypt hash for the given plaintext password."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time check that `password` matches `password_hash`."""
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_token(user_id: int, username: str) -> str:
    """Mint a JWT carrying the user's id + username, valid for TOKEN_TTL_DAYS."""
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(days=TOKEN_TTL_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode a JWT, raising HTTPException(401) on failure."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def current_user(authorization: str | None = Header(default=None)) -> dict:
    """FastAPI dependency: extract `{id, username}` from the Authorization header.

    Expects `Authorization: Bearer <jwt>`. Raises 401 if missing/invalid.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token(token)
    try:
        return {"id": int(payload["sub"]), "username": payload["username"]}
    except (KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Malformed token")


# Convenience: dependency that just returns the user id
async def current_user_id(user: dict = Depends(current_user)) -> int:
    return user["id"]
