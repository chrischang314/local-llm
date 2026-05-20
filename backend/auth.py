"""Password hashing and JWT helpers.

We use bcrypt for password hashing and JWT (HS256) for bearer tokens.
The signing key comes from the JWT_SECRET env var when provided. Otherwise,
we create one once under the persistent app data directory so browser logins
survive backend restarts.
"""

import logging
import os
import pathlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Header

LOGGER = logging.getLogger(__name__)


def _load_jwt_secret() -> str:
    configured_secret = os.getenv("JWT_SECRET")
    if configured_secret:
        return configured_secret

    secret_path = pathlib.Path(
        os.getenv(
            "JWT_SECRET_FILE",
            str(pathlib.Path(os.getenv("LOCAL_LLM_DATA_DIR", "data")) / "jwt_secret"),
        )
    )

    try:
        persisted_secret = secret_path.read_text(encoding="utf-8").strip()
        if persisted_secret:
            return persisted_secret
    except FileNotFoundError:
        pass
    except OSError as exc:
        LOGGER.warning("Could not read JWT secret file %s: %s", secret_path, exc)

    generated_secret = secrets.token_urlsafe(48)
    try:
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with secret_path.open("x", encoding="utf-8") as handle:
                handle.write(generated_secret + "\n")
            try:
                secret_path.chmod(0o600)
            except OSError:
                pass
            return generated_secret
        except FileExistsError:
            persisted_secret = secret_path.read_text(encoding="utf-8").strip()
            if persisted_secret:
                return persisted_secret
            secret_path.write_text(generated_secret + "\n", encoding="utf-8")
            return generated_secret
    except OSError as exc:
        LOGGER.warning(
            "Could not persist JWT secret at %s; falling back to an ephemeral key: %s",
            secret_path,
            exc,
        )
        return generated_secret


JWT_SECRET = _load_jwt_secret()
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
