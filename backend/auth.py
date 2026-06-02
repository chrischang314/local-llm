"""Shared server-side auth for Local LLM.

The browser stores only an HttpOnly session cookie. Users and sessions live in
the shared SQLite database pointed at by ``SHARED_AUTH_DB``; the app-local
``users`` table is still used for conversation and GitHub job ownership.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import User

AUTH_DB_ENV = "SHARED_AUTH_DB"
DEFAULT_AUTH_DB = Path.home() / ".local-webapps" / "auth.db"
PASSWORD_ITERATIONS = 200_000
PASSWORD_MIN_LENGTH = 8
SESSION_COOKIE_NAME = "projects_lan_session"
SESSION_TTL_DAYS = 30
COOKIE_DOMAIN_ENV = "AUTH_COOKIE_DOMAIN"
PROJECTS_LAN_COOKIE_DOMAIN_ENV = "PROJECTS_LAN_COOKIE_DOMAIN"


class UserAlreadyExistsError(ValueError):
    """Raised when registering a username that already has a password."""


class InvalidCredentialsError(ValueError):
    """Raised when username/password authentication fails."""


def auth_db_path() -> Path:
    return Path(os.getenv(AUTH_DB_ENV, str(DEFAULT_AUTH_DB))).expanduser().resolve()


def cookie_domain() -> str | None:
    configured = os.getenv(COOKIE_DOMAIN_ENV) or os.getenv(PROJECTS_LAN_COOKIE_DOMAIN_ENV)
    return configured.strip() if configured and configured.strip() else None


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(user["id"]),
        "username": user["username"],
        "created_at": user.get("created_at"),
        "updated_at": user.get("updated_at"),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalize_username(username: str) -> tuple[str, str]:
    clean = " ".join(username.strip().split())
    if not clean:
        raise ValueError("Username cannot be empty")
    if len(clean) > 80:
        raise ValueError("Username too long (max 80)")
    return clean, clean.casefold()


def _normalize_password(password: str) -> str:
    if not password:
        raise ValueError("Password cannot be empty")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
    if len(password) > 1024:
        raise ValueError("Password is too long")
    return password


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return base64.b64encode(digest).decode("ascii")


def _new_salt() -> str:
    return base64.b64encode(secrets.token_bytes(16)).decode("ascii")


def _public_user_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _ensure_columns(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "username_key" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN username_key TEXT")
        conn.execute("UPDATE users SET username_key = lower(trim(username)) WHERE username_key IS NULL")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_username_key_idx ON users(username_key)")
    if "password_hash" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "password_salt" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN password_salt TEXT")
    if "created_at" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
        conn.execute("UPDATE users SET created_at = ? WHERE created_at IS NULL", (_now(),))
    if "updated_at" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN updated_at TEXT")
        conn.execute("UPDATE users SET updated_at = ? WHERE updated_at IS NULL", (_now(),))


def init_shared_auth() -> None:
    db_path = auth_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                username_key TEXT NOT NULL UNIQUE,
                password_hash TEXT,
                password_salt TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_columns(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )


def register_user(username: str, password: str) -> dict[str, Any]:
    clean, key = _normalize_username(username)
    password = _normalize_password(password)
    now = _now()
    init_shared_auth()
    salt = _new_salt()
    password_hash = _hash_password(password, base64.b64decode(salt.encode("ascii")))
    with sqlite3.connect(auth_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE username_key = ?", (key,)).fetchone()
        if row is None:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    username, username_key, password_hash, password_salt, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (clean, key, password_hash, salt, now, now),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        elif row["password_hash"] is None or row["password_salt"] is None:
            conn.execute(
                """
                UPDATE users
                SET username = ?, password_hash = ?, password_salt = ?, updated_at = ?
                WHERE id = ?
                """,
                (clean, password_hash, salt, now, row["id"]),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
        else:
            raise UserAlreadyExistsError("That username is already registered")
    assert row is not None
    return _public_user_from_row(row)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    _, key = _normalize_username(username)
    init_shared_auth()
    with sqlite3.connect(auth_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE username_key = ?", (key,)).fetchone()
    return _public_user_from_row(row) if row is not None else None


def migrate_legacy_user(username: str) -> dict[str, Any]:
    """Seed a shared auth row after verifying an existing local-only user.

    The legacy local app allowed shorter passwords than the shared SSO
    contract. We should still let those users sign in once, but we do not copy
    the short password into the shared auth database.
    """

    existing = get_user_by_username(username)
    if existing is not None:
        return existing
    generated_password = secrets.token_urlsafe(48)
    return register_user(username, generated_password)


def authenticate_user(username: str, password: str) -> dict[str, Any]:
    _, key = _normalize_username(username)
    password = _normalize_password(password)
    init_shared_auth()
    with sqlite3.connect(auth_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE username_key = ?", (key,)).fetchone()
    if row is None or row["password_hash"] is None or row["password_salt"] is None:
        raise InvalidCredentialsError("Invalid username or password")
    salt = base64.b64decode(str(row["password_salt"]).encode("ascii"))
    candidate = _hash_password(password, salt)
    if not hmac.compare_digest(candidate, str(row["password_hash"])):
        raise InvalidCredentialsError("Invalid username or password")
    return _public_user_from_row(row)


def get_user(user_id: int) -> dict[str, Any] | None:
    init_shared_auth()
    with sqlite3.connect(auth_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _public_user_from_row(row) if row is not None else None


def create_session(user_id: int, *, ttl_days: int = SESSION_TTL_DAYS) -> str:
    if get_user(user_id) is None:
        raise InvalidCredentialsError("User not found")
    init_shared_auth()
    token = secrets.token_urlsafe(32)
    created = datetime.now(UTC)
    expires = created + timedelta(days=ttl_days)
    with sqlite3.connect(auth_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO auth_sessions (user_id, token_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, _token_hash(token), created.isoformat(), expires.isoformat()),
        )
    return token


def get_user_by_session_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    init_shared_auth()
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(auth_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT users.*
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ?
              AND auth_sessions.revoked_at IS NULL
              AND auth_sessions.expires_at > ?
            """,
            (_token_hash(token), now),
        ).fetchone()
    return _public_user_from_row(row) if row is not None else None


def revoke_session(token: str | None) -> None:
    if not token:
        return
    init_shared_auth()
    with sqlite3.connect(auth_db_path()) as conn:
        conn.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ?",
            (_now(), _token_hash(token)),
        )


def verify_legacy_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (TypeError, ValueError):
        return False


def set_session_cookie(response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=60 * 60 * 24 * SESSION_TTL_DAYS,
        httponly=True,
        samesite="lax",
        path="/",
        domain=cookie_domain(),
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        httponly=True,
        samesite="lax",
        path="/",
        domain=cookie_domain(),
    )


async def ensure_local_user(shared_user: dict[str, Any], db: AsyncSession) -> User:
    """Ensure a local ownership row exists for the shared authenticated user."""
    shared_id = int(shared_user["id"])
    username = str(shared_user["username"])
    local = (
        await db.execute(select(User).where(User.username == username))
    ).scalar_one_or_none()

    if local is None:
        existing_id = await db.get(User, shared_id)
        local = (
            User(username=username, password_hash=None)
            if existing_id is not None
            else User(id=shared_id, username=username, password_hash=None)
        )
        db.add(local)

    await db.commit()
    await db.refresh(local)
    return local


async def current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    shared_user = get_user_by_session_token(request.cookies.get(SESSION_COOKIE_NAME))
    if shared_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    local_user = await ensure_local_user(shared_user, db)
    return {**public_user(shared_user), "local_id": int(local_user.id)}


async def current_user_id(user: dict[str, Any] = Depends(current_user)) -> int:
    return int(user["local_id"])
