from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTH_DB = Path(os.getenv("SHARED_AUTH_DB", str(Path.home() / ".local-webapps" / "auth.db"))).expanduser().resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_username(username: str) -> tuple[str, str]:
    clean = " ".join(username.strip().split())
    if not clean:
        raise ValueError("Username cannot be empty")
    return clean[:80], clean.casefold()


def init_shared_auth() -> None:
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUTH_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                username_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def get_or_create_user(username: str) -> dict[str, Any]:
    clean, key = _normalize_username(username)
    now = _now()
    init_shared_auth()
    with sqlite3.connect(AUTH_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE username_key = ?", (key,)).fetchone()
        if row is None:
            cursor = conn.execute(
                "INSERT INTO users (username, username_key, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (clean, key, now, now),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        elif row["username"] != clean:
            conn.execute("UPDATE users SET username = ?, updated_at = ? WHERE id = ?", (clean, now, row["id"]))
            row = conn.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
    assert row is not None
    return {"id": row["id"], "username": row["username"], "created_at": row["created_at"], "updated_at": row["updated_at"]}
