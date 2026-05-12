"""Async SQLAlchemy engine + session setup, plus an in-place schema migration.

We don't pull in Alembic for what's still a small project — instead, the
`migrate_schema` helper adds any new columns to existing tables using
SQLite's `ALTER TABLE ADD COLUMN`. This is called from main.py's lifespan
after `Base.metadata.create_all` so that:

  * Fresh installs get the full schema via create_all.
  * Existing installs (e.g. early dev databases) gain the new columns
    without losing data.
"""

import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/chat.db")

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# Columns that were added to existing tables after the initial release.
# (table_name, column_name, column_def) — column_def is the post-`ADD COLUMN`
# SQL fragment, e.g. "TEXT DEFAULT ''".
_NEW_COLUMNS: list[tuple[str, str, str]] = [
    ("users", "password_hash", "TEXT"),
    ("conversations", "system_prompt", "TEXT DEFAULT ''"),
    ("conversations", "model", "TEXT"),
    ("conversations", "temperature", "REAL DEFAULT 0.7"),
    ("conversations", "top_p", "REAL DEFAULT 0.9"),
    ("conversations", "top_k", "INTEGER DEFAULT 40"),
]


async def migrate_schema():
    """Add any missing columns to existing tables. Idempotent."""
    async with engine.begin() as conn:
        for table, column, definition in _NEW_COLUMNS:
            result = await conn.execute(text(f"PRAGMA table_info({table})"))
            existing = {row[1] for row in result.fetchall()}
            if column not in existing:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                )
