import os
import pathlib
import sys
import unittest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("JWT_SECRET", "test-suite-secret")

import main  # noqa: E402
from database import Base  # noqa: E402
from models import Conversation, Message, User  # noqa: E402


class ChatMetadataTests(unittest.TestCase):
    def test_serialize_message_includes_assistant_route_metadata(self):
        message = Message(
            id=42,
            role="assistant",
            content="Done.",
            model="llama3.2:3b",
            backend_name="mac-mini",
            model_status="resident",
        )

        self.assertEqual(
            main._serialize_message(message),
            {
                "id": 42,
                "role": "assistant",
                "content": "Done.",
                "model": "llama3.2:3b",
                "backend_name": "mac-mini",
                "model_status": "resident",
            },
        )

    def test_serialize_message_keeps_user_messages_route_empty(self):
        message = Message(id=7, role="user", content="Hello")

        self.assertEqual(
            main._serialize_message(message),
            {
                "id": 7,
                "role": "user",
                "content": "Hello",
                "model": None,
                "backend_name": None,
                "model_status": None,
            },
        )


class ChatMetadataPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_persist_chat_result_saves_assistant_route_metadata(self):
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with session_factory() as db:
            db.add(User(id=1, username="test", password_hash="hash"))
            db.add(Conversation(id=1, user_id=1, title="New Chat"))
            await db.commit()

        original_session = main.AsyncSessionLocal
        main.AsyncSessionLocal = session_factory
        try:
            await main._persist_chat_result(
                "Assistant reply",
                True,
                1,
                "User prompt",
                {
                    "model": "llama3.2:3b",
                    "backend_name": "mac-mini",
                    "model_status": "resident",
                },
            )
        finally:
            main.AsyncSessionLocal = original_session

        async with session_factory() as db:
            messages = (
                await db.execute(select(Message).order_by(Message.id))
            ).scalars().all()

        await engine.dispose()

        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        assistant = messages[1]
        self.assertEqual(assistant.content, "Assistant reply")
        self.assertEqual(assistant.model, "llama3.2:3b")
        self.assertEqual(assistant.backend_name, "mac-mini")
        self.assertEqual(assistant.model_status, "resident")


if __name__ == "__main__":
    unittest.main()
