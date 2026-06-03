import os
import pathlib
import sys
import unittest
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("JWT_SECRET", "test-suite-secret")

import main  # noqa: E402
from database import Base  # noqa: E402
from models import Conversation, Message, User  # noqa: E402


class ConversationExportFormattingTests(unittest.TestCase):
    def test_export_payload_and_markdown_include_settings_and_messages(self):
        timestamp = datetime(2026, 5, 21, 12, 30, tzinfo=timezone.utc)
        conv = Conversation(
            id=42,
            user_id=1,
            title="Deploy Notes",
            model="llama3.2:3b",
            system_prompt="Be concise.",
            temperature=0.2,
            top_p=0.8,
            top_k=20,
            created_at=timestamp,
            updated_at=timestamp,
        )
        conv.messages = [
            Message(id=1, conversation_id=42, role="user", content="Summarize the rollout."),
            Message(id=2, conversation_id=42, role="assistant", content="All checks passed."),
        ]

        payload = main._conversation_export_payload(conv)
        markdown = main._conversation_export_markdown(payload)

        self.assertEqual(payload["title"], "Deploy Notes")
        self.assertEqual(payload["settings"]["temperature"], 0.2)
        self.assertEqual([message["role"] for message in payload["messages"]], ["user", "assistant"])
        self.assertIn("# Deploy Notes", markdown)
        self.assertIn("## System Prompt", markdown)
        self.assertIn("### User", markdown)
        self.assertIn("All checks passed.", markdown)
        self.assertEqual(
            main._conversation_export_filename(conv, "md"),
            "local-llm-20260521-deploy-notes.md",
        )


class ConversationExportRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.Session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_markdown_export_is_limited_to_conversation_owner(self):
        async with self.Session() as db:
            owner = User(username="owner", password_hash="x")
            other = User(username="other", password_hash="x")
            conv = Conversation(
                user=owner,
                title="Saved Answer",
                model="llama3.2:3b",
                messages=[
                    Message(role="user", content="How do I deploy?"),
                    Message(role="assistant", content="Run the checked deploy path."),
                ],
            )
            db.add_all([owner, other, conv])
            await db.commit()

            response = await main.export_conversation(conv.id, user_id=owner.id, db=db)

            self.assertIn("attachment;", response.headers["content-disposition"])
            self.assertIn("saved-answer.md", response.headers["content-disposition"])
            self.assertIn("Run the checked deploy path.", response.body.decode("utf-8"))

            with self.assertRaises(HTTPException) as raised:
                await main.export_conversation(conv.id, user_id=other.id, db=db)
            self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
