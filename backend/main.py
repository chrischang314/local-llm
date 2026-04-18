from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import httpx
import json
import os
import pathlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from database import get_db, engine, Base, AsyncSessionLocal
from models import User, Conversation
from models import Message as DBMessage

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pathlib.Path("data").mkdir(exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-Id"],
)


# --- Pydantic schemas ---

class LoginRequest(BaseModel):
    username: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "llama3.2"
    messages: list[ChatMessage]
    user_id: int
    conversation_id: int | None = None


# --- Auth ---

@app.post("/auth/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    username = request.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")

    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    if not user:
        user = User(username=username)
        db.add(user)
        await db.commit()
        await db.refresh(user)

    return {"id": user.id, "username": user.username}


# --- Conversations ---

@app.get("/conversations")
async def list_conversations(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()
    return [{"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()} for c in conversations]


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conv)
    await db.commit()
    return {"ok": True}


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: int, user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .options(selectinload(Conversation.messages))
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return [{"role": m.role, "content": m.content} for m in conv.messages]


# --- Models ---

@app.get("/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is it running?")


# --- Chat ---

@app.post("/chat")
async def chat(request: ChatRequest):
    async with AsyncSessionLocal() as db:
        is_new = request.conversation_id is None
        if not is_new:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == request.conversation_id,
                    Conversation.user_id == request.user_id,
                )
            )
            conv = result.scalar_one_or_none()
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_id = conv.id
        else:
            conv = Conversation(user_id=request.user_id)
            db.add(conv)
            await db.commit()
            await db.refresh(conv)
            conversation_id = conv.id

    async def stream_and_save():
        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": request.model,
                        "messages": [m.model_dump() for m in request.messages],
                        "stream": True,
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if content := data.get("message", {}).get("content"):
                                full_response += content
                                yield content
        except httpx.ConnectError:
            yield "[Error: Cannot connect to Ollama. Is it running?]"
        finally:
            if full_response:
                user_content = request.messages[-1].content
                async with AsyncSessionLocal() as db:
                    if is_new:
                        title = user_content[:50] + ("..." if len(user_content) > 50 else "")
                        conv = await db.get(Conversation, conversation_id)
                        if conv:
                            conv.title = title
                    db.add(DBMessage(conversation_id=conversation_id, role="user", content=user_content))
                    db.add(DBMessage(conversation_id=conversation_id, role="assistant", content=full_response))
                    conv = await db.get(Conversation, conversation_id)
                    if conv:
                        conv.updated_at = datetime.now(timezone.utc)
                    await db.commit()

    return StreamingResponse(
        stream_and_save(),
        media_type="text/plain",
        headers={"X-Conversation-Id": str(conversation_id)},
    )
