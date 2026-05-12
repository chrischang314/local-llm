"""FastAPI backend for the local-LLM chat app.

Routes overview
---------------
Auth (no token required):
  POST /auth/register           — create a user with username + password
  POST /auth/login              — verify password, return JWT bearer token

Authenticated (Authorization: Bearer <jwt>):
  GET    /health                — backend + ollama health (no auth required)
  GET    /conversations         — list current user's conversations
  POST   /conversations         — create an empty conversation (with settings)
  PATCH  /conversations/{id}    — update title, system_prompt, model, params
  DELETE /conversations/{id}    — delete a conversation
  GET    /conversations/{id}/messages          — full message history
  DELETE /conversations/{id}/messages/from/{message_id}
                                — delete the given message and all later ones
  GET    /models                — list installed Ollama models
  POST   /models/pull           — pull a model (streams progress lines)
  DELETE /models/{name}         — delete an installed model
  POST   /chat                  — stream a chat completion + persist messages

The /chat endpoint accepts a `regenerate` flag: when True, the last
assistant message in the conversation is removed before generating, and
the request's last user message is NOT re-saved (it's expected to
already exist in the DB).
"""

import asyncio
import json
import os
import pathlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from auth import (
    create_token,
    current_user,
    current_user_id,
    hash_password,
    verify_password,
)
from database import AsyncSessionLocal, Base, engine, get_db, migrate_schema
from models import Conversation
from models import Message as DBMessage
from models import User

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pathlib.Path("data").mkdir(exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Ensure any columns added since the original release exist on disk.
    await migrate_schema()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-Id"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    conversation_id: int | None = None
    # When True, treat the last user message as already persisted (used for
    # the regenerate flow). The previous assistant message is deleted first.
    regenerate: bool = False
    # Settings applied when creating a brand-new conversation. Ignored for
    # existing conversations (use PATCH /conversations/{id} to change them).
    model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


class ConversationCreate(BaseModel):
    title: str | None = None
    system_prompt: str | None = ""
    model: str | None = None
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    top_k: int | None = 40


class ConversationUpdate(BaseModel):
    title: str | None = None
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


class ModelPullRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _validate_credentials(username: str, password: str) -> tuple[str, str]:
    """Strip + sanity-check a username/password pair; raise 400 on issues."""
    username = username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if len(username) > 64:
        raise HTTPException(status_code=400, detail="Username too long (max 64)")
    if not password or len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
    return username, password


@app.post("/auth/register")
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Create a new user. Returns a bearer token on success."""
    username, password = _validate_credentials(request.username, request.password)

    existing = (
        await db.execute(select(User).where(User.username == username))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(username=username, password_hash=hash_password(password))
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "token": create_token(user.id, user.username),
    }


@app.post("/auth/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Verify a username + password pair and return a bearer token."""
    username, password = _validate_credentials(request.username, request.password)
    user = (
        await db.execute(select(User).where(User.username == username))
    ).scalar_one_or_none()

    if not user or not verify_password(password, user.password_hash or ""):
        # Same message for both branches to avoid user enumeration.
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "id": user.id,
        "username": user.username,
        "token": create_token(user.id, user.username),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Report backend status and whether Ollama is reachable.

    Returns 200 even if Ollama is unreachable — clients should check the
    `ollama` field. This keeps Docker healthchecks green while the LLM
    container is still starting.
    """
    ollama_status = "down"
    model_count = 0
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            ollama_status = "ok"
            model_count = len(r.json().get("models", []))
    except httpx.HTTPError:
        pass
    return {"backend": "ok", "ollama": ollama_status, "model_count": model_count}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def _serialize_conversation(c: Conversation) -> dict:
    return {
        "id": c.id,
        "title": c.title,
        "system_prompt": c.system_prompt or "",
        "model": c.model,
        "temperature": c.temperature if c.temperature is not None else 0.7,
        "top_p": c.top_p if c.top_p is not None else 0.9,
        "top_k": c.top_k if c.top_k is not None else 40,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
    }


@app.get("/conversations")
async def list_conversations(
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return [_serialize_conversation(c) for c in result.scalars().all()]


@app.post("/conversations")
async def create_conversation(
    body: ConversationCreate,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    conv = Conversation(
        user_id=user_id,
        title=body.title or "New Chat",
        system_prompt=body.system_prompt or "",
        model=body.model,
        temperature=body.temperature if body.temperature is not None else 0.7,
        top_p=body.top_p if body.top_p is not None else 0.9,
        top_k=body.top_k if body.top_k is not None else 40,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return _serialize_conversation(conv)


@app.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: int,
    body: ConversationUpdate,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    conv = await _get_owned_conv(db, conversation_id, user_id)

    if body.title is not None:
        title = body.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        conv.title = title[:200]
    if body.system_prompt is not None:
        conv.system_prompt = body.system_prompt
    if body.model is not None:
        conv.model = body.model
    if body.temperature is not None:
        conv.temperature = float(body.temperature)
    if body.top_p is not None:
        conv.top_p = float(body.top_p)
    if body.top_k is not None:
        conv.top_k = int(body.top_k)
    conv.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(conv)
    return _serialize_conversation(conv)


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    conv = await _get_owned_conv(db, conversation_id, user_id)
    await db.delete(conv)
    await db.commit()
    return {"ok": True}


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: int,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .options(selectinload(Conversation.messages))
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return [{"id": m.id, "role": m.role, "content": m.content} for m in conv.messages]


@app.delete("/conversations/{conversation_id}/messages/from/{message_id}")
async def truncate_messages(
    conversation_id: int,
    message_id: int,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Delete the given message and all messages with a larger id.

    Used by the edit-message flow: truncate the conversation at the message
    being edited, then POST /chat with the new content.
    """
    conv = await _get_owned_conv(db, conversation_id, user_id)
    result = await db.execute(
        select(DBMessage)
        .where(
            DBMessage.conversation_id == conv.id,
            DBMessage.id >= message_id,
        )
        .order_by(DBMessage.id)
    )
    for m in result.scalars().all():
        await db.delete(m)
    await db.commit()
    return {"ok": True}


async def _get_owned_conv(db: AsyncSession, conversation_id: int, user_id: int) -> Conversation:
    """Fetch a conversation, raising 404 if it doesn't exist or isn't owned."""
    conv = (
        await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@app.get("/models")
async def list_models(_: dict = Depends(current_user)):
    """List installed Ollama models. Returns Ollama's /api/tags payload."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        raise HTTPException(
            status_code=503, detail="Cannot connect to Ollama. Is it running?"
        )


@app.post("/models/pull")
async def pull_model(body: ModelPullRequest, _: dict = Depends(current_user)):
    """Pull a model, streaming Ollama's progress lines back to the client.

    Each line of the response body is a JSON object from Ollama's pull API
    (status, completed, total, digest). The frontend can use these to show
    a download progress bar.
    """
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Model name required")

    async def stream_pull():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/pull",
                    json={"name": name, "stream": True},
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            yield line + "\n"
        except httpx.ConnectError:
            yield json.dumps({"error": "Cannot connect to Ollama"}) + "\n"
        except httpx.HTTPError as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(stream_pull(), media_type="application/x-ndjson")


@app.delete("/models/{name:path}")
async def delete_model(name: str, _: dict = Depends(current_user)):
    """Delete an installed Ollama model."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(
                "DELETE",
                f"{OLLAMA_URL}/api/delete",
                json={"name": name},
            )
            response.raise_for_status()
            return {"ok": True}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Model {name} not found")
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")
    except httpx.HTTPError:
        raise HTTPException(
            status_code=503, detail="Cannot connect to Ollama. Is it running?"
        )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def _persist_chat_result(
    full_response: str,
    is_new: bool,
    conversation_id: int,
    user_content: str | None,
):
    """Save the result of a chat stream (or clean up if it produced nothing).

    `user_content` is None for regenerate flows (the user message is already
    persisted; we only save the new assistant reply).
    """
    async with AsyncSessionLocal() as db:
        conv = await db.get(Conversation, conversation_id)
        if not conv:
            return

        if full_response:
            if is_new and user_content is not None:
                title = user_content[:50] + ("..." if len(user_content) > 50 else "")
                conv.title = title
            if user_content is not None:
                db.add(
                    DBMessage(
                        conversation_id=conversation_id,
                        role="user",
                        content=user_content,
                    )
                )
            db.add(
                DBMessage(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_response,
                )
            )
            conv.updated_at = datetime.now(timezone.utc)
            await db.commit()
        elif is_new:
            # Nothing came back from the model — don't leave an orphan conv.
            await db.delete(conv)
            await db.commit()


async def _delete_last_assistant_message(conversation_id: int):
    """Remove the most recent assistant message from a conversation."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(DBMessage)
            .where(
                DBMessage.conversation_id == conversation_id,
                DBMessage.role == "assistant",
            )
            .order_by(DBMessage.id.desc())
            .limit(1)
        )
        msg = result.scalar_one_or_none()
        if msg:
            await db.delete(msg)
            await db.commit()


@app.post("/chat")
async def chat(
    request: ChatRequest,
    user_id: int = Depends(current_user_id),
):
    """Stream a chat completion and persist the resulting message pair.

    On a new conversation, the request's `model`, `system_prompt`, and
    sampling params are baked into the conversation row. On an existing
    conversation, those settings are read from the DB (so changes via
    PATCH apply to subsequent messages).
    """
    async with AsyncSessionLocal() as db:
        is_new = request.conversation_id is None
        if not is_new:
            conv = await _get_owned_conv(db, request.conversation_id, user_id)
        else:
            # Persist the settings provided at creation time. Title is
            # filled in once we have a user message.
            conv = Conversation(
                user_id=user_id,
                model=request.model,
                system_prompt=request.system_prompt or "",
                temperature=request.temperature if request.temperature is not None else 0.7,
                top_p=request.top_p if request.top_p is not None else 0.9,
                top_k=request.top_k if request.top_k is not None else 40,
            )
            db.add(conv)
            await db.commit()
            await db.refresh(conv)

        # Snapshot what we need before the session closes.
        conversation_id = conv.id
        model = conv.model or request.model or "llama3.2"
        system_prompt = conv.system_prompt or ""
        options = {
            "temperature": conv.temperature,
            "top_p": conv.top_p,
            "top_k": conv.top_k,
        }

    # For regenerate, drop the existing last-assistant message up front so
    # that even a client disconnect ends up in a consistent state.
    if request.regenerate and not is_new:
        await _delete_last_assistant_message(conversation_id)

    # Build the messages sent to Ollama: system prompt first (if any),
    # then the conversation history as supplied by the client.
    ollama_messages: list[dict] = []
    if system_prompt:
        ollama_messages.append({"role": "system", "content": system_prompt})
    ollama_messages.extend(m.model_dump() for m in request.messages)

    user_content = request.messages[-1].content if request.messages else ""
    # For regenerate, the user message is already in the DB.
    user_content_to_save = None if request.regenerate else user_content

    async def stream_and_save():
        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": model,
                        "messages": ollama_messages,
                        "stream": True,
                        "options": options,
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        if err := data.get("error"):
                            yield f"[Error: {err}]"
                        elif content := data.get("message", {}).get("content"):
                            full_response += content
                            yield content
        except httpx.ConnectError:
            yield "[Error: Cannot connect to Ollama. Is it running?]"
        except httpx.HTTPError as e:
            yield f"[Error: {e}]"
        finally:
            # Shield persistence so a client disconnect mid-stream doesn't
            # tear down the DB session before we finish writing.
            try:
                await asyncio.shield(
                    _persist_chat_result(
                        full_response,
                        is_new,
                        conversation_id,
                        user_content_to_save,
                    )
                )
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        stream_and_save(),
        media_type="text/plain",
        headers={"X-Conversation-Id": str(conversation_id)},
    )
