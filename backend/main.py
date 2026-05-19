from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import asyncio
import httpx
import json
import os
import pathlib
import uuid
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from database import get_db, engine, Base, AsyncSessionLocal
from models import User, Conversation
from models import Message as DBMessage
from ollama_router import ollama_router
from shared_auth import get_or_create_user, init_shared_auth

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pathlib.Path("data").mkdir(exist_ok=True)
    init_shared_auth()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-Id", "X-LLM-Backend"],
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
    try:
        return get_or_create_user(request.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    models = await ollama_router.list_models()
    if not models["models"]:
        raise HTTPException(status_code=503, detail="No available Ollama backends or models")
    return models


@app.get("/routing/status")
async def routing_status():
    return await ollama_router.status()


# --- Chat ---

async def _persist_chat_result(full_response: str, is_new: bool, conversation_id: int, user_content: str):
    async with AsyncSessionLocal() as db:
        if full_response:
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
        elif is_new:
            conv = await db.get(Conversation, conversation_id)
            if conv:
                await db.delete(conv)
                await db.commit()


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        selected_backend = await ollama_router.choose_backend(request.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

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
        user_content = request.messages[-1].content
        try:
            async with ollama_router.track_request(selected_backend):
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"{selected_backend.url}/api/chat",
                        json={
                            "model": request.model,
                            "messages": [m.model_dump() for m in request.messages],
                            "stream": True,
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            data = json.loads(line)
                            if err := data.get("error"):
                                yield f"[Error: {err}]"
                                return
                            if content := data.get("message", {}).get("content"):
                                full_response += content
                                yield content
        except httpx.HTTPError as exc:
            yield f"[Error: {selected_backend.name} is unavailable: {exc}]"
        finally:
            await asyncio.shield(_persist_chat_result(full_response, is_new, conversation_id, user_content))

    return StreamingResponse(
        stream_and_save(),
        media_type="text/plain",
        headers={
            "X-Conversation-Id": str(conversation_id),
            "X-LLM-Backend": selected_backend.name,
        },
    )


# =============================================================================
# OpenAI-compatible API  (/v1/*)
# Other apps can point to this server instead of api.openai.com
# =============================================================================

class ApiChatMessage(BaseModel):
    role: str
    content: str

class ApiChatRequest(BaseModel):
    model: str
    messages: list[ApiChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


@app.get("/v1/models")
async def v1_list_models():
    ollama_models = (await ollama_router.list_models())["models"]
    if not ollama_models:
        raise HTTPException(status_code=503, detail="No available Ollama backends or models")

    return {
        "object": "list",
        "data": [
            {
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
            for m in ollama_models
        ],
    }


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ApiChatRequest):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    try:
        selected_backend = await ollama_router.choose_backend(request.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    options = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens

    ollama_payload = {
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "stream": request.stream,
    }
    if options:
        ollama_payload["options"] = options

    if request.stream:
        async def sse_stream():
            try:
                async with ollama_router.track_request(selected_backend):
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream("POST", f"{selected_backend.url}/api/chat", json=ollama_payload) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                data = json.loads(line)
                                if err := data.get("error"):
                                    error_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": f"[Error: {err}]"},
                                            "finish_reason": "stop",
                                        }],
                                    }
                                    yield f"data: {json.dumps(error_chunk)}\n\n"
                                    break
                                content = data.get("message", {}).get("content", "")
                                done = data.get("done", False)
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"role": "assistant", "content": content} if not done else {},
                                        "finish_reason": "stop" if done else None,
                                    }],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except httpx.HTTPError:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"X-LLM-Backend": selected_backend.name},
        )

    else:
        try:
            async with ollama_router.track_request(selected_backend):
                async with httpx.AsyncClient(timeout=None) as client:
                    response = await client.post(f"{selected_backend.url}/api/chat", json=ollama_payload)
                    response.raise_for_status()
                    data = response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"{selected_backend.name} is unavailable: {exc}") from exc
        if err := data.get("error"):
            raise HTTPException(status_code=502, detail=err)

        content = data.get("message", {}).get("content", "")
        usage = data.get("eval_count", 0)
        prompt_tokens = data.get("prompt_eval_count", 0)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": usage,
                "total_tokens": prompt_tokens + usage,
            },
        }
