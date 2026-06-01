"""Shared state helpers for agentic code jobs."""

import os
import re
import hmac
import hashlib
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import AgentArtifact, AgentJob, AgentJobLog, AgentJobStep

TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "needs_review", "blocked"}
MAX_LOG_CHARS = 5 * 1024 * 1024
MAX_DIFF_CHARS = 2 * 1024 * 1024
TOKEN_PATTERNS = [
    re.compile(r"gh[psuor]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"x-access-token:[^@\\s]+", re.IGNORECASE),
]


def agent_jobs_enabled() -> bool:
    return os.getenv("AGENT_JOBS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def agent_executor_mode() -> str:
    return os.getenv("AGENT_EXECUTOR_MODE", "kubernetes").strip().lower() or "kubernetes"


def callback_token(job_id: str) -> str:
    secret = os.getenv("AGENT_SECRET_KEY", "")
    if not secret:
        return ""
    return hmac.new(secret.encode("utf-8"), job_id.encode("utf-8"), hashlib.sha256).hexdigest()


def new_job_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def redact_log(text: str) -> str:
    redacted = text or ""
    for pattern in TOKEN_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def serialize_job(job: AgentJob, *, include_children: bool = False) -> dict:
    payload = {
        "id": job.id,
        "status": job.status,
        "repo_full_name": job.repo_full_name,
        "base_branch": job.base_branch,
        "work_branch": job.work_branch,
        "model": job.model,
        "task": job.task,
        "test_command": job.test_command,
        "push_policy": job.push_policy,
        "commit_sha": job.commit_sha,
        "pr_url": job.pr_url,
        "error_summary": job.error_summary,
        "cancel_requested": bool(job.cancel_requested),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
    if include_children:
        payload["steps"] = [serialize_step(step) for step in job.steps]
        payload["artifacts"] = [serialize_artifact(artifact, include_content=False) for artifact in job.artifacts]
    return payload


def serialize_step(step: AgentJobStep) -> dict:
    return {
        "id": step.id,
        "position": step.position,
        "name": step.name,
        "status": step.status,
        "exit_code": step.exit_code,
        "started_at": step.started_at.isoformat() if step.started_at else None,
        "completed_at": step.completed_at.isoformat() if step.completed_at else None,
    }


def serialize_log(log: AgentJobLog) -> dict:
    return {
        "id": log.id,
        "level": log.level,
        "message": log.message,
        "created_at": log.created_at.isoformat() if log.created_at else None,
    }


def serialize_artifact(artifact: AgentArtifact, *, include_content: bool = True) -> dict:
    payload = {
        "id": artifact.id,
        "kind": artifact.kind,
        "name": artifact.name,
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
    }
    if include_content:
        payload["content"] = artifact.content
    return payload


async def append_job_log(
    db: AsyncSession,
    job_id: str,
    message: str,
    *,
    level: str = "info",
) -> AgentJobLog:
    total = await _job_log_size(db, job_id)
    sanitized = redact_log(message)
    if total + len(sanitized) > MAX_LOG_CHARS:
        sanitized = sanitized[: max(0, MAX_LOG_CHARS - total)]
        if not sanitized:
            sanitized = "[log limit reached]"
    log = AgentJobLog(job_id=job_id, level=level, message=sanitized)
    db.add(log)
    return log


async def _job_log_size(db: AsyncSession, job_id: str) -> int:
    result = await db.execute(select(AgentJobLog.message).where(AgentJobLog.job_id == job_id))
    return sum(len(row[0] or "") for row in result.fetchall())


def clamp_diff(diff: str) -> str:
    if len(diff or "") <= MAX_DIFF_CHARS:
        return diff or ""
    return (diff or "")[:MAX_DIFF_CHARS] + "\n[diff truncated]\n"
