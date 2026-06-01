import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import AgentJob, AgentJobEvent


AGENT_JOB_STATUSES = {
    "queued",
    "claimed",
    "running",
    "needs_input",
    "succeeded",
    "failed",
    "cancelled",
}
TERMINAL_AGENT_JOB_STATUSES = {"succeeded", "failed", "cancelled"}

ALLOWED_AGENT_JOB_TRANSITIONS = {
    "queued": {"claimed", "running", "succeeded", "failed", "cancelled"},
    "claimed": {"running", "succeeded", "failed", "cancelled"},
    "running": {"needs_input", "succeeded", "failed", "cancelled"},
    "needs_input": {"running", "failed", "cancelled"},
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
}


class AgentJobStateError(ValueError):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def agent_jobs_enabled() -> bool:
    value = os.getenv("AGENT_JOBS_ENABLED", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def require_agent_jobs_enabled():
    if not agent_jobs_enabled():
        raise HTTPException(status_code=404, detail="Agent jobs are disabled")


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def normalize_agent_job_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized not in AGENT_JOB_STATUSES:
        raise AgentJobStateError(f"Unknown agent job status: {status}")
    return normalized


def can_transition_agent_job(current_status: str, next_status: str) -> bool:
    current = normalize_agent_job_status(current_status)
    next_state = normalize_agent_job_status(next_status)
    return current == next_state or next_state in ALLOWED_AGENT_JOB_TRANSITIONS[current]


def serialize_agent_job_event(event: AgentJobEvent) -> dict:
    return {
        "id": event.id,
        "job_id": event.job_id,
        "status": event.status,
        "event_type": event.event_type,
        "message": event.message,
        "payload": _json_loads(event.payload_json, {}),
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }


def serialize_agent_job(job: AgentJob, *, include_events: bool = False) -> dict:
    payload = {
        "id": job.id,
        "user_id": job.user_id,
        "title": job.title,
        "prompt": job.prompt,
        "status": job.status,
        "status_detail": job.status_detail,
        "repository": {
            "owner": job.repository_owner,
            "name": job.repository_name,
        },
        "base_branch": job.base_branch,
        "target_branch": job.target_branch,
        "commit_sha": job.commit_sha,
        "issue_number": job.issue_number,
        "pull_request_number": job.pull_request_number,
        "github": {
            "installation_id": job.github_installation_id,
            "run_id": job.github_run_id,
            "check_run_id": job.github_check_run_id,
            "dispatch_event": job.dispatch_event,
        },
        "metadata": _json_loads(job.metadata_json, {}),
        "result": _json_loads(job.result_json, None),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
    if include_events:
        payload["events"] = [serialize_agent_job_event(event) for event in job.events]
    return payload


async def create_agent_job(
    db: AsyncSession,
    *,
    user_id: int,
    prompt: str,
    title: str | None = None,
    repository_owner: str | None = None,
    repository_name: str | None = None,
    base_branch: str | None = None,
    target_branch: str | None = None,
    issue_number: int | None = None,
    pull_request_number: int | None = None,
    github_installation_id: str | None = None,
    dispatch_event: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentJob:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise ValueError("Agent job prompt cannot be empty")

    clean_title = (title or clean_prompt.splitlines()[0]).strip()[:200]
    if not clean_title:
        clean_title = "Agent job"

    job = AgentJob(
        id=f"job_{uuid.uuid4().hex}",
        user_id=user_id,
        title=clean_title,
        prompt=clean_prompt,
        status="queued",
        status_detail="Queued",
        repository_owner=repository_owner,
        repository_name=repository_name,
        base_branch=base_branch,
        target_branch=target_branch,
        issue_number=issue_number,
        pull_request_number=pull_request_number,
        github_installation_id=github_installation_id,
        dispatch_event=dispatch_event,
        metadata_json=_json_dumps(metadata or {}) or "{}",
    )
    db.add(job)
    await db.flush()
    await append_agent_job_event(
        db,
        job,
        status="queued",
        event_type="state",
        message="Queued",
        commit=False,
    )
    await db.commit()
    return await get_agent_job(db, job.id, include_events=True) or job


async def get_agent_job(
    db: AsyncSession,
    job_id: str,
    *,
    include_events: bool = False,
) -> AgentJob | None:
    stmt = select(AgentJob).where(AgentJob.id == job_id)
    if include_events:
        stmt = stmt.options(selectinload(AgentJob.events)).execution_options(populate_existing=True)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_owned_agent_job(
    db: AsyncSession,
    *,
    user_id: int,
    job_id: str,
    include_events: bool = False,
) -> AgentJob | None:
    stmt = select(AgentJob).where(AgentJob.id == job_id, AgentJob.user_id == user_id)
    if include_events:
        stmt = stmt.options(selectinload(AgentJob.events)).execution_options(populate_existing=True)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_owned_agent_jobs(
    db: AsyncSession,
    *,
    user_id: int,
    status: str | None = None,
    limit: int = 50,
) -> list[AgentJob]:
    stmt = select(AgentJob).where(AgentJob.user_id == user_id)
    if status:
        stmt = stmt.where(AgentJob.status == normalize_agent_job_status(status))
    stmt = stmt.order_by(AgentJob.updated_at.desc()).limit(max(1, min(limit, 200)))
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def append_agent_job_event(
    db: AsyncSession,
    job: AgentJob,
    *,
    status: str | None = None,
    event_type: str = "state",
    message: str | None = None,
    payload: dict[str, Any] | None = None,
    commit: bool = True,
) -> AgentJobEvent:
    event = AgentJobEvent(
        job_id=job.id,
        status=normalize_agent_job_status(status or job.status),
        event_type=event_type,
        message=message,
        payload_json=_json_dumps(payload),
    )
    db.add(event)
    if commit:
        await db.commit()
    return event


async def transition_agent_job(
    db: AsyncSession,
    job: AgentJob,
    *,
    status: str,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> AgentJob:
    next_status = normalize_agent_job_status(status)
    if not can_transition_agent_job(job.status, next_status):
        raise AgentJobStateError(f"Cannot transition agent job from {job.status} to {next_status}")

    now = utcnow()
    if job.status != next_status:
        job.status = next_status
        if next_status in {"claimed", "running"} and not job.started_at:
            job.started_at = now
        if next_status in TERMINAL_AGENT_JOB_STATUSES and not job.completed_at:
            job.completed_at = now
    job.updated_at = now
    if message is not None:
        job.status_detail = message
    if result is not None:
        job.result_json = _json_dumps(result)

    await append_agent_job_event(
        db,
        job,
        status=next_status,
        event_type="state",
        message=message,
        payload=payload,
        commit=False,
    )
    await db.commit()
    return await get_agent_job(db, job.id, include_events=True) or job
