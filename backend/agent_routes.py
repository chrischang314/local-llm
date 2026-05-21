"""Agentic code job routes."""

import asyncio
import hmac
import json
import os
import re

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agent_executor import kubernetes_agent_executor
from agent_services import (
    TERMINAL_STATUSES,
    agent_executor_mode,
    agent_jobs_enabled,
    append_job_log,
    callback_token,
    clamp_diff,
    new_job_id,
    serialize_job,
    serialize_log,
    utcnow,
)
from auth import current_user_id
from database import AsyncSessionLocal, get_db
from github_client import github_app_client
from github_routes import allowed_installation_ids, current_installation
from models import AgentArtifact, AgentJob, AgentJobLog, AgentJobStep

router = APIRouter(prefix="/agent", tags=["agent"])

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


class AgentJobCreateRequest(BaseModel):
    repo_full_name: str = Field(min_length=3, max_length=200)
    base_branch: str = Field(min_length=1, max_length=200)
    task: str = Field(min_length=3, max_length=8000)
    model: str = Field(min_length=1, max_length=200)
    test_command: str | None = Field(default=None, max_length=1000)


class AgentInternalLogRequest(BaseModel):
    level: str = "info"
    message: str


class AgentInternalStepRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    status: str = Field(min_length=1, max_length=40)
    exit_code: int | None = None


class AgentInternalCompleteRequest(BaseModel):
    status: str
    diff: str | None = None
    commit_sha: str | None = None
    pr_url: str | None = None
    error_summary: str | None = None


class AgentInternalTokenResponse(BaseModel):
    token: str


def _validate_job_request(request: AgentJobCreateRequest) -> None:
    if not REPO_RE.match(request.repo_full_name):
        raise HTTPException(status_code=400, detail="repo_full_name must be owner/repo")
    if not BRANCH_RE.match(request.base_branch):
        raise HTTPException(status_code=400, detail="base_branch contains unsupported characters")
    if request.test_command and len(request.test_command.splitlines()) > 4:
        raise HTTPException(status_code=400, detail="test_command must be a short command")


def _initial_steps(job_id: str) -> list[AgentJobStep]:
    names = ["clone", "plan", "edit", "test", "push"]
    return [
        AgentJobStep(job_id=job_id, position=index + 1, name=name)
        for index, name in enumerate(names)
    ]


@router.get("/status")
async def agent_status(_: int = Depends(current_user_id)):
    missing = _missing_agent_config()
    return {
        "enabled": agent_jobs_enabled() and not missing,
        "requested_enabled": agent_jobs_enabled(),
        "executor_mode": agent_executor_mode(),
        "push_policy": "direct-main-after-tests",
        "missing": missing,
    }


def _missing_agent_config() -> list[str]:
    missing = []
    if not agent_jobs_enabled():
        missing.append("AGENT_JOBS_ENABLED")
    if not os.getenv("AGENT_SECRET_KEY"):
        missing.append("AGENT_SECRET_KEY")
    if agent_jobs_enabled() and not allowed_installation_ids():
        missing.append("GITHUB_ALLOWED_INSTALLATION_IDS")
    return missing + github_app_client.config().missing


@router.post("/jobs")
async def create_job(
    request: AgentJobCreateRequest,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    if not agent_jobs_enabled():
        raise HTTPException(status_code=503, detail="Agent jobs are disabled until sandbox canaries pass")
    _validate_job_request(request)

    installation = await current_installation(db, user_id)
    if not installation:
        raise HTTPException(status_code=404, detail="GitHub App is not connected")
    allowed_ids = allowed_installation_ids()
    if not allowed_ids:
        raise HTTPException(
            status_code=503,
            detail="GITHUB_ALLOWED_INSTALLATION_IDS is required before live agent jobs can run",
        )
    if installation.installation_id not in allowed_ids:
        raise HTTPException(status_code=403, detail="Connected GitHub installation is not allowed for agent jobs")
    await _validate_repo_branch_access(installation.installation_id, request.repo_full_name, request.base_branch)

    job_id = new_job_id()
    job = AgentJob(
        id=job_id,
        user_id=user_id,
        status="queued",
        repo_full_name=request.repo_full_name,
        base_branch=request.base_branch,
        work_branch=f"agent/{job_id[:12]}",
        model=request.model,
        task=request.task,
        test_command=(request.test_command or "").strip() or None,
        push_policy="direct-main-after-tests",
    )
    job.steps = _initial_steps(job_id)
    job.artifacts = []
    db.add(job)
    await append_job_log(db, job_id, "Queued agent job.")
    await db.commit()

    background_tasks.add_task(launch_agent_job, job_id)
    return serialize_job(job, include_children=True)


@router.get("/jobs")
async def list_jobs(
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentJob)
        .where(AgentJob.user_id == user_id)
        .order_by(AgentJob.created_at.desc())
        .limit(50)
    )
    return {"jobs": [serialize_job(job) for job in result.scalars().all()]}


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    job = await _owned_job(db, job_id, user_id, include_children=True)
    return serialize_job(job, include_children=True)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    job = await _owned_job(db, job_id, user_id)
    job.cancel_requested = True
    if job.status not in TERMINAL_STATUSES:
        await kubernetes_agent_executor.delete_job(job.id)
        await kubernetes_agent_executor.cleanup_secret(job.id)
        job.status = "canceled"
        job.completed_at = utcnow()
    job.updated_at = utcnow()
    await append_job_log(db, job.id, "Cancellation requested.", level="warning")
    await db.commit()
    return {"ok": True, "job": serialize_job(job)}


@router.get("/jobs/{job_id}/diff")
async def job_diff(
    job_id: str,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    await _owned_job(db, job_id, user_id)
    result = await db.execute(
        select(AgentArtifact)
        .where(AgentArtifact.job_id == job_id, AgentArtifact.kind == "diff")
        .order_by(AgentArtifact.created_at.desc())
    )
    artifact = result.scalars().first()
    return {"diff": artifact.content if artifact else ""}


@router.get("/jobs/{job_id}/events")
async def job_events(
    job_id: str,
    user_id: int = Depends(current_user_id),
):
    async def stream():
        last_log_id = 0
        while True:
            async with AsyncSessionLocal() as db:
                job = await _owned_job(db, job_id, user_id)
                result = await db.execute(
                    select(AgentJobLog)
                    .where(AgentJobLog.job_id == job_id, AgentJobLog.id > last_log_id)
                    .order_by(AgentJobLog.id)
                )
                logs = result.scalars().all()
                for log in logs:
                    last_log_id = max(last_log_id, log.id)
                    yield f"event: log\ndata: {json.dumps(serialize_log(log))}\n\n"
                yield f"event: status\ndata: {json.dumps(serialize_job(job))}\n\n"
                if job.status in TERMINAL_STATUSES:
                    break
            await asyncio.sleep(2)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/internal/jobs/{job_id}/log")
async def internal_log(
    job_id: str,
    request: AgentInternalLogRequest,
    x_agent_token: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
):
    _check_agent_token(job_id, x_agent_token)
    job = await db.get(AgentJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in TERMINAL_STATUSES or job.cancel_requested:
        raise HTTPException(status_code=409, detail="Job is no longer active")
    await append_job_log(db, job_id, request.message, level=request.level)
    job.updated_at = utcnow()
    await db.commit()
    return {"ok": True}


@router.post("/internal/jobs/{job_id}/step")
async def internal_step(
    job_id: str,
    request: AgentInternalStepRequest,
    x_agent_token: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
):
    _check_agent_token(job_id, x_agent_token)
    job = await db.get(AgentJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in TERMINAL_STATUSES or job.cancel_requested:
        raise HTTPException(status_code=409, detail="Job is no longer active")
    if request.status not in {"pending", "in_progress", "succeeded", "failed", "skipped", "canceled"}:
        raise HTTPException(status_code=400, detail="Unsupported step status")
    result = await db.execute(
        select(AgentJobStep).where(
            AgentJobStep.job_id == job_id,
            AgentJobStep.name == request.name,
        )
    )
    step = result.scalar_one_or_none()
    if not step:
        position_result = await db.execute(
            select(AgentJobStep.position)
            .where(AgentJobStep.job_id == job_id)
            .order_by(AgentJobStep.position.desc())
            .limit(1)
        )
        max_position = position_result.scalar_one_or_none() or 0
        step = AgentJobStep(job_id=job_id, position=max_position + 1, name=request.name)
        db.add(step)
    step.status = request.status
    step.exit_code = request.exit_code
    if request.status == "in_progress" and not step.started_at:
        step.started_at = utcnow()
    if request.status in {"succeeded", "failed", "skipped", "canceled"}:
        step.completed_at = utcnow()
    job.updated_at = utcnow()
    await db.commit()
    return {"ok": True}


@router.post("/internal/jobs/{job_id}/complete")
async def internal_complete(
    job_id: str,
    request: AgentInternalCompleteRequest,
    x_agent_token: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
):
    _check_agent_token(job_id, x_agent_token)
    job = await db.get(AgentJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in TERMINAL_STATUSES or job.cancel_requested:
        raise HTTPException(status_code=409, detail="Job is no longer active")
    if request.status not in TERMINAL_STATUSES:
        raise HTTPException(status_code=400, detail="Unsupported terminal status")
    job.status = request.status
    job.commit_sha = request.commit_sha
    job.pr_url = request.pr_url
    job.error_summary = request.error_summary
    job.completed_at = utcnow()
    job.updated_at = utcnow()
    if request.diff:
        db.add(
            AgentArtifact(
                job_id=job_id,
                kind="diff",
                name="changes.diff",
                content=clamp_diff(request.diff),
            )
        )
    await append_job_log(db, job_id, f"Job finished with status {request.status}.")
    await db.commit()
    await kubernetes_agent_executor.cleanup_secret(job_id)
    return {"ok": True}


@router.post("/internal/jobs/{job_id}/github-token")
async def internal_github_token(
    job_id: str,
    x_agent_token: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
):
    _check_agent_token(job_id, x_agent_token)
    job = await db.get(AgentJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in TERMINAL_STATUSES or job.cancel_requested:
        raise HTTPException(status_code=409, detail="Job is no longer active")
    installation = await current_installation(db, job.user_id)
    if not installation:
        raise HTTPException(status_code=404, detail="GitHub App is not connected")
    try:
        token_payload = await github_app_client.create_installation_token(installation.installation_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"GitHub token mint failed: {exc}") from exc
    return {"token": token_payload["token"]}


async def launch_agent_job(job_id: str) -> None:
    async with AsyncSessionLocal() as db:
        job = (
            await db.execute(
                select(AgentJob)
                .where(AgentJob.id == job_id)
                .options(selectinload(AgentJob.steps))
            )
        ).scalar_one_or_none()
        if not job or job.cancel_requested:
            return
        installation = await current_installation(db, job.user_id)
        if not installation:
            job.status = "failed"
            job.error_summary = "GitHub App is not connected"
            job.completed_at = utcnow()
            await append_job_log(db, job.id, job.error_summary, level="error")
            await db.commit()
            return

        try:
            job.status = "launching"
            job.started_at = utcnow()
            job.updated_at = utcnow()
            await append_job_log(db, job.id, "Minting short-lived GitHub installation token.")
            token_payload = await github_app_client.create_installation_token(installation.installation_id)
            token = token_payload["token"]
            if agent_executor_mode() != "kubernetes":
                raise RuntimeError(f"Unsupported agent executor mode: {agent_executor_mode()}")
            launch = await kubernetes_agent_executor.launch(job, token)
            job.status = "running"
            job.updated_at = utcnow()
            await append_job_log(
                db,
                job.id,
                f"Launched sandbox job {launch.namespace}/{launch.job_name}.",
            )
        except RuntimeError as exc:
            job.status = "failed"
            job.error_summary = str(exc)
            job.completed_at = utcnow()
            await append_job_log(db, job.id, str(exc), level="error")
        except httpx.HTTPError as exc:
            job.status = "failed"
            job.error_summary = f"Kubernetes or GitHub API error: {exc}"
            job.completed_at = utcnow()
            await append_job_log(db, job.id, job.error_summary, level="error")
        await db.commit()


async def _owned_job(
    db: AsyncSession,
    job_id: str,
    user_id: int,
    *,
    include_children: bool = False,
) -> AgentJob:
    stmt = select(AgentJob).where(AgentJob.id == job_id, AgentJob.user_id == user_id)
    if include_children:
        stmt = stmt.options(
            selectinload(AgentJob.steps),
            selectinload(AgentJob.artifacts),
        )
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


async def _validate_repo_branch_access(
    installation_id: str,
    repo_full_name: str,
    base_branch: str,
) -> None:
    owner, repo = repo_full_name.split("/", 1)
    try:
        branches = await github_app_client.branches(installation_id, owner, repo)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"GitHub branch validation failed: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"GitHub API unavailable: {exc}") from exc
    if not any(branch.get("name") == base_branch for branch in branches):
        raise HTTPException(status_code=400, detail="Selected branch is not visible to the GitHub App installation")


def _check_agent_token(job_id: str, value: str | None) -> None:
    expected = callback_token(job_id)
    if not expected or not value or not hmac.compare_digest(value, expected):
        raise HTTPException(status_code=401, detail="Invalid agent callback token")
