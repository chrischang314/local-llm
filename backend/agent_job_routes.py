from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_jobs import (
    AgentJobStateError,
    append_agent_job_event,
    create_agent_job,
    get_owned_agent_job,
    list_owned_agent_jobs,
    require_agent_jobs_enabled,
    serialize_agent_job,
    transition_agent_job,
)
from auth import current_user_id
from database import get_db
from github_app import GitHubAppClient, GitHubAppConfigurationError, GitHubAppRequestError
from models import GitHubInstallation
from secret_store import decrypt_secret


router = APIRouter(
    prefix="/agent/jobs",
    tags=["agent-jobs"],
    dependencies=[Depends(require_agent_jobs_enabled)],
)


class AgentJobCreateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    title: str | None = None
    repository_owner: str | None = None
    repository_name: str | None = None
    base_branch: str | None = "main"
    target_branch: str | None = None
    issue_number: int | None = None
    pull_request_number: int | None = None
    github_installation_id: str | None = None
    dispatch: bool = False
    dispatch_event: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentJobStateRequest(BaseModel):
    status: str
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None


class AgentJobEventRequest(BaseModel):
    event_type: str = "log"
    status: str | None = None
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def _configured_repo(
    owner: str | None,
    repo: str | None,
    client: GitHubAppClient,
) -> tuple[str | None, str | None]:
    return owner or client.config.default_owner, repo or client.config.default_repo


async def _current_github_connection(db: AsyncSession, user_id: int) -> GitHubInstallation | None:
    result = await db.execute(
        select(GitHubInstallation)
        .where(GitHubInstallation.user_id == user_id)
        .order_by(GitHubInstallation.updated_at.desc())
    )
    return result.scalars().first()


async def _dispatch_repository_event(
    *,
    db: AsyncSession,
    user_id: int,
    client: GitHubAppClient,
    owner: str,
    repo: str,
    event_type: str,
    client_payload: dict[str, Any],
) -> GitHubInstallation | None:
    connection = await _current_github_connection(db, user_id)
    if connection and (connection.auth_type or "app") == "oauth":
        try:
            token = decrypt_secret(connection.access_token_encrypted)
        except ValueError as exc:
            raise GitHubAppConfigurationError(str(exc)) from exc
        if not token:
            raise GitHubAppConfigurationError("GitHub OAuth token is missing; reconnect GitHub")
        await client.repository_dispatch_with_token(
            token=token,
            owner=owner,
            repo=repo,
            event_type=event_type,
            client_payload=client_payload,
        )
        return connection

    await client.repository_dispatch(
        owner=owner,
        repo=repo,
        event_type=event_type,
        client_payload=client_payload,
    )
    return connection


@router.post("", status_code=201)
async def create_job(
    body: AgentJobCreateRequest,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    github_client = GitHubAppClient()
    owner, repo = _configured_repo(body.repository_owner, body.repository_name, github_client)
    dispatch_event = body.dispatch_event or "local_llm.agent_job.requested"
    github_connection = await _current_github_connection(db, user_id)
    github_installation_id = body.github_installation_id or (
        github_connection.installation_id if github_connection else None
    )

    if body.dispatch and (not owner or not repo):
        raise HTTPException(
            status_code=400,
            detail="repository_owner/repository_name or GITHUB_DEFAULT_OWNER/GITHUB_DEFAULT_REPO is required",
        )

    try:
        job = await create_agent_job(
            db,
            user_id=user_id,
            prompt=body.prompt,
            title=body.title,
            repository_owner=owner,
            repository_name=repo,
            base_branch=body.base_branch,
            target_branch=body.target_branch,
            issue_number=body.issue_number,
            pull_request_number=body.pull_request_number,
            github_installation_id=github_installation_id,
            dispatch_event=dispatch_event if body.dispatch else None,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if body.dispatch:
        dispatch_payload = {
            "agent_job_id": job.id,
            "prompt": job.prompt,
            "title": job.title,
            "base_branch": job.base_branch,
            "target_branch": job.target_branch,
            "issue_number": job.issue_number,
            "pull_request_number": job.pull_request_number,
            "metadata": body.metadata,
        }
        try:
            github_connection = await _dispatch_repository_event(
                db=db,
                user_id=user_id,
                client=github_client,
                owner=owner,
                repo=repo,
                event_type=dispatch_event,
                client_payload=dispatch_payload,
            )
        except GitHubAppConfigurationError as exc:
            job = await transition_agent_job(
                db,
                job,
                status="failed",
                message=str(exc),
                payload={"dispatch_event": dispatch_event},
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GitHubAppRequestError as exc:
            job = await transition_agent_job(
                db,
                job,
                status="failed",
                message=str(exc),
                payload={"dispatch_event": dispatch_event},
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        await append_agent_job_event(
            db,
            job,
            event_type="github.dispatch",
            message="Repository dispatch accepted",
            payload={
                "owner": owner,
                "repo": repo,
                "event_type": dispatch_event,
                "auth_type": github_connection.auth_type if github_connection else "github_app",
            },
        )
        job = await get_owned_agent_job(db, user_id=user_id, job_id=job.id, include_events=True) or job

    return serialize_agent_job(job, include_events=True)


@router.get("")
async def list_jobs(
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    try:
        jobs = await list_owned_agent_jobs(db, user_id=user_id, status=status, limit=limit)
    except AgentJobStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [serialize_agent_job(job) for job in jobs]


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    job = await get_owned_agent_job(db, user_id=user_id, job_id=job_id, include_events=True)
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")
    return serialize_agent_job(job, include_events=True)


@router.post("/{job_id}/state")
async def update_job_state(
    job_id: str,
    body: AgentJobStateRequest,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    job = await get_owned_agent_job(db, user_id=user_id, job_id=job_id, include_events=True)
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")
    try:
        job = await transition_agent_job(
            db,
            job,
            status=body.status,
            message=body.message,
            payload=body.payload,
            result=body.result,
        )
    except AgentJobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return serialize_agent_job(job, include_events=True)


@router.post("/{job_id}/events", status_code=201)
async def append_job_event(
    job_id: str,
    body: AgentJobEventRequest,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    job = await get_owned_agent_job(db, user_id=user_id, job_id=job_id, include_events=True)
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")
    try:
        await append_agent_job_event(
            db,
            job,
            status=body.status,
            event_type=body.event_type,
            message=body.message,
            payload=body.payload,
        )
    except AgentJobStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = await get_owned_agent_job(db, user_id=user_id, job_id=job_id, include_events=True)
    return serialize_agent_job(job, include_events=True)
