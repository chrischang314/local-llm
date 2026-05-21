"""Authenticated GitHub App integration routes."""

import json
import os
import secrets
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_services import utcnow
from auth import current_user_id
from database import get_db
from github_client import github_app_client
from models import GitHubInstallation, GitHubInstallState

router = APIRouter(prefix="/github", tags=["github"])


class GitHubInstallCompleteRequest(BaseModel):
    installation_id: str
    state: str


def _serialize_installation(installation: GitHubInstallation | None) -> dict | None:
    if not installation:
        return None
    return {
        "installation_id": installation.installation_id,
        "account_login": installation.account_login,
        "account_type": installation.account_type,
        "app_slug": installation.app_slug,
        "repository_selection": installation.repository_selection,
        "permissions": json.loads(installation.permissions_json or "{}"),
        "updated_at": installation.updated_at.isoformat() if installation.updated_at else None,
    }


def _as_aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def allowed_installation_ids() -> set[str]:
    raw = os.getenv("GITHUB_ALLOWED_INSTALLATION_IDS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


async def current_installation(db: AsyncSession, user_id: int) -> GitHubInstallation | None:
    result = await db.execute(
        select(GitHubInstallation)
        .where(GitHubInstallation.user_id == user_id)
        .order_by(GitHubInstallation.updated_at.desc())
    )
    return result.scalars().first()


@router.get("/status")
async def github_status(
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    config = github_app_client.config()
    installation = await current_installation(db, user_id)
    return {
        "configured": config.configured,
        "missing": config.missing,
        "connected": installation is not None,
        "installation": _serialize_installation(installation),
    }


@router.post("/install/start")
async def start_github_install(
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    config = github_app_client.config()
    if not config.configured:
        return {
            "configured": False,
            "missing": config.missing,
            "install_url": None,
            "state": None,
        }

    state = secrets.token_urlsafe(32)
    db.add(
        GitHubInstallState(
            user_id=user_id,
            state=state,
            expires_at=utcnow() + timedelta(minutes=30),
        )
    )
    await db.commit()
    return {
        "configured": True,
        "missing": [],
        "install_url": github_app_client.install_url(state),
        "state": state,
    }


@router.post("/install/complete")
async def complete_github_install(
    request: GitHubInstallCompleteRequest,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    state = (
        await db.execute(
            select(GitHubInstallState).where(
                GitHubInstallState.user_id == user_id,
                GitHubInstallState.state == request.state,
            )
        )
    ).scalar_one_or_none()
    expires_at = _as_aware_utc(state.expires_at) if state else None
    if not state or state.consumed or not expires_at or expires_at < utcnow():
        raise HTTPException(status_code=400, detail="Installation state is invalid or expired")
    allowed_ids = allowed_installation_ids()
    if allowed_ids and request.installation_id not in allowed_ids:
        raise HTTPException(status_code=403, detail="GitHub installation id is not allowed for this deployment")

    existing_owner = (
        await db.execute(
            select(GitHubInstallation).where(
                GitHubInstallation.installation_id == request.installation_id,
                GitHubInstallation.user_id != user_id,
            )
        )
    ).scalar_one_or_none()
    if existing_owner:
        raise HTTPException(status_code=409, detail="GitHub installation is already connected to another local user")

    try:
        raw_installation = await github_app_client.get_installation(request.installation_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"GitHub installation lookup failed: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"GitHub API unavailable: {exc}") from exc

    summary = github_app_client.installation_summary(raw_installation)
    installation = await current_installation(db, user_id)
    if not installation:
        installation = GitHubInstallation(user_id=user_id, installation_id=summary["installation_id"])
        db.add(installation)

    installation.installation_id = summary["installation_id"]
    installation.account_login = summary["account_login"]
    installation.account_type = summary["account_type"]
    installation.app_slug = summary["app_slug"]
    installation.repository_selection = summary["repository_selection"]
    installation.permissions_json = json.dumps(summary["permissions"], sort_keys=True)
    installation.updated_at = utcnow()
    state.consumed = True
    await db.commit()
    await db.refresh(installation)
    return {"connected": True, "installation": _serialize_installation(installation)}


@router.delete("/install")
async def disconnect_github(
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    installation = await current_installation(db, user_id)
    if installation:
        await db.delete(installation)
        await db.commit()
    return {"ok": True}


@router.get("/repos")
async def list_repositories(
    query: str = "",
    page: int = Query(default=1, ge=1),
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    installation = await current_installation(db, user_id)
    if not installation:
        raise HTTPException(status_code=404, detail="GitHub App is not connected")
    try:
        return await github_app_client.repositories(
            installation.installation_id,
            query=query,
            page=page,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"GitHub repository lookup failed: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"GitHub API unavailable: {exc}") from exc


@router.get("/repos/{owner}/{repo}/branches")
async def list_branches(
    owner: str,
    repo: str,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    installation = await current_installation(db, user_id)
    if not installation:
        raise HTTPException(status_code=404, detail="GitHub App is not connected")
    try:
        return {"branches": await github_app_client.branches(installation.installation_id, owner, repo)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"GitHub branch lookup failed: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"GitHub API unavailable: {exc}") from exc
