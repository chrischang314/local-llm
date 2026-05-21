"""Authenticated GitHub integration routes."""

import json
import os
import secrets
import urllib.parse
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_services import utcnow
from auth import current_user_id
from database import get_db
from github_client import github_app_client
from models import GitHubInstallation, GitHubInstallState, GitHubOAuthConfig, GitHubOAuthServiceConfig
from secret_store import decrypt_secret, encrypt_secret

router = APIRouter(prefix="/github", tags=["github"])


class GitHubInstallCompleteRequest(BaseModel):
    installation_id: str
    state: str


class GitHubOAuthConfigRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=200)
    client_secret: str | None = Field(default=None, max_length=500)


def _serialize_installation(installation: GitHubInstallation | None) -> dict | None:
    if not installation:
        return None
    return {
        "installation_id": installation.installation_id,
        "auth_type": installation.auth_type or "app",
        "account_login": installation.account_login,
        "account_type": installation.account_type,
        "app_slug": installation.app_slug,
        "repository_selection": installation.repository_selection,
        "permissions": json.loads(installation.permissions_json or "{}"),
        "token_scope": installation.token_scope,
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


async def current_oauth_config(db: AsyncSession) -> GitHubOAuthServiceConfig | None:
    result = await db.execute(
        select(GitHubOAuthServiceConfig).order_by(GitHubOAuthServiceConfig.updated_at.desc())
    )
    config = result.scalars().first()
    if config:
        return config

    # Migration bridge for the previous UI that stored the OAuth App credentials
    # per Local LLM user. Promote the most recent row to the service-wide config
    # so existing deployments do not lose a configured OAuth App.
    legacy = (
        await db.execute(select(GitHubOAuthConfig).order_by(GitHubOAuthConfig.updated_at.desc()))
    ).scalars().first()
    if not legacy:
        return None
    config = GitHubOAuthServiceConfig(
        id=1,
        client_id=legacy.client_id,
        client_secret_encrypted=legacy.client_secret_encrypted,
        created_by_user_id=legacy.user_id,
        updated_by_user_id=legacy.user_id,
    )
    db.add(config)
    await db.commit()
    await db.refresh(config)
    return config


def _serialize_oauth_config(config: GitHubOAuthServiceConfig | None, request: Request) -> dict:
    return {
        "configured": config is not None,
        "client_id_configured": bool(config and config.client_id),
        "callback_url": str(request.url_for("github_oauth_callback")),
        "updated_at": config.updated_at.isoformat() if config and config.updated_at else None,
    }


def connection_auth_type(installation: GitHubInstallation | None) -> str:
    return (installation.auth_type if installation else "") or "app"


def connection_uses_oauth(installation: GitHubInstallation | None) -> bool:
    return connection_auth_type(installation) == "oauth"


async def github_token_for_installation(installation: GitHubInstallation) -> dict:
    if connection_uses_oauth(installation):
        try:
            token = decrypt_secret(installation.access_token_encrypted)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if not token:
            raise RuntimeError("GitHub OAuth token is missing; reconnect GitHub")
        return {"token": token, "expires_at": None}
    return await github_app_client.create_installation_token(installation.installation_id)


@router.get("/status")
async def github_status(
    request: Request,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    app_config = github_app_client.config()
    oauth_config = await current_oauth_config(db)
    installation = await current_installation(db, user_id)
    bypass_configured = github_app_client.bypass_token_configured()
    oauth_configured = oauth_config is not None
    legacy_configured = app_config.configured or bypass_configured
    configured = oauth_configured or legacy_configured
    missing = [] if configured else ["GitHub OAuth Client ID", "GitHub OAuth Client Secret"]
    mode = "oauth" if oauth_configured or connection_uses_oauth(installation) else (
        "bypass_token" if bypass_configured and not app_config.configured else "github_app"
    )
    return {
        "configured": configured,
        "missing": missing,
        "mode": mode,
        "legacy_app_configured": app_config.configured,
        "bypass_configured": bypass_configured,
        "oauth": _serialize_oauth_config(oauth_config, request),
        "connected": installation is not None,
        "connection": _serialize_installation(installation),
        "installation": _serialize_installation(installation),
    }


@router.post("/oauth/config")
async def save_github_oauth_config(
    request_body: GitHubOAuthConfigRequest,
    request: Request,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    client_id = request_body.client_id.strip()
    client_secret = (request_body.client_secret or "").strip()
    config = await current_oauth_config(db)
    if not config and not client_secret:
        raise HTTPException(status_code=400, detail="GitHub OAuth Client Secret is required")
    if not config:
        config = GitHubOAuthServiceConfig(
            id=1,
            client_id=client_id,
            client_secret_encrypted=encrypt_secret(client_secret),
            created_by_user_id=user_id,
            updated_by_user_id=user_id,
        )
        db.add(config)
    else:
        config.client_id = client_id
        if client_secret:
            config.client_secret_encrypted = encrypt_secret(client_secret)
        config.updated_by_user_id = user_id
        config.updated_at = utcnow()
    await db.commit()
    await db.refresh(config)
    return {"configured": True, "oauth": _serialize_oauth_config(config, request)}


@router.post("/oauth/start")
async def start_github_oauth(
    request: Request,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    config = await current_oauth_config(db)
    if not config:
        return {
            "configured": False,
            "missing": ["GitHub OAuth Client ID", "GitHub OAuth Client Secret"],
            "auth_url": None,
            "callback_url": str(request.url_for("github_oauth_callback")),
        }

    state = secrets.token_urlsafe(32)
    redirect_uri = str(request.url_for("github_oauth_callback"))
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
        "auth_url": github_app_client.oauth_authorize_url(config.client_id, redirect_uri, state),
        "state": state,
        "callback_url": redirect_uri,
        "mode": "oauth",
    }


@router.get("/oauth/callback", name="github_oauth_callback")
async def github_oauth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    frontend_url = str(request.url_for("github_oauth_callback")).split("/github/oauth/callback", 1)[0] or "/"
    if error:
        message = urllib.parse.quote(error_description or error)
        return RedirectResponse(f"{frontend_url}/?github_oauth=error&message={message}", status_code=303)
    if not code or not state:
        return RedirectResponse(f"{frontend_url}/?github_oauth=error&message=Missing%20GitHub%20OAuth%20code", status_code=303)

    state_row = (
        await db.execute(select(GitHubInstallState).where(GitHubInstallState.state == state))
    ).scalar_one_or_none()
    expires_at = _as_aware_utc(state_row.expires_at) if state_row else None
    if not state_row or state_row.consumed or not expires_at or expires_at < utcnow():
        return RedirectResponse(f"{frontend_url}/?github_oauth=error&message=Expired%20GitHub%20OAuth%20state", status_code=303)

    config = await current_oauth_config(db)
    if not config:
        return RedirectResponse(f"{frontend_url}/?github_oauth=error&message=GitHub%20OAuth%20is%20not%20configured", status_code=303)

    redirect_uri = str(request.url_for("github_oauth_callback"))
    try:
        token_payload = await github_app_client.exchange_oauth_code(
            client_id=config.client_id,
            client_secret=decrypt_secret(config.client_secret_encrypted),
            code=code,
            redirect_uri=redirect_uri,
        )
        user = await github_app_client.oauth_user(token_payload["access_token"])
    except (RuntimeError, ValueError, httpx.HTTPError) as exc:
        message = urllib.parse.quote(str(exc))
        return RedirectResponse(f"{frontend_url}/?github_oauth=error&message={message}", status_code=303)

    installation = await current_installation(db, state_row.user_id)
    if not installation:
        installation = GitHubInstallation(
            user_id=state_row.user_id,
            installation_id=f"oauth:{user.get('login') or user.get('id')}",
        )
        db.add(installation)

    scopes = token_payload.get("scope") or ""
    installation.installation_id = f"oauth:{user.get('login') or user.get('id')}"
    installation.auth_type = "oauth"
    installation.account_login = user.get("login")
    installation.account_type = user.get("type") or "User"
    installation.app_slug = "oauth"
    installation.repository_selection = "all-visible-to-token"
    installation.permissions_json = json.dumps({"scopes": scopes}, sort_keys=True)
    installation.access_token_encrypted = encrypt_secret(token_payload["access_token"])
    installation.token_scope = scopes
    installation.token_type = token_payload.get("token_type") or "bearer"
    installation.updated_at = utcnow()
    state_row.consumed = True
    await db.commit()
    return RedirectResponse(f"{frontend_url}/?github_oauth=connected", status_code=303)


@router.post("/install/start")
async def start_github_install(
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    config = github_app_client.config()
    if github_app_client.bypass_token_configured() and not config.configured:
        return {
            "configured": True,
            "missing": [],
            "install_url": None,
            "state": None,
            "mode": "bypass_token",
        }
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
    installation.auth_type = "app"
    installation.account_login = summary["account_login"]
    installation.account_type = summary["account_type"]
    installation.app_slug = summary["app_slug"]
    installation.repository_selection = summary["repository_selection"]
    installation.permissions_json = json.dumps(summary["permissions"], sort_keys=True)
    installation.access_token_encrypted = None
    installation.token_scope = None
    installation.token_type = None
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
        raise HTTPException(status_code=404, detail="GitHub is not connected")
    try:
        if connection_uses_oauth(installation):
            token = (await github_token_for_installation(installation))["token"]
            return await github_app_client.repositories_for_token(token, query=query, page=page)
        return await github_app_client.repositories(installation.installation_id, query=query, page=page)
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
        raise HTTPException(status_code=404, detail="GitHub is not connected")
    try:
        if connection_uses_oauth(installation):
            token = (await github_token_for_installation(installation))["token"]
            return {"branches": await github_app_client.branches_for_token(token, owner, repo)}
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
