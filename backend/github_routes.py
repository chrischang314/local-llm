"""Authenticated GitHub integration routes."""

import json
import secrets
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_jobs import (
    AgentJobStateError,
    append_agent_job_event,
    get_agent_job,
    require_agent_jobs_enabled,
    serialize_agent_job,
    transition_agent_job,
)
from auth import current_user, current_user_id
from database import get_db
from github_app import (
    GitHubAppClient,
    GitHubAppConfig,
    GitHubAppConfigurationError,
    GitHubAppRequestError,
    verify_webhook_signature,
)
from models import (
    GitHubInstallation,
    GitHubInstallState,
    GitHubOAuthConfig,
    GitHubOAuthServiceConfig,
    utcnow,
)
from secret_store import decrypt_secret, encrypt_secret


router = APIRouter(prefix="/github", tags=["github"])


class RepositoryDispatchRequest(BaseModel):
    repository_owner: str | None = None
    repository_name: str | None = None
    event_type: str = "local_llm.agent_job.requested"
    client_payload: dict[str, Any] = Field(default_factory=dict)


class GitHubOAuthConfigRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=200)
    client_secret: str | None = Field(default=None, max_length=500)


def _as_aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _serialize_connection(installation: GitHubInstallation | None) -> dict | None:
    if not installation:
        return None
    try:
        permissions = json.loads(installation.permissions_json or "{}")
    except json.JSONDecodeError:
        permissions = {}
    return {
        "installation_id": installation.installation_id,
        "auth_type": installation.auth_type or "app",
        "account_login": installation.account_login,
        "account_type": installation.account_type,
        "app_slug": installation.app_slug,
        "repository_selection": installation.repository_selection,
        "permissions": permissions,
        "token_scope": installation.token_scope,
        "updated_at": installation.updated_at.isoformat() if installation.updated_at else None,
    }


async def current_service_oauth_config(db: AsyncSession) -> GitHubOAuthServiceConfig | None:
    result = await db.execute(
        select(GitHubOAuthServiceConfig).order_by(GitHubOAuthServiceConfig.updated_at.desc())
    )
    config = result.scalars().first()
    if config:
        return config

    # Earlier iterations stored the OAuth app config on individual users. Treat
    # the first legacy config as a migration bridge so the deployed LAN app does
    # not lose an already-entered service OAuth app.
    legacy = (await db.execute(select(GitHubOAuthConfig).order_by(GitHubOAuthConfig.updated_at.desc()))).scalars().first()
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


async def current_github_connection(db: AsyncSession, user_id: int) -> GitHubInstallation | None:
    result = await db.execute(
        select(GitHubInstallation)
        .where(GitHubInstallation.user_id == user_id)
        .order_by(GitHubInstallation.updated_at.desc())
    )
    return result.scalars().first()


def connection_uses_oauth(installation: GitHubInstallation | None) -> bool:
    return ((installation.auth_type if installation else "") or "app") == "oauth"


async def github_token_for_connection(installation: GitHubInstallation) -> dict[str, Any]:
    if connection_uses_oauth(installation):
        try:
            token = decrypt_secret(installation.access_token_encrypted)
        except ValueError as exc:
            raise GitHubAppConfigurationError(str(exc)) from exc
        if not token:
            raise GitHubAppConfigurationError("GitHub OAuth token is missing; reconnect GitHub")
        return {"token": token, "expires_at": None}
    client = GitHubAppClient()
    payload = await client._request(
        "POST",
        f"/app/installations/{installation.installation_id}/access_tokens",
        token=client.app_jwt(),
    )
    token = payload.get("token") if payload else None
    if not token:
        raise GitHubAppRequestError("GitHub App installation token response did not include a token")
    return {"token": token, "expires_at": payload.get("expires_at")}


def _frontend_base_url(request: Request) -> str:
    return str(request.url_for("github_oauth_callback")).split("/github/oauth/callback", 1)[0] or "/"


def _extract_job_id(payload: dict[str, Any]) -> str | None:
    candidates = [
        payload,
        payload.get("client_payload") or {},
        payload.get("repository_dispatch") or {},
        payload.get("workflow_job") or {},
        payload.get("workflow_run") or {},
        payload.get("check_run") or {},
    ]
    for item in candidates:
        if not isinstance(item, dict):
            continue
        for key in ("agent_job_id", "job_id", "external_id"):
            value = item.get(key)
            if isinstance(value, str) and value.startswith("job_"):
                return value
    return None


def _status_from_github_event(event: str | None, payload: dict[str, Any]) -> str | None:
    action = (payload.get("action") or "").lower()
    subject = payload.get("workflow_job") or payload.get("workflow_run") or payload.get("check_run") or {}
    status = (subject.get("status") or "").lower()
    conclusion = (subject.get("conclusion") or "").lower()

    if action in {"queued", "requested", "rerequested"} or status == "queued":
        return "queued"
    if action == "in_progress" or status == "in_progress":
        return "running"
    if action == "completed" or status == "completed":
        if conclusion == "success":
            return "succeeded"
        if conclusion in {"cancelled", "timed_out"}:
            return "cancelled"
        return "failed"
    if event == "repository_dispatch":
        return "queued"
    return None


def _github_event_message(event: str | None, payload: dict[str, Any], status: str | None) -> str:
    action = payload.get("action")
    subject = payload.get("workflow_job") or payload.get("workflow_run") or payload.get("check_run") or {}
    conclusion = subject.get("conclusion")
    parts = ["GitHub"]
    if event:
        parts.append(event)
    if action:
        parts.append(action)
    if status:
        parts.append(f"mapped to {status}")
    if conclusion:
        parts.append(f"({conclusion})")
    return " ".join(parts)


@router.get("/status")
async def github_status(
    request: Request,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    oauth_config = await current_service_oauth_config(db)
    connection = await current_github_connection(db, user_id)
    app_config = GitHubAppConfig.from_env()
    service_configured = oauth_config is not None
    legacy_configured = app_config.configured
    configured = service_configured or legacy_configured
    missing = [] if configured else ["GitHub OAuth Client ID", "GitHub OAuth Client Secret"]
    return {
        "configured": configured,
        "connected": connection is not None,
        "mode": "oauth" if service_configured or connection_uses_oauth(connection) else "github_app",
        "missing": missing,
        "oauth": {
            "configured": service_configured,
            "client_id_configured": bool(oauth_config and oauth_config.client_id),
            "callback_url": str(request.url_for("github_oauth_callback")),
            "updated_at": oauth_config.updated_at.isoformat() if oauth_config and oauth_config.updated_at else None,
        },
        "legacy_app_configured": legacy_configured,
        "connection": _serialize_connection(connection),
        "installation": _serialize_connection(connection),
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
    config = await current_service_oauth_config(db)
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
    return {
        "configured": True,
        "oauth": {
            "configured": True,
            "client_id_configured": True,
            "callback_url": str(request.url_for("github_oauth_callback")),
            "updated_at": config.updated_at.isoformat() if config.updated_at else None,
        },
    }


@router.post("/oauth/start")
async def start_github_oauth(
    request: Request,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    config = await current_service_oauth_config(db)
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
        "auth_url": GitHubAppClient.oauth_authorize_url(config.client_id, redirect_uri, state),
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
    frontend_url = _frontend_base_url(request)
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

    config = await current_service_oauth_config(db)
    if not config:
        return RedirectResponse(f"{frontend_url}/?github_oauth=error&message=GitHub%20OAuth%20is%20not%20configured", status_code=303)

    redirect_uri = str(request.url_for("github_oauth_callback"))
    github_client = GitHubAppClient()
    try:
        token_payload = await github_client.exchange_oauth_code(
            client_id=config.client_id,
            client_secret=decrypt_secret(config.client_secret_encrypted),
            code=code,
            redirect_uri=redirect_uri,
        )
        github_user = await github_client.oauth_user(token_payload["access_token"])
    except (GitHubAppRequestError, GitHubAppConfigurationError, ValueError, httpx.HTTPError) as exc:
        message = urllib.parse.quote(str(exc))
        return RedirectResponse(f"{frontend_url}/?github_oauth=error&message={message}", status_code=303)

    connection = await current_github_connection(db, state_row.user_id)
    if not connection:
        connection = GitHubInstallation(
            user_id=state_row.user_id,
            installation_id=f"oauth:{github_user.get('login') or github_user.get('id')}",
        )
        db.add(connection)

    scopes = token_payload.get("scope") or ""
    connection.installation_id = f"oauth:{github_user.get('login') or github_user.get('id')}"
    connection.auth_type = "oauth"
    connection.account_login = github_user.get("login")
    connection.account_type = github_user.get("type") or "User"
    connection.app_slug = "oauth"
    connection.repository_selection = "all-visible-to-token"
    connection.permissions_json = json.dumps({"scopes": scopes}, sort_keys=True)
    connection.access_token_encrypted = encrypt_secret(token_payload["access_token"])
    connection.token_scope = scopes
    connection.token_type = token_payload.get("token_type") or "bearer"
    connection.updated_at = utcnow()
    state_row.consumed = True
    await db.commit()
    return RedirectResponse(f"{frontend_url}/?github_oauth=connected", status_code=303)


@router.delete("/install")
async def disconnect_github(
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    connection = await current_github_connection(db, user_id)
    if connection:
        await db.delete(connection)
        await db.commit()
    return {"ok": True}


@router.get("/repos")
@router.get("/repositories")
async def list_repositories(
    query: str = "",
    page: int = Query(default=1, ge=1),
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    connection = await current_github_connection(db, user_id)
    if not connection:
        raise HTTPException(status_code=404, detail="GitHub is not connected")
    try:
        if connection_uses_oauth(connection):
            token = (await github_token_for_connection(connection))["token"]
            return await GitHubAppClient().repositories_for_token(token, query=query, page=page)
        return await GitHubAppClient().installation_request(
            "GET",
            "/installation/repositories",
            json_body=None,
        )
    except GitHubAppConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GitHubAppRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
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
    connection = await current_github_connection(db, user_id)
    if not connection:
        raise HTTPException(status_code=404, detail="GitHub is not connected")
    try:
        token = (await github_token_for_connection(connection))["token"]
        return {"branches": await GitHubAppClient().branches_for_token(token, owner, repo)}
    except GitHubAppConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GitHubAppRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"GitHub branch lookup failed: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"GitHub API unavailable: {exc}") from exc


@router.post("/repository-dispatch", status_code=202, dependencies=[Depends(require_agent_jobs_enabled)])
async def repository_dispatch(
    body: RepositoryDispatchRequest,
    user_id: int = Depends(current_user_id),
    db: AsyncSession = Depends(get_db),
):
    owner = body.repository_owner
    repo = body.repository_name
    if not owner or not repo:
        raise HTTPException(status_code=400, detail="repository_owner and repository_name are required")
    connection = await current_github_connection(db, user_id)
    if not connection:
        raise HTTPException(status_code=404, detail="GitHub is not connected")
    try:
        token = (await github_token_for_connection(connection))["token"]
        await GitHubAppClient().repository_dispatch_with_token(
            token=token,
            owner=owner,
            repo=repo,
            event_type=body.event_type,
            client_payload=body.client_payload,
        )
    except GitHubAppConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GitHubAppRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "owner": owner, "repo": repo, "event_type": body.event_type}


@router.post("/webhook")
async def github_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    x_github_event: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
):
    body = await request.body()
    config = GitHubAppConfig.from_env()
    if config.webhook_secret and not verify_webhook_signature(
        config.webhook_secret,
        body,
        x_hub_signature_256,
    ):
        raise HTTPException(status_code=401, detail="Invalid GitHub webhook signature")

    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    job_id = _extract_job_id(payload)
    next_status = _status_from_github_event(x_github_event, payload)
    job_updated = False
    serialized_job = None

    if job_id:
        job = await get_agent_job(db, job_id, include_events=True)
        if job:
            event_payload = {
                "github_event": x_github_event,
                "github_delivery": x_github_delivery,
                "action": payload.get("action"),
            }
            message = _github_event_message(x_github_event, payload, next_status)
            if next_status:
                try:
                    job = await transition_agent_job(
                        db,
                        job,
                        status=next_status,
                        message=message,
                        payload=event_payload,
                    )
                    job_updated = True
                except AgentJobStateError:
                    await append_agent_job_event(
                        db,
                        job,
                        event_type="github.webhook",
                        message=message,
                        payload=event_payload,
                    )
            else:
                await append_agent_job_event(
                    db,
                    job,
                    event_type="github.webhook",
                    message=message,
                    payload=event_payload,
                )
            job = await get_agent_job(db, job_id, include_events=True)
            serialized_job = serialize_agent_job(job, include_events=True) if job else None

    return {
        "ok": True,
        "event": x_github_event,
        "delivery": x_github_delivery,
        "job_id": job_id,
        "job_updated": job_updated,
        "job": serialized_job,
    }
