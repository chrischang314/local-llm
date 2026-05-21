"""Small GitHub client used by the agentic code workflow."""

import os
import pathlib
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import jwt

GITHUB_API = "https://api.github.com"
GITHUB_WEB = "https://github.com"
GITHUB_API_VERSION = "2026-03-10"


@dataclass(frozen=True)
class GitHubAppConfig:
    app_id: str | None
    app_slug: str | None
    private_key: str | None

    @property
    def missing(self) -> list[str]:
        missing = []
        if not self.app_id:
            missing.append("GITHUB_APP_ID")
        if not self.app_slug:
            missing.append("GITHUB_APP_SLUG")
        if not self.private_key:
            missing.append("GITHUB_APP_PRIVATE_KEY or GITHUB_APP_PRIVATE_KEY_FILE")
        return missing

    @property
    def configured(self) -> bool:
        return not self.missing


class GitHubAppClient:
    def _read_secret(self, env_name: str, file_env_name: str) -> str:
        value = os.getenv(env_name, "")
        path = os.getenv(file_env_name, "")
        if value:
            return value
        if not path:
            return ""
        try:
            return pathlib.Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def bypass_token(self) -> str:
        """Emergency local test token used only when no GitHub App exists.

        This is intentionally separate from the GitHub App config because it is
        a trusted-LAN/testing escape hatch, not the production authorization
        model. Prefer GitHub App installation tokens for normal operation.
        """
        return self._read_secret("GITHUB_BYPASS_TOKEN", "GITHUB_BYPASS_TOKEN_FILE")

    def bypass_token_configured(self) -> bool:
        return bool(self.bypass_token())

    def config(self) -> GitHubAppConfig:
        key = self._read_secret("GITHUB_APP_PRIVATE_KEY", "GITHUB_APP_PRIVATE_KEY_FILE")
        if key:
            key = key.replace("\\n", "\n")
        return GitHubAppConfig(
            app_id=os.getenv("GITHUB_APP_ID"),
            app_slug=os.getenv("GITHUB_APP_SLUG"),
            private_key=key,
        )

    def install_url(self, state: str) -> str | None:
        config = self.config()
        if not config.app_slug:
            return None
        return f"{GITHUB_WEB}/apps/{config.app_slug}/installations/new?state={state}"

    @staticmethod
    def oauth_authorize_url(client_id: str, redirect_uri: str, state: str) -> str:
        params = urllib.parse.urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": "repo",
                "state": state,
                "allow_signup": "true",
            }
        )
        return f"{GITHUB_WEB}/login/oauth/authorize?{params}"

    async def exchange_oauth_code(
        self,
        *,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{GITHUB_WEB}/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            response.raise_for_status()
            payload = response.json()
        if payload.get("error"):
            raise RuntimeError(payload.get("error_description") or payload["error"])
        if not payload.get("access_token"):
            raise RuntimeError("GitHub OAuth did not return an access token")
        return payload

    def app_jwt(self) -> str:
        config = self.config()
        if not config.configured:
            raise RuntimeError(f"GitHub App is not configured: {', '.join(config.missing)}")
        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + 9 * 60,
            "iss": config.app_id,
        }
        return jwt.encode(payload, config.private_key, algorithm="RS256")

    def app_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.app_jwt()}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }

    @staticmethod
    def token_headers(token: str) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }

    async def get_installation(self, installation_id: str) -> dict[str, Any]:
        token = self.bypass_token()
        if token:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{GITHUB_API}/user", headers=self.token_headers(token))
                response.raise_for_status()
                user = response.json()
            return {
                "id": installation_id,
                "account": {
                    "login": user.get("login"),
                    "type": user.get("type") or "User",
                },
                "app_slug": "bypass-token",
                "repository_selection": "all",
                "permissions": {
                    "contents": "write",
                    "metadata": "read",
                    "pull_requests": "write",
                },
            }
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{GITHUB_API}/app/installations/{installation_id}",
                headers=self.app_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def oauth_user(self, token: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{GITHUB_API}/user", headers=self.token_headers(token))
            response.raise_for_status()
            return response.json()

    async def create_installation_token(self, installation_id: str) -> dict[str, Any]:
        token = self.bypass_token()
        if token:
            return {"token": token, "expires_at": None}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
                headers=self.app_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def repositories(
        self,
        installation_id: str,
        *,
        query: str = "",
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        token = (await self.create_installation_token(installation_id))["token"]
        async with httpx.AsyncClient(timeout=20.0) as client:
            if self.bypass_token_configured():
                response = await client.get(
                    f"{GITHUB_API}/user/repos",
                    headers=self.token_headers(token),
                    params={
                        "affiliation": "owner,collaborator,organization_member",
                        "sort": "updated",
                        "page": max(page, 1),
                        "per_page": min(max(per_page, 1), 100),
                    },
                )
            else:
                response = await client.get(
                    f"{GITHUB_API}/installation/repositories",
                    headers=self.token_headers(token),
                    params={"page": max(page, 1), "per_page": min(max(per_page, 1), 100)},
                )
            response.raise_for_status()
            payload = response.json()

        repos = payload if isinstance(payload, list) else payload.get("repositories", [])
        if query:
            needle = query.lower()
            repos = [
                repo for repo in repos
                if needle in repo.get("full_name", "").lower()
                or needle in (repo.get("description") or "").lower()
            ]
        return {
            "repositories": [self._repo_summary(repo) for repo in repos],
            "total_count": len(repos) if isinstance(payload, list) else payload.get("total_count", len(repos)),
        }

    async def repositories_for_token(
        self,
        token: str,
        *,
        query: str = "",
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{GITHUB_API}/user/repos",
                headers=self.token_headers(token),
                params={
                    "affiliation": "owner,collaborator,organization_member",
                    "sort": "updated",
                    "page": max(page, 1),
                    "per_page": min(max(per_page, 1), 100),
                },
            )
            response.raise_for_status()
            repos = response.json()
        if query:
            needle = query.lower()
            repos = [
                repo for repo in repos
                if needle in repo.get("full_name", "").lower()
                or needle in (repo.get("description") or "").lower()
            ]
        return {
            "repositories": [self._repo_summary(repo) for repo in repos],
            "total_count": len(repos),
        }

    async def branches(self, installation_id: str, owner: str, repo: str) -> list[dict[str, Any]]:
        token = (await self.create_installation_token(installation_id))["token"]
        return await self.branches_for_token(token, owner, repo)

    async def branches_for_token(self, token: str, owner: str, repo: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/branches",
                headers=self.token_headers(token),
                params={"per_page": 100},
            )
            response.raise_for_status()
            return [
                {
                    "name": branch.get("name"),
                    "protected": bool(branch.get("protected")),
                    "commit_sha": branch.get("commit", {}).get("sha"),
                }
                for branch in response.json()
            ]

    @staticmethod
    def installation_summary(payload: dict[str, Any]) -> dict[str, Any]:
        account = payload.get("account") or {}
        permissions = payload.get("permissions") or {}
        return {
            "installation_id": str(payload.get("id")),
            "account_login": account.get("login"),
            "account_type": account.get("type"),
            "app_slug": payload.get("app_slug"),
            "repository_selection": payload.get("repository_selection"),
            "permissions": permissions,
        }

    @staticmethod
    def _repo_summary(repo: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": repo.get("id"),
            "name": repo.get("name"),
            "full_name": repo.get("full_name"),
            "private": bool(repo.get("private")),
            "description": repo.get("description"),
            "default_branch": repo.get("default_branch"),
            "html_url": repo.get("html_url"),
            "archived": bool(repo.get("archived")),
            "disabled": bool(repo.get("disabled")),
        }


def installation_token_expires_at(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("expires_at")
    if not raw:
        return datetime.now(timezone.utc) + timedelta(hours=1)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


github_app_client = GitHubAppClient()
