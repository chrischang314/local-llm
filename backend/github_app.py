import hmac
import os
import pathlib
import time
import urllib.parse
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import httpx
import jwt


class GitHubAppConfigurationError(RuntimeError):
    pass


class GitHubAppRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitHubAppConfig:
    app_id: str | None
    installation_id: str | None
    private_key: str | None
    api_url: str
    default_owner: str | None
    default_repo: str | None
    webhook_secret: str | None
    user_agent: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "GitHubAppConfig":
        return cls(
            app_id=os.getenv("GITHUB_APP_ID"),
            installation_id=os.getenv("GITHUB_APP_INSTALLATION_ID"),
            private_key=_load_private_key(),
            api_url=os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/"),
            default_owner=os.getenv("GITHUB_DEFAULT_OWNER") or os.getenv("GITHUB_REPOSITORY_OWNER"),
            default_repo=os.getenv("GITHUB_DEFAULT_REPO") or os.getenv("GITHUB_REPOSITORY_NAME"),
            webhook_secret=os.getenv("GITHUB_WEBHOOK_SECRET"),
            user_agent=os.getenv("GITHUB_APP_USER_AGENT", "local-llm-agent-jobs"),
            timeout_seconds=float(os.getenv("GITHUB_APP_TIMEOUT_SECONDS", "15")),
        )

    @property
    def app_credentials_configured(self) -> bool:
        return bool(self.app_id and self.private_key)

    @property
    def installation_configured(self) -> bool:
        return bool(self.installation_id or (self.default_owner and self.default_repo))

    @property
    def configured(self) -> bool:
        return self.app_credentials_configured and self.installation_configured


def _load_private_key() -> str | None:
    raw = os.getenv("GITHUB_APP_PRIVATE_KEY")
    if raw:
        return raw.replace("\\n", "\n")

    key_file = os.getenv("GITHUB_APP_PRIVATE_KEY_FILE")
    if not key_file:
        return None

    try:
        return pathlib.Path(key_file).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise GitHubAppConfigurationError(f"Cannot read GitHub App private key file: {exc}") from exc


def verify_webhook_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


class GitHubAppClient:
    def __init__(self, config: GitHubAppConfig | None = None):
        self.config = config or GitHubAppConfig.from_env()

    def app_jwt(self) -> str:
        if not self.config.app_id or not self.config.private_key:
            raise GitHubAppConfigurationError(
                "GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY or GITHUB_APP_PRIVATE_KEY_FILE are required"
            )

        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + 9 * 60,
            "iss": self.config.app_id,
        }
        return jwt.encode(payload, self.config.private_key, algorithm="RS256")

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": self.config.user_agent,
            "X-GitHub-Api-Version": "2022-11-28",
        }

    @staticmethod
    def oauth_authorize_url(client_id: str, redirect_uri: str, state: str) -> str:
        params = urllib.parse.urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": "repo",
                "state": state,
                "allow_signup": "true",
                "prompt": "select_account",
            }
        )
        return f"https://github.com/login/oauth/authorize?{params}"

    async def exchange_oauth_code(
        self,
        *,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                "https://github.com/login/oauth/access_token",
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
            raise GitHubAppRequestError(payload.get("error_description") or payload["error"])
        if not payload.get("access_token"):
            raise GitHubAppRequestError("GitHub OAuth did not return an access token")
        return payload

    async def oauth_user(self, token: str) -> dict[str, Any]:
        payload = await self._request("GET", "/user", token=token)
        if not payload:
            raise GitHubAppRequestError("GitHub user lookup returned no data")
        return payload

    async def _request(
        self,
        method: str,
        path: str,
        *,
        token: str,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        url = path if path.startswith("http://") or path.startswith("https://") else f"{self.config.api_url}{path}"
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.request(
                method,
                url,
                headers=self._headers(token),
                json=json_body,
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise GitHubAppRequestError(
                f"GitHub API {method} {path} failed with {response.status_code}: {response.text}"
            ) from exc
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def installation_id_for_repo(self, owner: str, repo: str) -> str:
        payload = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/installation",
            token=self.app_jwt(),
        )
        installation_id = payload.get("id") if payload else None
        if not installation_id:
            raise GitHubAppRequestError(f"No GitHub App installation found for {owner}/{repo}")
        return str(installation_id)

    async def installation_token(
        self,
        *,
        owner: str | None = None,
        repo: str | None = None,
    ) -> dict[str, Any]:
        installation_id = self.config.installation_id
        if not installation_id:
            owner = owner or self.config.default_owner
            repo = repo or self.config.default_repo
            if not owner or not repo:
                raise GitHubAppConfigurationError(
                    "GITHUB_APP_INSTALLATION_ID or GITHUB_DEFAULT_OWNER/GITHUB_DEFAULT_REPO is required"
                )
            installation_id = await self.installation_id_for_repo(owner, repo)

        payload = await self._request(
            "POST",
            f"/app/installations/{installation_id}/access_tokens",
            token=self.app_jwt(),
        )
        token = payload.get("token") if payload else None
        if not token:
            raise GitHubAppRequestError("GitHub App installation token response did not include a token")
        return {
            "token": token,
            "expires_at": payload.get("expires_at"),
            "installation_id": str(installation_id),
        }

    async def installation_request(
        self,
        method: str,
        path: str,
        *,
        owner: str | None = None,
        repo: str | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        token_payload = await self.installation_token(owner=owner, repo=repo)
        return await self._request(
            method,
            path,
            token=token_payload["token"],
            json_body=json_body,
        )

    async def repository_dispatch(
        self,
        *,
        owner: str,
        repo: str,
        event_type: str,
        client_payload: dict[str, Any],
    ) -> None:
        await self.installation_request(
            "POST",
            f"/repos/{owner}/{repo}/dispatches",
            owner=owner,
            repo=repo,
            json_body={
                "event_type": event_type,
                "client_payload": client_payload,
            },
        )

    async def repository_dispatch_with_token(
        self,
        *,
        token: str,
        owner: str,
        repo: str,
        event_type: str,
        client_payload: dict[str, Any],
    ) -> None:
        await self._request(
            "POST",
            f"/repos/{owner}/{repo}/dispatches",
            token=token,
            json_body={
                "event_type": event_type,
                "client_payload": client_payload,
            },
        )

    async def repositories_for_token(
        self,
        token: str,
        *,
        query: str = "",
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.get(
                f"{self.config.api_url}/user/repos",
                headers=self._headers(token),
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

    async def branches_for_token(self, token: str, owner: str, repo: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.get(
                f"{self.config.api_url}/repos/{owner}/{repo}/branches",
                headers=self._headers(token),
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

    async def create_issue_comment(
        self,
        *,
        owner: str,
        repo: str,
        issue_number: int,
        body: str,
    ) -> dict[str, Any] | None:
        return await self.installation_request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
            owner=owner,
            repo=repo,
            json_body={"body": body},
        )

    async def create_check_run(
        self,
        *,
        owner: str,
        repo: str,
        name: str,
        head_sha: str,
        status: str = "queued",
        conclusion: str | None = None,
        output: dict[str, Any] | None = None,
        external_id: str | None = None,
    ) -> dict[str, Any] | None:
        body: dict[str, Any] = {
            "name": name,
            "head_sha": head_sha,
            "status": status,
        }
        if conclusion:
            body["conclusion"] = conclusion
        if output:
            body["output"] = output
        if external_id:
            body["external_id"] = external_id
        return await self.installation_request(
            "POST",
            f"/repos/{owner}/{repo}/check-runs",
            owner=owner,
            repo=repo,
            json_body=body,
        )
