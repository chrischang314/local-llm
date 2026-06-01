import os
from typing import Any, Mapping, Sequence

import httpx


class AgentExecutorError(RuntimeError):
    """Base error for agent runner calls."""


class AgentExecutorNotConfigured(AgentExecutorError):
    """Raised when backend code tries to use the runner without a URL."""


class AgentExecutorRequestError(AgentExecutorError):
    """Raised when the runner rejects or fails a request."""


class AgentExecutorClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        configured_base_url = os.getenv("AGENT_RUNNER_URL", "") if base_url is None else base_url
        self.base_url = configured_base_url.rstrip("/")
        self.token = token if token is not None else os.getenv("AGENT_RUNNER_TOKEN", "")
        self.timeout_seconds = (
            float(os.getenv("AGENT_RUNNER_TIMEOUT_SECONDS", "90"))
            if timeout_seconds is None
            else timeout_seconds
        )
        self.transport = transport

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        cwd: str = ".",
        timeout_seconds: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self.base_url:
            raise AgentExecutorNotConfigured("AGENT_RUNNER_URL is not configured")
        if not argv:
            raise ValueError("argv must contain at least one command")

        payload: dict[str, Any] = {
            "argv": list(argv),
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "env": dict(env or {}),
        }
        if stdin is not None:
            payload["stdin"] = stdin

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/runs",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise AgentExecutorRequestError(
                f"agent runner returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AgentExecutorRequestError(f"agent runner request failed: {exc}") from exc


agent_executor_client = AgentExecutorClient()
