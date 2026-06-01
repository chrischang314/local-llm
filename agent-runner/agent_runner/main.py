import asyncio
import hmac
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


app = FastAPI(title="Local LLM Agent Runner")

ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RunRequest(BaseModel):
    argv: list[str] = Field(..., min_length=1, max_length=64)
    stdin: str | None = None
    cwd: str = "."
    timeout_seconds: float | None = None
    env: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class PreparedRun:
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    timeout_seconds: float
    stdin_bytes: bytes


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _csv_env(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {item.strip() for item in raw.split(",") if item.strip()}


def workspace_root() -> Path:
    root = Path(os.getenv("RUNNER_WORKSPACE", "/workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def token_required() -> bool:
    return (
        _truthy(os.getenv("RUNNER_ACCESS_REQUIRED"))
        or _truthy(os.getenv("RUNNER_AUTH_REQUIRED"))
        or _truthy(os.getenv("RUNNER_REQUIRE_TOKEN"))
        or bool(os.getenv("RUNNER_TOKEN"))
    )


def configured_token() -> str:
    return os.getenv("RUNNER_TOKEN", "")


def header_value(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def supplied_token(headers: Mapping[str, str]) -> str | None:
    explicit = header_value(headers, "x-runner-token")
    if explicit:
        return explicit

    auth = header_value(headers, "authorization")
    if not auth:
        return None
    scheme, _, value = auth.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value


async def require_runner_token(request: Request) -> None:
    expected = configured_token()
    if token_required() and not expected:
        raise HTTPException(status_code=503, detail="Runner token is required but not configured")
    if not expected:
        return

    provided = supplied_token(request.headers)
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid runner token")


def _resolve_cwd(cwd: str) -> Path:
    root = workspace_root()
    candidate = Path(cwd or ".")
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="cwd must stay inside the runner workspace") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _validate_argv(argv: list[str]) -> list[str]:
    max_arg_count = _int_env("RUNNER_MAX_ARG_COUNT", 64)
    max_arg_bytes = _int_env("RUNNER_MAX_ARG_BYTES", 8192)
    allowed = _csv_env("RUNNER_ALLOWED_COMMANDS", "python,python3,pytest,git,rg")

    if not argv or len(argv) > max_arg_count:
        raise HTTPException(status_code=400, detail=f"argv must contain 1-{max_arg_count} entries")

    cleaned: list[str] = []
    total_bytes = 0
    for arg in argv:
        if not isinstance(arg, str) or not arg or "\x00" in arg:
            raise HTTPException(status_code=400, detail="argv entries must be non-empty strings")
        total_bytes += len(arg.encode("utf-8"))
        cleaned.append(arg)

    if total_bytes > max_arg_bytes:
        raise HTTPException(status_code=400, detail=f"argv exceeds {max_arg_bytes} bytes")

    command = cleaned[0]
    if "/" in command or "\\" in command:
        raise HTTPException(status_code=400, detail="command must be a bare executable name")
    if command not in allowed:
        raise HTTPException(status_code=400, detail=f"command '{command}' is not allowed")
    return cleaned


def _timeout_seconds(requested: float | None) -> float:
    default_timeout = _float_env("RUNNER_DEFAULT_TIMEOUT_SECONDS", 30.0)
    max_timeout = _float_env("RUNNER_MAX_TIMEOUT_SECONDS", 60.0)
    timeout = default_timeout if requested is None else float(requested)
    if timeout <= 0 or timeout > max_timeout:
        raise HTTPException(status_code=400, detail=f"timeout must be between 0 and {max_timeout} seconds")
    return timeout


def _stdin_bytes(stdin: str | None) -> bytes:
    payload = (stdin or "").encode("utf-8")
    max_stdin_bytes = _int_env("RUNNER_MAX_STDIN_BYTES", 65536)
    if len(payload) > max_stdin_bytes:
        raise HTTPException(status_code=400, detail=f"stdin exceeds {max_stdin_bytes} bytes")
    return payload


def _command_env(extra_env: dict[str, str]) -> dict[str, str]:
    allowed_env = _csv_env("RUNNER_ALLOWED_ENV", "PYTHONPATH,PYTHONUNBUFFERED,NO_COLOR")
    env = {
        "PATH": os.getenv("RUNNER_COMMAND_PATH") or os.getenv("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workspace_root()),
        "TMPDIR": os.getenv("RUNNER_TMPDIR", "/tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    for key, value in extra_env.items():
        if not ENV_NAME_RE.match(key):
            raise HTTPException(status_code=400, detail=f"invalid environment variable name '{key}'")
        if key not in allowed_env:
            raise HTTPException(status_code=400, detail=f"environment variable '{key}' is not allowed")
        if len(str(value).encode("utf-8")) > 4096:
            raise HTTPException(status_code=400, detail=f"environment variable '{key}' is too large")
        env[key] = str(value)

    return env


def prepare_run(request: RunRequest) -> PreparedRun:
    return PreparedRun(
        argv=_validate_argv(request.argv),
        cwd=_resolve_cwd(request.cwd),
        env=_command_env(request.env),
        timeout_seconds=_timeout_seconds(request.timeout_seconds),
        stdin_bytes=_stdin_bytes(request.stdin),
    )


def _decode_output(payload: bytes, max_bytes: int) -> tuple[str, bool]:
    truncated = len(payload) > max_bytes
    if truncated:
        payload = payload[:max_bytes]
    return payload.decode("utf-8", errors="replace"), truncated


async def run_sandbox_command(request: RunRequest) -> dict[str, Any]:
    prepared = prepare_run(request)
    run_id = uuid.uuid4().hex
    started = time.monotonic()
    timed_out = False

    try:
        process = await asyncio.create_subprocess_exec(
            *prepared.argv,
            cwd=str(prepared.cwd),
            env=prepared.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"command '{prepared.argv[0]}' is not installed") from exc

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(prepared.stdin_bytes),
            timeout=prepared.timeout_seconds,
        )
    except asyncio.TimeoutError:
        timed_out = True
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()

    max_output_bytes = _int_env("RUNNER_MAX_OUTPUT_BYTES", 65536)
    stdout, stdout_truncated = _decode_output(stdout_bytes, max_output_bytes)
    stderr, stderr_truncated = _decode_output(stderr_bytes, max_output_bytes)
    duration_ms = int((time.monotonic() - started) * 1000)

    return {
        "id": run_id,
        "argv": prepared.argv,
        "cwd": str(prepared.cwd),
        "exit_code": process.returncode if process.returncode is not None else -1,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


@app.get("/health")
async def health():
    try:
        root = workspace_root()
        workspace_ready = root.exists() and os.access(root, os.W_OK)
    except OSError:
        root = Path(os.getenv("RUNNER_WORKSPACE", "/workspace"))
        workspace_ready = False

    requires_token = token_required()
    token_ready = (not requires_token) or bool(configured_token())
    ready = workspace_ready and token_ready
    payload = {
        "status": "ok" if ready else "misconfigured",
        "workspace": str(root),
        "workspace_writable": workspace_ready,
        "token_required": requires_token,
        "token_configured": bool(configured_token()),
        "allowed_commands": sorted(_csv_env("RUNNER_ALLOWED_COMMANDS", "python,python3,pytest,git,rg")),
        "max_timeout_seconds": _float_env("RUNNER_MAX_TIMEOUT_SECONDS", 60.0),
        "max_output_bytes": _int_env("RUNNER_MAX_OUTPUT_BYTES", 65536),
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.post("/runs")
async def create_run(request: RunRequest, _: None = Depends(require_runner_token)):
    return await run_sandbox_command(request)
