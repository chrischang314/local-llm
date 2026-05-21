"""Isolated agent runner for Local LLM code jobs.

The backend creates a short-lived Kubernetes Job with a GitHub App installation
token. This process clones one repository, runs a bounded local-LLM tool loop,
tests the result, and reports logs/artifacts back to the backend.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE = Path("/workspace")
REPO_DIR = WORKSPACE / "repo"
MAX_ITERATIONS = 20
MAX_FILE_READ = 120_000
MAX_OBSERVATION = 20_000
MAX_WRITE = 1_000_000
MAX_COMMAND_SECONDS = 120
MAX_TEST_SECONDS = 900
MAX_IMPLEMENTER_ITERATIONS = 12
MAX_REVISION_ITERATIONS = 8
MAX_REVIEW_CYCLES = 3
MAX_REVIEW_DIFF = 120_000

SECRET_KEYS = ("GITHUB_TOKEN", "AGENT_CALLBACK_TOKEN", "AGENT_SECRET")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", "__pycache__"}


@dataclass
class AgentDecision:
    approved: bool
    summary: str
    issues: list[str]


@dataclass
class TestResult:
    supplied: bool
    passed: bool
    exit_code: int | None
    output: str


@dataclass
class QualityResult:
    satisfactory: bool
    review_only: bool
    revised: bool
    summary: str


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def read_secret(name: str, file_name: str) -> str:
    value = os.environ.pop(name, "")
    path = os.environ.pop(file_name, "")
    if value:
        return value
    if not path:
        return ""
    secret_path = Path(path)
    try:
        value = secret_path.read_text(encoding="utf-8")
        try:
            secret_path.unlink()
        except OSError:
            pass
        return value
    except OSError:
        return ""


JOB_ID = env("AGENT_JOB_ID")
REPO_FULL_NAME = env("REPO_FULL_NAME")
BASE_BRANCH = env("BASE_BRANCH", "main")
WORK_BRANCH = env("WORK_BRANCH", f"agent/{JOB_ID[:12]}")
MODEL = env("MODEL", "llama3.2")
TASK = read_secret("TASK", "TASK_FILE")
TEST_COMMAND = env("TEST_COMMAND")
INITIAL_GITHUB_TOKEN = read_secret("GITHUB_TOKEN", "GITHUB_TOKEN_FILE")
CALLBACK_TOKEN = read_secret("AGENT_CALLBACK_TOKEN", "AGENT_CALLBACK_TOKEN_FILE")
LLM_API_URL = env("LOCAL_LLM_API_URL")
CALLBACK_URL = env("LOCAL_LLM_CALLBACK_URL")


def redact(text: str) -> str:
    out = text or ""
    for value in (INITIAL_GITHUB_TOKEN, CALLBACK_TOKEN):
        if value:
            out = out.replace(value, "[REDACTED]")
    out = re.sub(r"gh[psuor]_[A-Za-z0-9_]{20,}", "[REDACTED]", out)
    out = re.sub(r"github_pat_[A-Za-z0-9_]{20,}", "[REDACTED]", out)
    out = re.sub(r"x-access-token:[^@\\s]+", "x-access-token:[REDACTED]", out, flags=re.I)
    return out


def callback(path: str, payload: dict[str, Any], timeout: float = 10.0) -> None:
    if not CALLBACK_URL or not CALLBACK_TOKEN:
        return
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{CALLBACK_URL}{path}",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Agent-Token": CALLBACK_TOKEN,
        },
    )
    try:
        urllib.request.urlopen(request, timeout=timeout).read()
    except Exception as exc:  # noqa: BLE001 - logging must not abort cleanup
        print(f"callback failed: {exc}", file=sys.stderr, flush=True)


def log(message: str, level: str = "info") -> None:
    message = redact(message)
    print(message, flush=True)
    callback("/log", {"level": level, "message": message})


def step(name: str, status: str, exit_code: int | None = None) -> None:
    payload: dict[str, Any] = {"name": name, "status": status}
    if exit_code is not None:
        payload["exit_code"] = exit_code
    callback("/step", payload)


def run(
    args: list[str] | str,
    *,
    cwd: Path | None = None,
    timeout: int = MAX_COMMAND_SECONDS,
    shell: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    for key in SECRET_KEYS:
        child_env.pop(key, None)
    child_env.setdefault("HOME", str(WORKSPACE / "home"))
    child_env.setdefault("TMPDIR", str(WORKSPACE / "tmp"))
    completed = subprocess.run(
        args,
        cwd=str(cwd or REPO_DIR),
        timeout=timeout,
        shell=shell,
        text=True,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(redact(completed.stdout[-MAX_OBSERVATION:]))
    return completed


def setup_git_auth(token: str) -> Path:
    askpass = WORKSPACE / "tmp" / "git-askpass.sh"
    askpass.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) echo x-access-token ;;\n"
        f"  *Password*) printf '%s\\n' '{token}' ;;\n"
        f"  *) printf '%s\\n' '{token}' ;;\n"
        "esac\n",
        encoding="utf-8",
        newline="\n",
    )
    askpass.chmod(0o700)
    os.environ["GIT_ASKPASS"] = str(askpass)
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    return askpass


def clear_git_auth(askpass: Path | None) -> None:
    os.environ.pop("GIT_ASKPASS", None)
    os.environ.pop("GIT_TERMINAL_PROMPT", None)
    if askpass:
        try:
            askpass.unlink()
        except FileNotFoundError:
            pass


def run_with_git_token(token: str, args: list[str], *, timeout: int = 300, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    askpass = setup_git_auth(token)
    try:
        if args and args[0] == "git":
            args = _safe_git_args(args)
        return run(args, cwd=cwd, timeout=timeout, check=check)
    finally:
        clear_git_auth(askpass)


def run_git_safe(args: list[str], *, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(_safe_git_args(args), timeout=timeout, check=check)


def _safe_git_args(args: list[str]) -> list[str]:
    if not args or args[0] != "git":
        return args
    return ["git", "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=", *args[1:]]


def repo_api_url(path: str) -> str:
    return f"https://api.github.com/repos/{REPO_FULL_NAME}{path}"


def github_json(token: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        repo_api_url(path),
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def setup_repo() -> None:
    global INITIAL_GITHUB_TOKEN
    (WORKSPACE / "home").mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "tmp").mkdir(parents=True, exist_ok=True)
    token = INITIAL_GITHUB_TOKEN
    INITIAL_GITHUB_TOKEN = ""
    if not token:
        raise RuntimeError("missing initial GitHub installation token")
    clone_url = f"https://github.com/{REPO_FULL_NAME}.git"
    log(f"Cloning {REPO_FULL_NAME}@{BASE_BRANCH}.")
    run_with_git_token(
        token,
        ["git", "clone", "--branch", BASE_BRANCH, "--depth", "1", clone_url, str(REPO_DIR)],
        cwd=WORKSPACE,
        timeout=300,
        check=True,
    )
    run(["git", "config", "user.name", "Local LLM Agent"], check=True)
    run(["git", "config", "user.email", "local-llm-agent@users.noreply.github.com"], check=True)
    run(["git", "switch", "-c", WORK_BRANCH], check=True)


def safe_path(raw_path: str) -> Path:
    if not raw_path or raw_path.strip() in {".", "./"}:
        return REPO_DIR
    target = (REPO_DIR / raw_path).resolve()
    repo = REPO_DIR.resolve()
    if target == repo:
        return target
    if repo not in target.parents:
        raise ValueError("path escapes workspace")
    rel = target.relative_to(repo)
    if rel.parts and rel.parts[0] == ".git":
        raise ValueError("writes to .git are not allowed")
    return target


def trim(text: str, limit: int = MAX_OBSERVATION) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]\n"


def list_files(path: str = ".") -> str:
    root = safe_path(path)
    if root.is_file():
        return str(root.relative_to(REPO_DIR))
    files: list[str] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            full = Path(current) / name
            try:
                files.append(str(full.relative_to(REPO_DIR)))
            except ValueError:
                continue
            if len(files) >= 400:
                files.append("[file list truncated]")
                return "\n".join(files)
    return "\n".join(sorted(files)) or "[empty]"


def read_file(path: str) -> str:
    target = safe_path(path)
    if not target.is_file():
        return "[not a file]"
    return trim(target.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_READ])


def search_repo(query: str) -> str:
    if not query.strip():
        return "[empty query]"
    result = run(["rg", "--line-number", "--hidden", "--glob", "!/.git/**", query], timeout=45)
    return trim(result.stdout or "[no matches]")


def write_file(path: str, content: str) -> str:
    target = safe_path(path)
    if len(content) > MAX_WRITE:
        return "[refused: content too large]"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    return f"[wrote {path}]"


def run_shell(command: str) -> str:
    return "[refused: arbitrary shell is disabled in the agent tool loop]"


def diff() -> str:
    result = run(["git", "diff", "--no-ext-diff", "--binary"], timeout=45)
    return trim(result.stdout or "[no diff]", 200_000)


def extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


def parse_action(raw: str) -> dict[str, Any]:
    parsed = extract_json_object(raw)
    if "action" not in parsed:
        raise ValueError("model response must be a JSON object with an action")
    return parsed


def parse_decision(raw: str) -> AgentDecision:
    parsed = extract_json_object(raw)
    status = str(parsed.get("status", "")).strip().lower()
    summary = str(parsed.get("summary") or "").strip() or "No summary supplied."
    raw_issues = parsed.get("issues") or []
    if isinstance(raw_issues, str):
        issues = [raw_issues]
    elif isinstance(raw_issues, list):
        issues = [str(item) for item in raw_issues if str(item).strip()]
    else:
        issues = []
    approved = status in {"approved", "pass", "passed", "satisfactory"}
    if not approved and not issues:
        issues = [summary]
    return AgentDecision(approved=approved, summary=summary, issues=issues)


def call_llm(messages: list[dict[str, str]]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 1800,
    }
    request = urllib.request.Request(
        LLM_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def request_github_token() -> str:
    request = urllib.request.Request(
        f"{CALLBACK_URL}/github-token",
        data=b"{}",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Agent-Token": CALLBACK_TOKEN,
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    token = payload.get("token")
    if not token:
        raise RuntimeError("backend did not return a GitHub token")
    return token


def base_task_context() -> str:
    return (
        f"Repository: {REPO_FULL_NAME}\n"
        f"Base branch: {BASE_BRANCH}\n"
        f"Task:\n{TASK}\n\n"
        f"Test command: {TEST_COMMAND or '[none supplied]'}"
    )


def run_tool_loop(role: str, assignment: str, *, max_iterations: int) -> str:
    system = (
        f"You are the {role} subagent for a Local LLM code job running inside an isolated Kubernetes Job. "
        "Respond with exactly one JSON object per turn. Available actions are: "
        "list_files {path}, read_file {path}, search {query}, write_file {path, content}, "
        "inspect_diff {}, finish {summary}. "
        "Only modify files that are necessary for the task. Prefer reading before writing. "
        "Do not write outside the repository. Shell commands are not available. "
        "Leave concise, maintainable changes that another reviewer can inspect."
    )
    user = (
        f"{base_task_context()}\n\n"
        f"Assignment for this subagent:\n{assignment}\n\n"
        "Begin by inspecting the repository or current diff before editing."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    final_summary = "Tool loop reached iteration limit."

    for iteration in range(1, max_iterations + 1):
        log(f"{role.capitalize()} subagent iteration {iteration}/{max_iterations}.")
        try:
            action = parse_action(call_llm(messages))
        except Exception as exc:  # noqa: BLE001
            observation = f"[invalid model action: {exc}]"
            messages.append({"role": "assistant", "content": json.dumps({"action": "invalid"})})
            messages.append({"role": "user", "content": observation})
            continue

        name = str(action.get("action", "")).strip()
        args = action.get("args") or {}
        messages.append({"role": "assistant", "content": json.dumps(action)})

        try:
            if name == "list_files":
                observation = list_files(str(args.get("path", ".")))
            elif name == "read_file":
                observation = read_file(str(args.get("path", "")))
            elif name == "search":
                observation = search_repo(str(args.get("query", "")))
            elif name == "write_file":
                observation = write_file(str(args.get("path", "")), str(args.get("content", "")))
            elif name == "run_shell":
                observation = run_shell(str(args.get("command", "")))
            elif name == "inspect_diff":
                observation = diff()
            elif name == "finish":
                final_summary = str(args.get("summary") or "Finished.")
                log(final_summary)
                return final_summary
            else:
                observation = f"[unknown action: {name}]"
        except Exception as exc:  # noqa: BLE001
            observation = f"[tool error: {exc}]"

        messages.append({"role": "user", "content": trim(observation)})

    log(final_summary, level="warning")
    return final_summary


def reviewer_agent(cycle: int) -> AgentDecision:
    current_diff = diff()
    if current_diff.strip() == "[no diff]":
        return AgentDecision(False, "No file changes are present.", ["No diff to review."])
    system = (
        "You are a strict reviewer subagent for a Local LLM code job. "
        "Review the current diff for correctness, maintainability, focused scope, safety, and testability. "
        "Return exactly one JSON object with fields: status ('approved' or 'changes_requested'), "
        "summary, and issues (array of strings). Approve only when the diff appears ready for tests."
    )
    user = (
        f"{base_task_context()}\n\n"
        f"Review cycle: {cycle}/{MAX_REVIEW_CYCLES}\n\n"
        f"Current diff:\n{trim(current_diff, MAX_REVIEW_DIFF)}"
    )
    try:
        decision = parse_decision(call_llm([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]))
    except Exception as exc:  # noqa: BLE001
        return AgentDecision(False, "Reviewer response could not be parsed.", [str(exc)])
    return decision


def tester_agent(result: TestResult) -> AgentDecision:
    if not result.supplied:
        return AgentDecision(
            False,
            "No test command was supplied, so direct base-branch push is not allowed.",
            ["Supply a test command for automatic direct push."],
        )
    if result.passed:
        return AgentDecision(True, "Configured tests passed.", [])
    return AgentDecision(
        False,
        f"Configured tests failed with exit code {result.exit_code}.",
        [trim(result.output, 8_000)],
    )


def revision_assignment(source: str, feedback: AgentDecision) -> str:
    issues = "\n".join(f"- {issue}" for issue in feedback.issues) or "- No issue details supplied."
    return (
        f"The {source} subagent requested changes. Revise the repository to address this feedback, "
        "then inspect the diff and finish when the revision is ready for another review.\n\n"
        f"Summary: {feedback.summary}\n"
        f"Issues:\n{issues}"
    )


def run_quality_loop() -> QualityResult:
    revised = False
    for cycle in range(1, MAX_REVIEW_CYCLES + 1):
        log(f"Quality loop cycle {cycle}/{MAX_REVIEW_CYCLES}.")

        step("review", "in_progress")
        review = reviewer_agent(cycle)
        if not review.approved:
            step("review", "failed")
            log(f"Reviewer requested changes: {review.summary}", level="warning")
            for issue in review.issues:
                log(f"Reviewer issue: {issue}", level="warning")
            if cycle == MAX_REVIEW_CYCLES:
                return QualityResult(False, False, revised, "Reviewer did not approve the final diff.")
            step("revise", "in_progress")
            run_tool_loop("revision", revision_assignment("reviewer", review), max_iterations=MAX_REVISION_ITERATIONS)
            step("revise", "succeeded")
            revised = True
            continue
        step("review", "succeeded")
        log(f"Reviewer approved: {review.summary}")

        step("test", "in_progress" if TEST_COMMAND.strip() else "skipped")
        test_result = run_tests()
        test_decision = tester_agent(test_result)
        if test_decision.approved:
            step("test", "succeeded", test_result.exit_code)
            return QualityResult(True, False, revised, "Reviewer approved and configured tests passed.")

        if not test_result.supplied:
            step("test", "skipped")
            return QualityResult(True, True, revised, "Reviewer approved, but no test command was supplied.")

        step("test", "failed", test_result.exit_code)
        log(f"Testing agent requested changes: {test_decision.summary}", level="warning")
        if cycle == MAX_REVIEW_CYCLES:
            return QualityResult(False, False, revised, "Tests did not pass after the final revision cycle.")
        step("revise", "in_progress")
        run_tool_loop("test-fix", revision_assignment("testing", test_decision), max_iterations=MAX_REVISION_ITERATIONS)
        step("revise", "succeeded")
        revised = True

    return QualityResult(False, False, revised, "Quality loop ended without approval.")


def commit_if_needed() -> str | None:
    status = run(["git", "status", "--porcelain"], timeout=30, check=True).stdout.strip()
    if not status:
        return None
    run(["git", "add", "-A"], check=True)
    run(["git", "commit", "-m", f"Agent job {JOB_ID[:12]}"], timeout=120, check=True)
    sha = run(["git", "rev-parse", "HEAD"], check=True).stdout.strip()
    return sha


def branch_is_protected(token: str) -> bool:
    encoded = urllib.parse.quote(BASE_BRANCH, safe="")
    try:
        payload = github_json(token, "GET", f"/branches/{encoded}")
        return bool(payload.get("protected"))
    except urllib.error.HTTPError as exc:
        log(f"Could not inspect branch protection: HTTP {exc.code}", level="warning")
        return True


def create_pr(token: str) -> str | None:
    try:
        payload = github_json(
            token,
            "POST",
            "/pulls",
            {
                "title": f"Agent job {JOB_ID[:12]}",
                "head": WORK_BRANCH,
                "base": BASE_BRANCH,
                "body": f"Automated Local LLM agent job `{JOB_ID}`.\n\nTask:\n{TASK}",
            },
        )
        return payload.get("html_url")
    except urllib.error.HTTPError as exc:
        log(f"PR creation failed: HTTP {exc.code} {redact(exc.read().decode('utf-8', 'replace'))}", level="warning")
        return None


def push_work_branch(token: str) -> None:
    log(f"Pushing review branch {WORK_BRANCH}.")
    run_with_git_token(token, ["git", "push", "-u", "origin", f"HEAD:{WORK_BRANCH}"], timeout=300, check=True)


def run_tests() -> TestResult:
    if not TEST_COMMAND.strip():
        log("No test command supplied; direct push to base branch is disabled.", level="warning")
        return TestResult(supplied=False, passed=False, exit_code=None, output="No test command supplied.")
    log(f"Running tests: {TEST_COMMAND}")
    result = run(TEST_COMMAND, shell=True, timeout=MAX_TEST_SECONDS)
    output = redact(result.stdout)
    log(trim(f"Test exit={result.returncode}\n{output}", 200_000), level="info" if result.returncode == 0 else "error")
    return TestResult(
        supplied=True,
        passed=result.returncode == 0,
        exit_code=result.returncode,
        output=output,
    )


def reset_git_config() -> None:
    git_config = REPO_DIR / ".git" / "config"
    if not git_config.is_file():
        raise RuntimeError("repository git config is missing")
    safe_url = f"https://github.com/{REPO_FULL_NAME}.git"
    git_config.write_text(
        "[core]\n"
        "\trepositoryformatversion = 0\n"
        "\tfilemode = true\n"
        "\tbare = false\n"
        "\tlogallrefupdates = true\n"
        "[remote \"origin\"]\n"
        f"\turl = {safe_url}\n"
        "\tfetch = +refs/heads/*:refs/remotes/origin/*\n",
        encoding="utf-8",
        newline="\n",
    )


def finish(status: str, **extra: Any) -> None:
    payload = {"status": status, **extra}
    callback("/complete", payload, timeout=20.0)


def main() -> int:
    start = time.time()
    try:
        missing = [
            name for name, value in {
                "AGENT_JOB_ID": JOB_ID,
                "REPO_FULL_NAME": REPO_FULL_NAME,
                "GITHUB_TOKEN": INITIAL_GITHUB_TOKEN,
                "AGENT_CALLBACK_TOKEN": CALLBACK_TOKEN,
                "LOCAL_LLM_API_URL": LLM_API_URL,
                "LOCAL_LLM_CALLBACK_URL": CALLBACK_URL,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"missing required environment variables: {', '.join(missing)}")

        step("clone", "in_progress")
        setup_repo()
        step("clone", "succeeded")

        step("implement", "in_progress")
        run_tool_loop(
            "implementation",
            "Create the initial code change for the requested task. Include docs or tests when they are necessary.",
            max_iterations=MAX_IMPLEMENTER_ITERATIONS,
        )
        step("implement", "succeeded")

        quality = run_quality_loop()
        if not quality.revised:
            step("revise", "skipped")

        current_diff = diff()
        if not quality.satisfactory:
            step("push", "skipped")
            finish("failed", diff=current_diff, error_summary=quality.summary)
            return 1

        commit_sha = commit_if_needed()
        if not commit_sha:
            step("push", "skipped")
            finish("failed", diff="", error_summary="Agent finished without making file changes.")
            return 1

        if quality.review_only:
            step("push", "in_progress")
            reset_git_config()
            push_token = request_github_token()
            push_work_branch(push_token)
            pr_url = create_pr(push_token)
            step("push", "succeeded")
            finish("needs_review", diff=current_diff, commit_sha=commit_sha, pr_url=pr_url, error_summary="No test command supplied; base branch was not updated.")
            return 0

        step("push", "in_progress")
        log("Rebasing onto the latest base branch before push.")
        reset_git_config()
        push_token = request_github_token()
        run_with_git_token(push_token, ["git", "fetch", "origin", BASE_BRANCH], timeout=300, check=True)
        run_git_safe(["git", "rebase", f"origin/{BASE_BRANCH}"], timeout=300, check=True)

        commit_sha = run(["git", "rev-parse", "HEAD"], check=True).stdout.strip()
        if branch_is_protected(push_token):
            log(f"{BASE_BRANCH} is protected; creating a review PR instead of direct push.", level="warning")
            push_work_branch(push_token)
            pr_url = create_pr(push_token)
            step("push", "succeeded")
            finish("needs_review", diff=current_diff, commit_sha=commit_sha, pr_url=pr_url, error_summary="Base branch is protected.")
            return 0

        try:
            log(f"Pushing tested commit directly to {BASE_BRANCH}.")
            run_with_git_token(push_token, ["git", "push", "origin", f"HEAD:{BASE_BRANCH}"], timeout=300, check=True)
            step("push", "succeeded")
            finish("succeeded", diff=current_diff, commit_sha=commit_sha)
            return 0
        except Exception as exc:  # noqa: BLE001
            log(f"Direct push failed, creating PR instead: {exc}", level="warning")
            push_work_branch(push_token)
            pr_url = create_pr(push_token)
            step("push", "succeeded")
            finish("needs_review", diff=current_diff, commit_sha=commit_sha, pr_url=pr_url, error_summary="Direct push failed or branch diverged.")
            return 0
    except Exception as exc:  # noqa: BLE001
        error = redact(str(exc))
        log(error, level="error")
        step("push", "failed")
        finish("failed", diff=diff() if REPO_DIR.exists() else "", error_summary=error)
        return 1
    finally:
        log(f"Runner finished in {int(time.time() - start)}s.")


if __name__ == "__main__":
    raise SystemExit(main())
