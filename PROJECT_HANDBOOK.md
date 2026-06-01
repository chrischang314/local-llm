# Local LLM Project Handbook

## Design Overview

The Local LLM app separates model routing from worker ownership. The frontend
lets users choose a model; the backend decides which reachable Ollama backend
should serve it.

Essential always-on capacity lives on stable machines such as the Mac Mini.
Personal Windows PCs are optional capacity: they can add GPU throughput when
available, but the system must keep working when they are off, gaming, or
rebooting.

## Optional Windows Workers

Windows PCs are not joined as native K3s nodes in this setup. Instead, each PC
runs Ollama locally and exposes it over the LAN. A PC can use either the older
Docker Desktop container path or the native Windows Ollama launcher, depending
on what is durable on that host. Kubernetes represents the PC with a tiny
"switch" Deployment:

- `replicas: 1` means the user wants that PC worker on.
- `replicas: 0` means the user wants that PC worker off.

A scheduled PowerShell controller on the PC watches the switch Deployment and
starts or stops the matching local runtime. Docker workers start a Docker
Compose service. Native workers start a scheduled task that launches
`ollama.exe serve` as a detached process. The controller writes status
annotations back to Kubernetes so dashboards can distinguish desired state from
actual state.

## Routing Rules

Backend routing config lives in:

- `k8s/local-llm/ollama-backends-configmap.yaml`
- `k8s/local-llm/live-default-app-overrides.yaml`

The router should only choose a backend that is enabled, reachable, and has the
requested model installed. Optional PC workers should appear after essential
capacity for small models unless a larger GPU is intentionally preferred.

Assistant messages persist the route chosen at generation time: selected model,
backend name, and whether the model was resident or loaded on demand. The UI
shows that label above each assistant reply so routing behavior is visible
without opening logs.

## Health Reporting

The unauthenticated `/health` route is intentionally lightweight enough for
dashboards and launchpad checks. It reports backend status, Ollama model count,
and a compact worker summary derived from the router's backend health cache.
The frontend sidebar uses that same response to show `available/enabled`
workers and active routed requests. Worker readiness warning/error severity
changes the compact sidebar indicator even when Ollama itself is still up.
The health summary also overlays Kubernetes optional-worker switch state when it
is available. A PC worker whose switch Deployment is scaled to `0` is counted in
the total inventory but treated as intentionally disabled, so turning a PC off
for gaming or maintenance does not create a false degraded-capacity warning.

The authenticated `/workers` route is the operational view. It combines router
health with Kubernetes switch annotations and returns a readiness summary. That
summary separates capacity from repair clues: unavailable enabled workers,
desired/actual switch drift, stale controller heartbeats, and worker-control
RBAC or service-account failures each become explicit issues with a suggested
next check. The Settings > Workers panel renders those issues above the switch
buttons so a user can see whether the problem is routing, the PC runtime, or the
controller loop before opening Kubernetes logs.

## Authentication

The app uses simple username/password login for a trusted LAN. The browser keeps
the bearer token in `localStorage` so the user remains signed in across tab
closes. The backend signs tokens with `JWT_SECRET` when configured; otherwise it
stores a generated signing key under the persistent app data directory
(`/app/data/jwt_secret` in containers). That generated key is runtime state and
must stay out of git.

## Conversation Portability

Conversation history belongs to the signed-in local user. The export route uses
the same ownership check as message loading, then serializes the conversation
metadata, model settings, optional system prompt, and ordered messages. Markdown
is the default for human-readable notes; JSON is available for scripts or later
import tooling.

## Web Research

Research is intentionally opt-in per message. The frontend toggle marks only the
next chat request with `web_research: true`, then resets before later prompts
can leave the LAN. The backend fetches public search results through
`backend/research_client.py`, rejects private/link-local/reserved result targets,
and injects a dated system context before the user prompt. Source URLs are
appended to the model answer so the saved conversation remains easy to audit.

Keep research providers no-secret by default. If a provider later needs API
keys, tokens, cookies, or browser state, store them outside git and expose only
non-secret status through `/research/status`.

## Frontend Asset Reliability

The LAN UI should not require public CDNs during normal page load. Browser
libraries for Markdown rendering, HTML sanitization, syntax highlighting, and
icons are pinned in `package-lock.json`, copied into `frontend/vendor/`, and
served by the same nginx frontend image as the application HTML. This removes
runtime drift from URLs such as `lucide@latest` and keeps the chat and docs
pages usable when internet access is unavailable but the LAN is healthy.

Assistant Markdown is rendered as sanitized HTML and then enhanced in
`frontend/app.js` before it is shown in the chat. Keep the enhancement pass
responsible for viewport-sensitive markup, such as wrapping wide tables and
normalizing media/link behavior, while CSS owns sizing and overflow rules.

When changing a browser library version, update the pinned npm dependency,
refresh `frontend/vendor/`, and run the static frontend test that checks both
HTML pages for CDN references and missing vendored assets.

## Agentic Code Mode

The main workspace has separate Chat and Code modes. Chat remains a
conversational UI over routed Ollama workers; Code mode is an authenticated
operational workflow that prompts a repository change, connects to GitHub
through a site-configured GitHub OAuth redirect, and executes inside a dedicated
Kubernetes sandbox namespace.

The backend stores GitHub installation records, queued jobs, ordered steps, logs,
and diff artifacts in SQLite. The frontend talks to `/github/*` for installation
status, repository listing, and branch lookup, then `/agent/*` for create/list,
detail, SSE events, cancellation, and diff retrieval.

The intended authorization path is service-configured GitHub OAuth. A Local LLM
operator enters the OAuth App Client ID and Client Secret once in the collapsed
Settings integration setup panel; the backend stores them encrypted in SQLite.
After that, each Local LLM user clicks the Sign in with GitHub button,
authorizes on github.com, and returns through the callback with their own
encrypted GitHub token. The legacy GitHub App installation path is still present
for deployments that prefer installation tokens. A temporary `GITHUB_BYPASS_TOKEN`
escape hatch exists only for controlled LAN smoke tests against disposable
repositories and should not replace the OAuth flow for normal use.

The execution path is:

1. User chooses repository, branch, model, task, and optional test command.
2. Backend obtains a repository access token from the connected OAuth account
   or legacy GitHub App installation and creates a Kubernetes Job in
   `local-llm-sandbox`.
3. Runner clones the repository and creates an `agent/<job-id>` branch.
4. An implementation subagent uses the local OpenAI-compatible API for a bounded
   read/search/write/run-shell/inspect-diff tool loop inside the isolated
   repository workspace.
5. A reviewer subagent inspects the diff for correctness, maintainability,
   focused scope, safety, and testability.
6. A testing agent runs the configured test command. If review finds no diff,
   the runner first executes the configured test command once and passes that
   failure output to the revision subagent. Reviewer or test failures are sent
   to a revision subagent, and the review/test cycle repeats up to three times
   before the job fails.
7. After the reviewer and testing gates are satisfied, the runner commits the
   final changes, requests a fresh push token, rebases, and pushes only after
   tests pass.
8. Branch protection, divergence, or a missing test command prevents direct base
   branch updates. In those cases the runner pushes the agent branch and opens a
   PR, and the job ends as `needs_review`.

The sandbox namespace has restricted Pod Security labels, quota/limit defaults,
default-deny networking, no service-account token in execution pods, no hostPath
mounts, no Docker socket, a read-only container root filesystem, dropped Linux
capabilities, seccomp `RuntimeDefault`, and bounded CPU, memory, and ephemeral
storage. Runtime secrets are copied by a restricted init container into an
in-memory `emptyDir`; the runner container receives only file paths, reads and
unlinks the files at startup, and never starts with raw GitHub or callback tokens
in its environment. The runner callback token is per job, and Kubernetes Secrets
are patched with Job owner references after launch so normal TTL cleanup can
garbage-collect them. Keep `AGENT_JOBS_ENABLED=false` until NetworkPolicy
enforcement and canary jobs are verified live.

Residual risk: standard Kubernetes NetworkPolicy is IP/CIDR based, not
GitHub-domain aware. The current policy allows public TCP 443 while blocking
private/LAN ranges so GitHub operations can work. Add a controlled egress proxy
before enabling jobs on an untrusted network or for sensitive private code.

## Operational Safety

Secrets and local runtime state stay out of git. In particular, do not commit
kube tokens, SSH keys, Docker credential files, `.docker-worker/`, or files
copied back from `C:\ProgramData\LocalLlmWorker`.

When adding another Windows worker, copy the pattern that matches the host. For
native Ollama, copy the current CHRIS-PC-1 pattern: add a switch Deployment, a
restricted worker-controller service account, routing config, docs, a native
Ollama launcher task, and a controller task. Use the Docker Desktop pattern only
for machines where Docker Desktop is already reliable without manual desktop
repair.
