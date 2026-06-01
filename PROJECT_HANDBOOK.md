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
runs Docker Desktop and an Ollama container locally. Kubernetes represents the
PC with a tiny "switch" Deployment:

- `replicas: 1` means the user wants that PC worker on.
- `replicas: 0` means the user wants that PC worker off.

A scheduled PowerShell controller on the PC watches the switch Deployment and
starts or stops the matching Docker Compose service. It writes status
annotations back to Kubernetes so dashboards can distinguish desired state from
actual state.

## Routing Rules

Backend routing config lives in:

- `k8s/local-llm/ollama-backends-configmap.yaml`
- `k8s/local-llm/live-default-app-overrides.yaml`

The router should only choose a backend that is enabled, reachable, and has the
requested model installed. Optional PC workers should appear after essential
capacity for small models unless a larger GPU is intentionally preferred.

## Health Reporting

The unauthenticated `/health` route is intentionally lightweight enough for
dashboards and launchpad checks. It reports backend status, Ollama model count,
and a compact worker summary derived from the router's backend health cache.
The frontend sidebar uses that same response to show `available/enabled`
workers and active routed requests.

## Authentication

The app uses simple username/password login for a trusted LAN. The browser keeps
the bearer token in `localStorage` so the user remains signed in across tab
closes. The backend signs tokens with `JWT_SECRET` when configured; otherwise it
stores a generated signing key under the persistent app data directory
(`/app/data/jwt_secret` in containers). That generated key is runtime state and
must stay out of git.

## GitHub OAuth

GitHub authorization is service-style, not per-user developer-key entry. A Local
LLM operator configures one GitHub OAuth App through the UI. The Client ID and
Client Secret are stored in SQLite as an encrypted service config. After that,
each Local LLM user clicks **Sign in with GitHub**, is redirected to GitHub's
authorize page, and returns through `/github/oauth/callback`. The callback state
maps the browser flow back to the Local LLM user that started it, then the
backend encrypts that user's GitHub token separately.

GitHub status and sign-in routes must stay available even when agent jobs are
disabled. Agent job execution can remain behind `AGENT_JOBS_ENABLED`, but users
should still be able to connect or reconnect GitHub from the UI.

## Web Research

The local model remains the answer generator. Web research is a request-time
retrieval layer that runs only when the user or API client opts in with
`web_research: true`. The UI toggle applies to one message and then resets. The
backend searches the public web, builds a dated source brief from search
results, and inserts that brief as system context before sending the prompt to
Ollama.

The research layer is intentionally isolated in `backend/research_client.py`:
it has no database writes, no default API-key dependency, tight timeouts, source
caps, and degraded failure behavior. Result page fetching is off by default to
avoid backend-side network exposure; if operators enable
`WEB_RESEARCH_FETCH_PAGES=true`, the client rejects private, loopback,
link-local, multicast, and reserved targets and does not follow source-page
redirects. If search is disabled or unavailable, chat should still answer
normally and report the degraded status through response headers.

Because prompts may leave the LAN when Research is enabled, the UI keeps the
toggle explicit. Future provider integrations should keep secrets in runtime
configuration or ignored local stores, never in git, and should preserve mocked
unit tests for parser, timeout, and error paths.

## Agent Runner Sandbox

Agent execution is intentionally separated from the chat backend. The
`agent-runner` image exposes a token-protected internal API that runs a single
argv command in a constrained `/workspace` emptyDir and returns exit code,
stdout, stderr, timeout state, and truncation flags. The runner does not invoke a
shell and only permits commands from `RUNNER_ALLOWED_COMMANDS`.

Kubernetes owns the sandbox boundary: non-root UID/GID, no service-account token,
no privilege escalation, dropped capabilities, runtime-default seccomp,
read-only root filesystem, bounded emptyDir volumes, resource limits, and a
NetworkPolicy with no egress. Backend code should call it through
`backend/agent_executor_client.py` so auth, timeout, and error wrapping stay
centralized.

## Operational Safety

Secrets and local runtime state stay out of git. In particular, do not commit
kube tokens, SSH keys, Docker credential files, or `.docker-worker/`.
The agent-runner token is also runtime-only and should live in the
`local-llm-agent-runner-auth` Kubernetes Secret.

When adding another Windows worker, copy the CHRIS-PC-1 pattern: add a switch
Deployment, a restricted worker-controller service account, routing config, docs,
and an interactive Windows scheduled task for Docker Desktop plus the controller.
