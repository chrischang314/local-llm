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

## Agentic Code Mode

The main workspace has separate Chat and Code modes. Chat remains a
conversational UI over routed Ollama workers; Code mode is an authenticated
operational workflow that prompts a repository change, connects to GitHub
through a GitHub App installation, and executes inside a dedicated Kubernetes
sandbox namespace.

The backend stores GitHub installation records, queued jobs, ordered steps, logs,
and diff artifacts in SQLite. The frontend talks to `/github/*` for installation
status, repository listing, and branch lookup, then `/agent/*` for create/list,
detail, SSE events, cancellation, and diff retrieval.

The intended authorization path is a GitHub App installation. A temporary
`GITHUB_BYPASS_TOKEN` escape hatch exists only for controlled LAN smoke tests
against disposable repositories before the GitHub App is configured. It should
not replace installation tokens for normal use.

The execution path is:

1. User chooses repository, branch, model, task, and optional test command.
2. Backend mints a short-lived GitHub App installation token and creates a
   Kubernetes Job in `local-llm-sandbox`.
3. Runner clones the repository and creates an `agent/<job-id>` branch.
4. An implementation subagent uses the local OpenAI-compatible API for a bounded
   read/search/write/inspect-diff tool loop.
5. A reviewer subagent inspects the diff for correctness, maintainability,
   focused scope, safety, and testability.
6. A testing agent runs the configured test command. Reviewer or test failures
   are sent to a revision subagent, and the review/test cycle repeats up to
   three times before the job fails.
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
kube tokens, SSH keys, Docker credential files, or `.docker-worker/`.

When adding another Windows worker, copy the CHRIS-PC-1 pattern: add a switch
Deployment, a restricted worker-controller service account, routing config, docs,
and an interactive Windows scheduled task for Docker Desktop plus the controller.
