# Local LLM Project Handbook

## Design Overview

The Local LLM app separates model routing from worker ownership. The frontend
lets users choose a model; the backend decides which reachable Ollama backend
should serve it.

Essential always-on capacity lives on stable machines such as the Mac Mini.
Personal Windows PCs are optional capacity: they can add GPU throughput when
available, but the system must keep working when they are off, gaming, or
rebooting.

## Frontend Packaging

The LAN frontend is intentionally static and self-contained. Runtime libraries
for Markdown rendering, HTML sanitization, syntax highlighting, and icons are
vendored under `frontend/vendor/` and loaded from same-origin paths. This keeps
the chat UI, API docs, Markdown rendering, code highlighting, and Lucide icons
available without public CDN access.

When changing those libraries, update the pinned filename, the HTML reference,
and the frontend static test together. Do not reintroduce unpinned CDN URLs or
`@latest` assets in runtime HTML.

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

The app uses simple username/password login for a trusted LAN, backed by the
shared projects.lan server-side SSO database. `SHARED_AUTH_DB` points at the
shared SQLite file containing users and `auth_sessions`. Login/register set the
`projects_lan_session` cookie as `HttpOnly`, `SameSite=Lax`, and `Path=/`; the
browser does not keep auth authority in `localStorage`.

The `local-llm` namespace deployment binds `shared-auth-nfs` through a static
PV to the same NFS auth DB used by the default deployment. A separate dynamic
claim would create a different SSO database and break cross-app sessions.

Local LLM still keeps app-local user rows for conversation ownership. On each
authenticated request, the shared cookie identifies the shared user, and the
backend ensures a matching local ownership row. Existing legacy local users can
seed the shared auth DB on first successful login with their old password.

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
Optional page-body excerpt fetching is deliberately narrower than citation
collection: hostname results stay as sources, but excerpt fetches are limited to
literal public IP URLs to avoid DNS-rebinding and internal-network SSRF paths.

Keep research providers no-secret by default. If a provider later needs API
keys, tokens, cookies, or browser state, store them outside git and expose only
non-secret status through `/research/status`.

## Frontend Asset Reliability

The LAN UI should not require public CDNs during normal page load. Browser
libraries for Markdown rendering, HTML sanitization, syntax highlighting, and
icons are pinned in `package-lock.json`, listed in
`scripts/frontend-vendor-assets.json`, copied into `frontend/vendor/`, and served
by the same nginx frontend image as the application HTML. This removes runtime
drift from URLs such as `lucide@latest` and keeps the chat and docs pages usable
when internet access is unavailable but the LAN is healthy.

Assistant Markdown is rendered as sanitized HTML and then enhanced in
`frontend/app.js` before it is shown in the chat. Keep the enhancement pass
responsible for viewport-sensitive markup, such as wrapping wide tables and
normalizing media/link behavior, while CSS owns sizing and overflow rules.

When changing a browser library version, update the pinned npm dependency,
update the vendor manifest, refresh `frontend/vendor/`, and run the static
frontend test that checks both HTML pages for CDN references, package-pin drift,
and missing vendored assets.

## Chat-Only Boundary

Local LLM is intentionally a routed Ollama chat app. It owns conversation
history, model settings, model route labels, conversation export, optional
web-research augmentation, and worker routing for chat inference.

Coding automation, repository mutation, browser or desktop execution, external
app sign-in, delegated workflows, and command runners are out of scope here and
belong in Local Agent.

Model pull/delete and worker on/off controls are operator-gated chat
infrastructure controls. They should not grow into repository, browser,
desktop, or delegated automation workflows.

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
