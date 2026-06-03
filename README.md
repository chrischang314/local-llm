# Local LLM

Local LLM provides the homelab Ollama frontend/backend and Kubernetes routing
configuration for local model workers.

## Frontend Assets And API Docs

The frontend serves Marked, DOMPurify, Highlight.js, and Lucide from
`frontend/vendor/` instead of loading public CDNs at runtime. This keeps the LAN
UI usable when internet access or CDN availability is unreliable and avoids
unpinned `latest` browser assets.

The API docs use `http://localllm.lan/v1` as the primary OpenAI-compatible LAN
base URL. `http://localhost:8001/v1` remains the Docker Compose and local
development alternate.

## Kubernetes Workers

Personal Windows PCs are integrated as optional external Ollama workers rather
than native K3s nodes. Kubernetes owns a tiny switch Deployment per PC under the
`local-llm` namespace; scaling that Deployment to `1` means "turn this PC worker
on", and scaling it to `0` means "turn it off".

Current external workers:

- `chris-pc-2`: `http://192.168.4.24:11434`, RTX 5070, large tier,
  Docker Desktop worker.
- `chris-pc-1`: `http://192.168.4.27:11434`, RTX 4060, medium tier,
  native Windows Ollama worker.

See `docs/k8s-personal-workers.md` for setup, dashboard switch behavior, and
worker controller commands.

## Common Commands

```powershell
# Apply live default-namespace routing overrides.
.\scripts\deploy-live-default-app-overrides.ps1

# Turn the Docker Desktop worker on or off locally.
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-mode.ps1 -Mode on
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-mode.ps1 -Mode off

# Run one Docker worker-controller reconciliation.
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-controller.ps1 -Once

# Run one native Ollama worker-controller reconciliation.
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-native-worker-controller.ps1 -Once

# Refresh pinned same-origin frontend vendor assets after npm dependency updates.
npm install
npm run vendor:frontend
```

The Docker worker scripts use `.docker-worker/config.json` for unattended Docker
calls. The native worker installer writes runtime files under
`C:\ProgramData\LocalLlmWorker`. Both locations are local machine state and must
stay out of git.

## Frontend Assets And API Docs

The chat page and API docs serve Marked, DOMPurify, Highlight.js, and Lucide
from `frontend/vendor/` instead of runtime CDN URLs, so the LAN UI keeps
rendering when public asset CDNs are unreachable. Dependency versions are pinned
in `package-lock.json` and listed in `scripts/frontend-vendor-assets.json`;
after changing them, run `npm install` and `npm run vendor:frontend`, then
commit only the manifest and refreshed files under `frontend/vendor/`, not
`node_modules/`. The frontend static test checks that the manifest, package
pins, HTML entrypoints, and vendored files stay in sync.

The API docs use `http://localllm.lan/v1` as the primary OpenAI-compatible base
URL for the LAN deployment. `http://localhost:8001/v1` remains documented as
the Docker Compose and local development fallback.

## Chat Rendering

Assistant Markdown is sanitized before display and then enhanced for responsive
chat use. Wide tables render in a horizontal scroll region inside the message
bubble instead of forcing the mobile viewport wider. Code blocks, images, video,
audio controls, long links, and long cell text are constrained to the bubble so
saved and live-streamed replies remain readable on phones and desktop browsers.

## Web Research

Chat can optionally add live web research before calling Ollama. Turn on the
Research button in the composer for the next question that needs current facts.
When enabled, the backend searches the web, adds a dated research brief to the
model context, and appends a `Sources` footer with full URLs after the model
answer. The toggle resets after the message sends so later prompts stay fully
local unless Research is turned on again.

The OpenAI-compatible API accepts the same opt-in fields:

```json
{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "What changed in Kubernetes this week?"}],
  "web_research": true,
  "research_query": "optional explicit search query"
}
```

Useful environment variables:

- `WEB_RESEARCH_ENABLED`: set to `false` to disable the feature server-side.
- `WEB_RESEARCH_PROVIDER`: currently `mojeek` by default; `duckduckgo` is also
  available but may return challenge pages on some networks.
- `WEB_RESEARCH_MAX_RESULTS`: number of sources to read, default `3`.
- `WEB_RESEARCH_TIMEOUT_SECONDS`: network timeout per research request.
- `WEB_RESEARCH_FETCH_PAGES`: set to `true` to fetch result page excerpts after
  private/link-local/reserved targets are rejected. Default is `false`; search
  snippets and source URLs are still used. For SSRF protection, page-body fetches
  are limited to literal public IP URLs; normal hostname results are kept as
  citations but are not fetched for excerpts.
- `WEB_RESEARCH_MAX_CONTEXT_CHARS`: prompt budget for retrieved context.

`GET /research/status` returns only non-secret feature status for the signed-in
UI. Do not add API keys, cookies, or browser storage to git if another research
provider is added later.

## Health Status

`GET /health` returns the backend state, Ollama model count, and a compact
worker summary:

- `workers.enabled`: configured workers that are allowed to receive traffic.
- `workers.available`: enabled workers that answered their Ollama health check.
- `workers.unavailable`: enabled workers that are not currently reachable.
- `workers.busy`: requests currently routed through workers.
- `workers.loaded_model_count`: resident models across all configured workers.
- `workers.readiness`: a compact state, severity, and summary for launchpad or
  sidebar status.
- `workers.readiness.issue_count`: non-sensitive count of readiness issues
  behind the compact summary.

The chat sidebar uses the same endpoint so the at-a-glance status shows both
model count and the worker readiness severity/summary. If Ollama is up but
enabled worker capacity is degraded, the sidebar switches to a warning state
instead of looking fully healthy. Long worker-capacity summaries wrap inside the
sidebar footer so the bottom-left controls stay contained.

When an optional Windows worker switch is scaled to `0`, `/health` treats that
worker as intentionally disabled. Disabled optional workers are still counted in
the total worker inventory, but they do not make capacity look degraded unless
their switch is turned on and Ollama remains unreachable.

The authenticated `GET /workers` endpoint also returns a `readiness` object with
issue details and suggested next checks for switch sync mismatches, stale worker
controller heartbeats, unreachable enabled workers, and worker-control access
problems. The Settings > Workers panel shows that summary above the worker
switches.

The legacy `local-llm` namespace backend is a PVC-backed singleton used for
worker-control support. Its Kubernetes manifest uses a `Recreate` rollout and a
5-second `/health` readiness timeout because worker health checks can briefly
wait on external Windows Ollama hosts.

## Conversation Export

Saved conversations can be exported from the chat header. The export button is
enabled after a saved conversation with messages is open and downloads a
Markdown transcript.

The authenticated API also supports:

- `GET /conversations/{id}/export?format=markdown`
- `GET /conversations/{id}/export?format=json`

Both formats include the conversation title, model, sampling settings, optional
system prompt, timestamps, and ordered messages. The endpoint only returns
conversations owned by the signed-in user.

## Chat Routing Labels

Assistant replies show the model route used for that response, for example
`via mac-mini - llama3.2:3b - resident`. The backend stores that metadata with
the saved assistant message, so reopening a conversation still shows which
worker handled each reply and whether the model was already resident or loaded
on demand.

## Chat-Only Boundary

Local LLM is scoped to local model chat, model selection, conversation storage,
conversation export, optional web-research augmentation, and worker routing for
chat inference. Coding automation, repository changes, desktop tasks, external
app sign-in, browser automation, delegated execution, and command runners belong
in the separate Local Agent project.

The model and worker controls remain only to manage chat inference capacity.
They do not expose repository access, code execution, browser automation, or
agent job orchestration. Model pull/delete and worker on/off routes require an
operator identity through `LOCAL_LLM_ADMIN_USERS` or `LOCAL_LLM_ADMIN_TOKEN`.

## Login Persistence

Local LLM uses the shared projects.lan server-side SSO contract. Login and
register create or verify users in the SQLite database pointed at by
`SHARED_AUTH_DB`, then set the `projects_lan_session` cookie with `HttpOnly`,
`SameSite=Lax`, and `Path=/`. The browser does not store auth tokens in
`localStorage`; it may only cache display-only user state.

Set `SHARED_AUTH_DB` to the same SQLite file used by the other local apps. For
Docker Compose this repo mounts `/shared-auth/auth.db`; without an override,
local non-container runs default to `~/.local-webapps/auth.db`. Set
`AUTH_COOKIE_DOMAIN` or `PROJECTS_LAN_COOKIE_DOMAIN` only after browser testing
proves the parent domain is accepted. The first LAN SSO rollout relies on the
host-scoped cookie at `projects.lan` through the launchpad proxy.

The `local-llm` Kubernetes namespace uses a static `local-llm-shared-auth-nfs`
PV/PVC binding for `shared-auth-nfs`. It intentionally points at the existing
shared-auth NFS path used by the default deployment so both LAN paths read the
same users and server-side sessions.
