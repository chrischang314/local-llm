# Local LLM

Local LLM provides the homelab Ollama frontend/backend and Kubernetes routing
configuration for local model workers.

## Kubernetes Workers

Personal Windows PCs are integrated as optional external Ollama workers rather
than native K3s nodes. Kubernetes owns a tiny switch Deployment per PC under the
`local-llm` namespace; scaling that Deployment to `1` means "turn this PC worker
on", and scaling it to `0` means "turn it off".

Current external workers:

- `chris-pc-2`: `http://192.168.4.24:11434`, RTX 5070, large tier.
- `chris-pc-1`: `http://192.168.4.27:11434`, RTX 4060, medium tier.

See `docs/k8s-personal-workers.md` for setup, dashboard switch behavior, and
worker controller commands.

## Common Commands

```powershell
# Apply live default-namespace routing overrides.
.\scripts\deploy-live-default-app-overrides.ps1

# Turn an external Windows worker on or off locally.
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-mode.ps1 -Mode on
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-mode.ps1 -Mode off

# Run one worker-controller reconciliation.
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-controller.ps1 -Once

# Run the Kubernetes agent-runner canary.
powershell -ExecutionPolicy Bypass -File .\scripts\agent-runner-canary.ps1
```

The worker scripts use `.docker-worker/config.json` for unattended Docker calls.
That directory is intentionally ignored by git.

## Health Status

`GET /health` returns the backend state, Ollama model count, and a compact
worker summary:

- `workers.enabled`: configured workers that are allowed to receive traffic.
- `workers.available`: enabled workers that answered their Ollama health check.
- `workers.busy`: requests currently routed through workers.
- `workers.loaded_model_count`: resident models across all configured workers.

The chat sidebar uses the same endpoint so the at-a-glance status shows both
model count and available worker capacity. Long worker-capacity messages wrap
inside the sidebar footer instead of widening or clipping the bottom-left
controls.

## Web Research

Chat can optionally add live web research before calling Ollama. Turn on the
Research toggle in the chat topbar for the next question that needs current
facts. When enabled, the backend searches the web and adds a dated research
brief to the model context. The toggle resets after the message sends so later
prompts do not leave the LAN unless Research is turned on again. The saved
conversation still contains only the user message and assistant answer. When
sources are retrieved, the backend appends a `Sources` footer with full URLs
after the model answer.

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
  snippets and source URLs are still used.
- `WEB_RESEARCH_MAX_CONTEXT_CHARS`: prompt budget for retrieved context.

`GET /research/status` returns only non-secret feature status for the signed-in
UI. Do not add API keys, tokens, cookies, or browser storage to git if another
research provider is added later.

## Code Jobs Workspace

The frontend includes a Code Jobs workspace alongside chat. It keeps the same
same-origin API pattern as the rest of the app:

- `GET /github/status` checks whether the service GitHub OAuth app is configured
  and whether the current Local LLM user has signed in with GitHub.
- `POST /github/oauth/config` saves the one-time service OAuth Client ID and
  Client Secret in the local SQLite database, encrypted with the persistent app
  signing key. End users do not enter these fields after setup.
- `POST /github/oauth/start` returns a GitHub authorization URL for the current
  Local LLM user. The UI redirects there, GitHub shows its normal authorize
  page, and `/github/oauth/callback` stores that user's encrypted GitHub token.
- `GET /github/repositories` fills repository pickers when that optional route
  is available for the signed-in GitHub account.
- `GET /agent/jobs` lists queued, running, and completed coding jobs.
- `POST /agent/jobs` starts a job with repository, branch, mode, instructions,
  and options for tests and pull request creation.
- `GET /agent/jobs/{id}` refreshes job detail, logs, and links.

nginx proxies `/github` and `/agent` to the backend. If those backend
routes are not deployed yet, the workspace shows an unavailable state and chat
continues to work normally.

To configure the OAuth app, create a GitHub OAuth App with callback URL
`http://localllm.lan/github/oauth/callback` (or the callback shown in the UI).
Save its Client ID and Client Secret once in the Code Jobs GitHub setup panel.
After that, each Local LLM user clicks **Sign in with GitHub**, authorizes on
github.com, and returns signed in on their own browser session.

## Agent Runner Sandbox

The Kubernetes-only agent runner lives under `agent-runner/` and exposes an
internal `POST /runs` API for bounded command execution. The runner requires a
bearer token in cluster, rejects shell/path-based commands, keeps work under
`/workspace`, truncates large output, and deploys with restricted pod security
plus a deny-egress NetworkPolicy.

Apply `k8s/local-llm/agent-runner.yaml` after creating the
`local-llm-agent-runner-auth` secret. See `docs/agent-runner-sandbox.md` for the
secret command, canary script, and backend client usage.

## Login Persistence

The browser stores the active login token in `localStorage`, so closing and
reopening a tab should keep the user signed in until they click log out or the
30-day token expires.

Backend restarts keep accepting existing tokens because the JWT signing key is
persisted at `/app/data/jwt_secret` by default. `JWT_SECRET` still takes
priority when set, and `JWT_SECRET_FILE` can point the generated key somewhere
else. Do not delete the chat data volume if you want existing sessions to stay
valid across redeploys.
