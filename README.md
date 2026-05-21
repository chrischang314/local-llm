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
model count and available worker capacity.

## Code Change Mode And GitHub App Integration

The main workspace includes a gated Code mode for authenticated users. It can
connect a GitHub App installation, list installed repositories and branches, and
queue agentic coding jobs that run inside isolated Kubernetes Jobs. The GitHub
button remains clickable when the backend is not configured so the UI can show
which required settings are missing instead of presenting a dead disabled state.

Live code execution is disabled by default. Enable it only after the sandbox
namespace, NetworkPolicy enforcement, and canary tests pass:

```powershell
kubectl apply -f .\k8s\local-llm\agent-sandbox.yaml
```

Required backend environment variables:

- `GITHUB_APP_ID`
- `GITHUB_APP_SLUG`
- `GITHUB_APP_PRIVATE_KEY` or `GITHUB_APP_PRIVATE_KEY_FILE`
- `AGENT_SECRET_KEY`
- `GITHUB_ALLOWED_INSTALLATION_IDS`
- `AGENT_JOBS_ENABLED=true` after sandbox validation

`GITHUB_ALLOWED_INSTALLATION_IDS` is a comma-separated allowlist. The app may
show GitHub connection status without it, but live job creation is blocked until
the connected installation id is explicitly allowed.

For a controlled live smoke test before a GitHub App exists, the backend also
recognizes `GITHUB_BYPASS_TOKEN` or `GITHUB_BYPASS_TOKEN_FILE`. This treats a
single Kubernetes Secret-backed GitHub token as the installation token source and
should be temporary, LAN-only, and limited to disposable repositories. The normal
production path remains the GitHub App installation flow above.

The runner image is built from `agent-runner/` and published by GitHub Actions as
`ghcr.io/<owner>/<repo>/agent-runner`. Each job gets a short-lived GitHub App
installation token, clones the selected repository, creates an `agent/<job-id>`
branch, then runs a bounded multi-agent quality loop:

1. an implementation subagent drafts the change,
2. a reviewer subagent checks correctness, scope, maintainability, and safety,
3. a testing agent runs the configured test command,
4. a revision subagent fixes reviewer or test failures, repeating up to three
   review/test cycles.

The model tool loop can read/search/write files, inspect diffs, and run bounded
shell commands inside the isolated repository workspace. If no diff is present,
the reviewer step runs the configured test command once to give the revision
subagent concrete failure output. The runner requests a fresh installation token
for push/PR operations only after the reviewer and testing gates are satisfied.
If the branch is protected or the push diverges, the runner pushes the agent
branch and creates a PR instead. If no test command is supplied, it must not
update the base branch.

## Login Persistence

The browser stores the active login token in `localStorage`, so closing and
reopening a tab should keep the user signed in until they click log out or the
30-day token expires.

Backend restarts keep accepting existing tokens because the JWT signing key is
persisted at `/app/data/jwt_secret` by default. `JWT_SECRET` still takes
priority when set, and `JWT_SECRET_FILE` can point the generated key somewhere
else. Do not delete the chat data volume if you want existing sessions to stay
valid across redeploys.
