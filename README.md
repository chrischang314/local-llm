# Local LLM

Local LLM provides the homelab Ollama frontend/backend and Kubernetes routing
configuration for local model workers.

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
```

The Docker worker scripts use `.docker-worker/config.json` for unattended Docker
calls. The native worker installer writes runtime files under
`C:\ProgramData\LocalLlmWorker`. Both locations are local machine state and must
stay out of git.

## Health Status

`GET /health` returns the backend state, Ollama model count, and a compact
worker summary:

- `workers.enabled`: configured workers that are allowed to receive traffic.
- `workers.available`: enabled workers that answered their Ollama health check.
- `workers.busy`: requests currently routed through workers.
- `workers.loaded_model_count`: resident models across all configured workers.

The chat sidebar uses the same endpoint so the at-a-glance status shows both
model count and available worker capacity.

## Code Change Mode And GitHub Integration

The main workspace includes a gated Code mode for authenticated users. It can
connect to GitHub from the website, list authorized repositories and branches,
and queue agentic coding jobs that run inside isolated Kubernetes Jobs.

The normal connection path is GitHub OAuth. Configure one service OAuth App in
Code mode using the displayed callback URL
(`http://localllm.lan/github/oauth/callback` on the LAN deployment). Local LLM
stores that Client ID and Client Secret encrypted in SQLite. After that, every
Local LLM user gets a normal **Sign in with GitHub** button: it redirects to
github.com, the user authorizes there, and the callback stores that user's
encrypted GitHub token for repository access. End users do not paste OAuth app
keys.

Live code execution is disabled by default. Enable it only after the sandbox
namespace, NetworkPolicy enforcement, and canary tests pass:

```powershell
kubectl apply -f .\k8s\local-llm\agent-sandbox.yaml
```

Required backend environment variables for live execution:

- `AGENT_SECRET_KEY`
- `AGENT_JOBS_ENABLED=true` after sandbox validation

The older GitHub App installation path remains supported for deployments that
want installation tokens. That legacy path still uses `GITHUB_APP_ID`,
`GITHUB_APP_SLUG`, `GITHUB_APP_PRIVATE_KEY` or `GITHUB_APP_PRIVATE_KEY_FILE`,
and `GITHUB_ALLOWED_INSTALLATION_IDS`. OAuth-connected users do not need those
GitHub App env vars.

For a controlled live smoke test before a GitHub App exists, the backend also
recognizes `GITHUB_BYPASS_TOKEN` or `GITHUB_BYPASS_TOKEN_FILE`. This treats a
single Kubernetes Secret-backed GitHub token as the installation token source and
should be temporary, LAN-only, and limited to disposable repositories. The normal
production path remains the site-driven OAuth redirect flow above.

The runner image is built from `agent-runner/` and published by GitHub Actions as
`ghcr.io/<owner>/<repo>/agent-runner`. Each job gets a GitHub repository access
token from the connected OAuth account or legacy GitHub App installation, clones
the selected repository, creates an `agent/<job-id>` branch, then runs a bounded
multi-agent quality loop:

1. an implementation subagent drafts the change,
2. a reviewer subagent checks correctness, scope, maintainability, and safety,
3. a testing agent runs the configured test command,
4. a revision subagent fixes reviewer or test failures, repeating up to three
   review/test cycles.

The model tool loop can read/search/write files, inspect diffs, and run bounded
shell commands inside the isolated repository workspace. If no diff is present,
the reviewer step runs the configured test command once to give the revision
subagent concrete failure output. The runner requests a fresh installation token
or OAuth repository token for push/PR operations only after the reviewer and testing gates are satisfied.
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
