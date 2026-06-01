# Local LLM Handoff

## Current State

- `chris-pc-1` is registered as an optional external Ollama worker at
  `http://192.168.4.27:11434`.
- `/health` now includes a `workers` summary with total, enabled, available,
  unavailable, busy, loaded model counts, a compact readiness state, and a
  public-safe readiness issue count. The frontend sidebar renders this as model
  count plus worker readiness severity and summary. It overlays Kubernetes
  optional-worker switches, so CHRIS-PC-1/CHRIS-PC-2 scaled to `0` count as
  intentionally disabled rather than degraded capacity.
- The authenticated `/workers` response now includes a readiness summary with
  actionable issue rows for unavailable enabled workers, switch sync mismatches,
  stale controller heartbeats, and worker-control access failures. Settings >
  Workers renders the summary above the worker switch list.
- Assistant messages now store and display route metadata in the chat UI
  (`backend_name`, `model`, and `model_status`) so saved conversations show
  which worker handled each reply.
- Assistant Markdown now gets a post-render enhancement pass. Tables are wrapped
  in `.markdown-table-scroll`, media/long links are constrained to the chat
  bubble, and the same pass runs while responses stream so mobile and desktop
  rendering stay consistent.
- `chris-pc-1-ollama-switch` lives in the `local-llm` namespace and is currently
  the Kubernetes on/off switch for that PC.
- The legacy backend in the `local-llm` namespace is a PVC-backed singleton.
  Keep `k8s/local-llm/backend.yaml` on `strategy.type: Recreate`, and keep the
  `/health` readiness timeout above the Kubernetes 1-second default because the
  endpoint summarizes external worker state.
- Login tokens are stored in browser `localStorage`. The backend signs them
  with `JWT_SECRET` when set, otherwise it generates and reuses
  `/app/data/jwt_secret` on the persistent chat-data volume.
- CHRIS-PC-1 now runs the native Windows Ollama worker, not the Docker Desktop
  worker. The launcher task is `Local LLM Native Ollama CHRIS-PC-1`, and the
  watcher task is `Local LLM CHRIS-PC-1 Native Worker Controller`.
- CHRIS-PC-1 runtime files live under `C:\ProgramData\LocalLlmWorker`, and its
  model store is `C:\ProgramData\Ollama\models`.
- `llama3.2:3b` is installed on CHRIS-PC-1 and was verified with a direct
  generate request from inside the Kubernetes backend pod.
- CHRIS-PC-2 is still unavailable as of 2026-05-26: SSH on
  `chris@192.168.4.24` rejects the installed key, and Ollama is not answering on
  `http://192.168.4.24:11434`. The Kubernetes switch is desired on but actual
  off.
- A gated Code mode now exists in the main workspace for GitHub-backed code
  execution. The normal connection path is service-configured GitHub OAuth: one
  Local LLM operator saves the OAuth App Client ID/Secret in the collapsed
  Settings > Integrations service setup panel, then every Local LLM user clicks
  **Sign in with GitHub** in Code mode for the GitHub authorize redirect and
  gets their own encrypted token. The backend exposes `/github/*` and
  `/agent/*`, nginx proxies both prefixes, and the runner image lives in
  `agent-runner/`.
- `AGENT_JOBS_ENABLED` should remain `false` until the live
  `local-llm-sandbox` NetworkPolicy and canary checks pass. The UI will show the
  feature as disabled but still lets users inspect GitHub connection state,
  complete GitHub sign-in, and use the collapsed service setup panel if the
  shared OAuth App has not been configured.
- Saved conversations now have an authenticated export path:
  `GET /conversations/{id}/export?format=markdown|json`. The chat header export
  button downloads Markdown for the active saved conversation.
- Frontend runtime libraries are pinned and served from `frontend/vendor/`
  instead of public CDN URLs. The docs page now presents
  `http://localllm.lan/v1` as the primary API base URL and keeps
  `http://localhost:8001/v1` as the Compose/dev fallback.

## Safe Continuation Notes

- Do not commit kubeconfigs, service-account tokens, SSH keys, Docker auth
  files, anything under `.docker-worker/`, or any generated `data/jwt_secret`
  file.
- Do not commit `node_modules/`. If frontend runtime packages change, use
  `npm install` and `npm run vendor:frontend`, then review the refreshed
  `frontend/vendor/` files and `package-lock.json`.
- Do not commit files copied back from `C:\ProgramData\LocalLlmWorker`; that
  directory contains the worker kubeconfig plus local runtime logs.
- Do not commit GitHub OAuth Client Secrets, OAuth access tokens,
  `GITHUB_APP_PRIVATE_KEY`, GitHub installation tokens, `AGENT_SECRET_KEY`,
  webhook secrets, kube service-account tokens, or runner callback payloads that
  contain secrets. The service OAuth app credentials live in
  `github_oauth_service_configs` and are encrypted in SQLite; per-user GitHub
  access tokens live in `github_installations`.
- `GITHUB_BYPASS_TOKEN` / `GITHUB_BYPASS_TOKEN_FILE` is a temporary live-test
  escape hatch for disposable repositories when no GitHub App exists. Do not use
  it as the normal authorization model and remove the Secret/env after testing.
- Code Jobs direct-push policy is deliberately conservative: tests must pass
  before pushing to the selected base branch. No test command means the runner
  can push only an `agent/<job-id>` branch and open a PR.
- Code jobs now run a bounded multi-agent loop: implementation subagent,
  reviewer subagent, testing agent, and revision subagent. Reviewer or test
  failures can trigger up to three review/test cycles before the job fails.
- Runners receive a per-job callback token derived from `AGENT_SECRET_KEY`, not
  the global secret. The initial clone token is used only for clone; after tests
  pass, the runner asks the backend for a fresh repository access token for
  push/PR work.
- The model tool loop can list, read, search, write, inspect diff, run bounded
  shell commands inside the isolated repository workspace, and finish. Secrets
  are removed from the runner environment before shell commands execute.
- If the reviewer sees no diff, the runner now executes the configured test
  command once and passes the failing output to the revision subagent. This keeps
  no-op jobs from looping without actionable context.
- `GITHUB_ALLOWED_INSTALLATION_IDS` is required only for the legacy GitHub App
  path. OAuth-connected users do not need GitHub App env vars or the install-id
  allowlist.
- If an existing browser token was signed before this persistence change, the
  user may need to sign in once after deployment. New tokens should survive tab
  closes and backend restarts until logout or token expiry.
- Agent runner callbacks use `http://backend.default.svc.cluster.local:8000` in
  this cluster because the local-llm backend Service is named `backend`.
- If CHRIS-PC-1 appears offline, first check the native scheduled tasks and
  launch logs on that PC, then check the switch annotations:

```powershell
Get-ScheduledTask -TaskName "Local LLM Native Ollama CHRIS-PC-1","Local LLM CHRIS-PC-1 Native Worker Controller"
Get-Content C:\ProgramData\LocalLlmWorker\ollama-launch.log -Tail 40
kubectl -n local-llm get deploy chris-pc-1-ollama-switch -o yaml
```

- To reapply routing after config changes:

```powershell
kubectl apply -f .\k8s\local-llm\ollama-backends-configmap.yaml
.\scripts\deploy-live-default-app-overrides.ps1
```

- To apply the sandbox resources without enabling jobs:

```powershell
kubectl apply -f .\k8s\local-llm\agent-sandbox.yaml
```

- To refresh the CHRIS-PC-1 controller files, copy the scripts to
  `C:\Users\chris\Projects\local-llm\scripts` on that host and rerun
  `install-local-ollama-native-worker.ps1` with the CHRIS-PC-1 parameters in
  `docs/k8s-personal-workers.md`.

## Verification Notes

- Focused health unit coverage lives in `tests/test_health.py`.
- Worker readiness summary coverage also lives in `tests/test_health.py`.
- Compact sidebar health-label coverage lives in
  `tests/frontend-health-status.test.mjs`.
- Same-origin frontend asset and LAN-first docs coverage also lives in
  `tests/frontend-health-status.test.mjs`.
- Conversation export unit and route coverage lives in
  `tests/test_conversation_export.py`.
- Chat route-metadata serialization coverage lives in
  `tests/test_chat_metadata.py`.
- Chat rich-content responsiveness was manually verified against a local static
  server with desktop and mobile browser viewports.
- Agent feature coverage lives in `tests/test_agent_features.py`.
- For a live smoke check, call `http://localllm.lan/health` and confirm the
  `workers` object is present, then open the chat UI and inspect the sidebar
  health label.
- Before setting `AGENT_JOBS_ENABLED=true`, run sandbox canaries for a successful
  command, a failing command, timeout behavior, blocked kube API access, blocked
  app-secret access, and blocked Docker socket access.
- Live canary on 2026-05-21: an agent-labeled pod in `local-llm-sandbox`
  reached `https://kubernetes.default.svc` and received `401 Unauthorized`.
  That means the current live cluster did not enforce the intended kube-API
  egress block. Keep `AGENT_JOBS_ENABLED=false` until CNI/NetworkPolicy
  enforcement is fixed or an additional egress-control layer is added.
- Kubernetes NetworkPolicy cannot reliably express GitHub-by-domain egress.
  This manifest blocks private/LAN ranges and permits public TCP 443 for GitHub
  operations. Treat public egress as a known residual risk until a domain-aware
  egress proxy is added.
- Live UI smoke on 2026-05-21 verified `localllm.lan` exposes Code as a main
  workspace mode, removes the old sidebar Code Jobs button, keeps GitHub setup
  clickable, and keeps Run Code Change disabled until GitHub sign-in and sandbox
  gates are satisfied.
- Live GitHub job E2E on 2026-05-21 used temporary `GITHUB_BYPASS_TOKEN`
  wiring against disposable repo `chrischang314/local-llm-agent-e2e`, branch
  `e2e-20260520-230509`. Job `6ed523aa62f3486dbdd4096e7f98e1c6` ran model
  `qwen2.5-coder:7b`, wrote `subtract(a, b)`, passed
  `python -m unittest discover -s tests` in the runner pod, and pushed commit
  `e7f7543f8b1e64936411ebfa0d29127075d450be` directly to that branch.
- After that live proof, the sandbox job was deleted, Secret
  `local-llm-agent-bypass` was deleted, backend env was reset to
  `AGENT_JOBS_ENABLED=false`, the backend rolled out successfully, and
  `http://localllm.lan/health` still returned `backend=ok` and `ollama=ok`.
- Live worker repair on 2026-05-26 moved CHRIS-PC-1 from the brittle Docker
  Desktop path to native Ollama. From inside the live backend pod,
  `http://192.168.4.27:11434/api/generate` with `llama3.2:3b` returned
  `PC1_OK`, and `http://localllm.lan/health` reported `workers.available=2`.
