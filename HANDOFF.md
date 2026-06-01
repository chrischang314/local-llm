# Local LLM Handoff

## Current State

- The agent runner sandbox is implemented under `agent-runner/`, with a
  Kubernetes deployment/service/network policy at
  `k8s/local-llm/agent-runner.yaml`.
- `backend/agent_executor_client.py` is ready for backend-owned agent endpoints;
  no frontend UI was changed by Worker 2.
- `chris-pc-1` is registered as an optional external Ollama worker at
  `http://192.168.4.27:11434`.
- `/health` now includes a `workers` summary with total, enabled, available,
  busy, and loaded model counts. The frontend sidebar renders this as
  model count plus worker availability, with long capacity warnings wrapped
  inside the footer so the bottom-left controls stay contained.
- `chris-pc-1-ollama-switch` lives in the `local-llm` namespace and is currently
  the Kubernetes on/off switch for that PC.
- Login tokens are stored in browser `localStorage`. The backend signs them
  with `JWT_SECRET` when set, otherwise it generates and reuses
  `/app/data/jwt_secret` on the persistent chat-data volume.
- Chat now has optional web research. The frontend Research toggle sends
  `web_research: true` for one message and then resets. The backend retrieves
  public search snippets with `backend/research_client.py`, injects them as
  system context, and returns `X-Research-Status` plus
  `X-Research-Source-Count` headers. When sources are found, the backend appends
  a `Sources` footer with full URLs to the answer.
- GitHub now uses a service-style OAuth web flow. The service OAuth Client ID
  and Client Secret are entered once through the Code Jobs setup UI, stored in
  `github_oauth_service_configs`, and encrypted via `backend/secret_store.py`.
  Each Local LLM user then clicks **Sign in with GitHub** and gets their own
  encrypted token in `github_installations`.
- The frontend now has a separate Code Jobs workspace. It is frontend-only and
  calls same-origin `/github/*` and `/agent/jobs*` routes when those backend
  routes are available.
- The CHRIS-PC-1 controller runs as the Windows scheduled task
  `Local LLM CHRIS-PC-1 Worker Controller`.
- Docker Desktop is started at logon by the `Start Docker Desktop` scheduled
  task so the controller can use the Docker named pipe from the interactive
  Windows session.
- `llama3.2:3b` is installed on CHRIS-PC-1 and was verified with a direct
  generate request from inside the Kubernetes backend pod.

## Safe Continuation Notes

- Do not commit kubeconfigs, service-account tokens, SSH keys, Docker auth
  files, anything under `.docker-worker/`, or any generated `data/jwt_secret`
  file.
- Do not commit GitHub OAuth Client Secrets or encrypted local database files.
  The callback URL must match the GitHub OAuth App registration; for the LAN
  deployment it should normally be
  `http://localllm.lan/github/oauth/callback`.
- Do not commit research provider API keys, cookies, search session state, or
  exported browser storage if a provider beyond the no-key DuckDuckGo HTML
  fetcher is added later. Keep tests on mocked transports. Result page fetching
  is off by default; if `WEB_RESEARCH_FETCH_PAGES=true` is enabled, preserve the
  private-address and redirect safety tests.
- Do not commit the agent-runner bearer token. Create or mirror the
  `local-llm-agent-runner-auth` secret per namespace as runtime state.
- Before enabling backend routes that call the runner, make sure
  `AGENT_RUNNER_URL` and `AGENT_RUNNER_TOKEN` are set for that backend
  Deployment and then run `scripts/agent-runner-canary.ps1`.
- If an existing browser token was signed before this persistence change, the
  user may need to sign in once after deployment. New tokens should survive tab
  closes and backend restarts until logout or token expiry.
- GitHub status and OAuth routes intentionally stay available even when
  `AGENT_JOBS_ENABLED=false`; `/agent/jobs` and repository dispatch remain
  feature-gated.
- Backend workers should provide `GET /github/status`,
  `POST /github/oauth/config`, `POST /github/oauth/start`,
  `GET /github/oauth/callback`, `GET /github/repositories`, plus
  `GET /agent/jobs`, `POST /agent/jobs`, and `GET /agent/jobs/{id}` for the new
  workspace.
- Keep frontend calls same-origin. The nginx config now proxies `/github` and
  `/agent` alongside existing chat/model routes.
- If CHRIS-PC-1 appears offline, first check Docker Desktop and the scheduled
  tasks on that PC, then check the switch annotations:

```powershell
kubectl -n local-llm get deploy chris-pc-1-ollama-switch -o yaml
```

- To reapply routing after config changes:

```powershell
kubectl apply -f .\k8s\local-llm\ollama-backends-configmap.yaml
.\scripts\deploy-live-default-app-overrides.ps1
```

- To refresh the CHRIS-PC-1 controller files, copy the scripts to
  `C:\Users\chris\Projects\local-llm\scripts` on that host and rerun
  `install-local-ollama-worker-controller.ps1` with the CHRIS-PC-1 parameters in
  `docs/k8s-personal-workers.md`.

## Verification Notes

- Focused health unit coverage lives in `tests/test_health.py`.
- Focused web research coverage lives in `tests/test_web_research.py`. It uses
  `httpx.MockTransport` and should not perform live internet calls.
- Runner unit coverage lives in `tests/test_agent_runner.py` and
  `tests/test_agent_executor_client.py`.
- Runner live smoke coverage is documented in
  `docs/agent-runner-sandbox.md` and automated by
  `scripts/agent-runner-canary.ps1`.
- For the Code Jobs workspace, verify both the unavailable state when backend
  routes are missing and the happy path once the backend exposes `/github/*` and
  `/agent/jobs*`.
- For GitHub OAuth, verify the unconfigured setup panel, configured
  "Sign in with GitHub" button, redirect URL generation, callback handling, and
  per-user connection status. Unit coverage lives in
  `tests/test_agent_jobs_api.py`.
- For a live smoke check, call `http://localllm.lan/health` and confirm the
  `workers` object is present, then open the chat UI and inspect the sidebar
  health label. Include a long/degraded worker-capacity label when checking the
  footer layout.
- For web research smoke, sign in, verify `GET /research/status`, turn on the
  topbar Research toggle, send a current-events style prompt, and confirm the
  stream briefly shows research status before the model answer.
