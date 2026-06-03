# Local LLM Handoff

## Current State

- `chris-pc-1` is registered as an optional external Ollama worker at
  `http://192.168.4.27:11434`.
- `/health` now includes a `workers` summary with total, enabled, available,
  unavailable, busy, loaded model counts, a compact readiness state, and a
  public-safe readiness issue count. The frontend sidebar renders this as model
  count plus worker readiness severity and summary, with long warnings wrapped
  inside the footer so the bottom-left controls stay contained. It overlays
  Kubernetes optional-worker switches, so CHRIS-PC-1/CHRIS-PC-2 scaled to `0`
  count as intentionally disabled rather than degraded capacity.
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
- Login/register now use the shared projects.lan SSO contract. Users and
  server-side sessions live in `SHARED_AUTH_DB`, and browsers authenticate with
  the HttpOnly `projects_lan_session` cookie (`SameSite=Lax`, `Path=/`).
  `localStorage` may only keep display state; it is not auth authority.
  The `local-llm` namespace backend binds `shared-auth-nfs` through the static
  `local-llm-shared-auth-nfs` PV so it uses the same NFS auth DB as the default
  deployment; do not replace it with a dynamically provisioned empty claim.
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
- Local LLM is chat-only. Coding automation, repository mutation, external app
  sign-in, desktop/browser execution, command runners, and broader delegated
  workflows have moved to the separate Local Agent project. Model and worker
  controls remain only as operator-gated chat inference infrastructure controls.
- Saved conversations now have an authenticated export path:
  `GET /conversations/{id}/export?format=markdown|json`. The chat header export
  button downloads Markdown for the active saved conversation.
- Frontend runtime libraries are pinned and served from `frontend/vendor/`
  instead of public CDN URLs. `scripts/frontend-vendor-assets.json` is the
  shared manifest used by the copy script and static tests. The docs page now
  presents `http://localllm.lan/v1` as the primary API base URL and keeps
  `http://localhost:8001/v1` as the Compose/dev fallback.
- Chat now has optional web research. The composer Research toggle sends
  `web_research: true` for one message and then resets. The backend retrieves
  public search snippets with `backend/research_client.py`, injects them as
  system context, and returns `X-Research-Status` plus
  `X-Research-Source-Count` headers. When sources are found, the backend appends
  a `Sources` footer with full URLs to the answer.

## Safe Continuation Notes

- Do not commit kubeconfigs, service-account tokens, SSH keys, Docker auth
  files, anything under `.docker-worker/`, or any generated `data/jwt_secret`
  file.
- Do not commit `node_modules/`. If frontend runtime packages change, update
  `scripts/frontend-vendor-assets.json`, use `npm install` and
  `npm run vendor:frontend`, then review the refreshed `frontend/vendor/`
  files and `package-lock.json`.
- Do not commit files copied back from `C:\ProgramData\LocalLlmWorker`; that
  directory contains the worker kubeconfig plus local runtime logs.
- Do not commit research provider API keys, cookies, search session state, or
  exported browser storage if a provider beyond the no-key HTML search fetchers
  is added later. Keep tests on mocked transports. Result page fetching is off
  by default; if `WEB_RESEARCH_FETCH_PAGES=true` is enabled, preserve the
  private-address, redirect, and hostname/DNS-rebinding safety tests. The
  fetcher keeps normal hostname results as citations, but page-body excerpt
  fetches are restricted to literal public IP URLs.
- Existing JWT/localStorage browser sessions will need to sign in again.
  Existing local users can still log in with their old password once; the
  backend verifies the legacy bcrypt hash and seeds the shared auth DB.
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
- Focused web research coverage lives in `tests/test_web_research.py`. It uses
  `httpx.MockTransport` and should not perform live internet calls.
- Chat-only boundary coverage lives in `tests/test_chat_only_boundary.py`.
- For a live smoke check, call `http://localllm.lan/health` and confirm the
  `workers` object is present, then open the chat UI and inspect the sidebar
  health label with a long/degraded worker-capacity summary.
- Live worker repair on 2026-05-26 moved CHRIS-PC-1 from the brittle Docker
  Desktop path to native Ollama. From inside the live backend pod,
  `http://192.168.4.27:11434/api/generate` with `llama3.2:3b` returned
  `PC1_OK`, and `http://localllm.lan/health` reported `workers.available=2`.
